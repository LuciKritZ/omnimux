import asyncio
import datetime
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import aiosqlite

from omnimux.config import settings
from omnimux.exceptions import OmniMuxError

logger = logging.getLogger(__name__)


class NoKeysAvailableError(OmniMuxError):
    """Raised when no API keys are available (all exhausted/cooldown)."""


class RouterCore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        # Concurrency semaphores and active in-flight counters per key
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._in_flight: dict[str, int] = {}
        # In-memory cooldown cache to avoid hitting DB constantly for expired keys
        self._cooldowns: dict[str, float] = {}
        # Round-robin counter per platform/selection
        self._rr_index: dict[str, int] = {}
        # Cached concurrency limit per key
        self._key_concurrency_limits: dict[str, int] = {}
        # Session affinity cache: session_id -> (platform, key_hash, expires_at)
        self._session_affinity: dict[str, tuple[str, str, float]] = {}

    def get_session_affinity(self, session_id: str) -> tuple[str, str, float] | None:
        """Retrieves active session affinity if not expired."""
        if session_id not in self._session_affinity:
            return None
        platform, key_hash, expires_at = self._session_affinity[session_id]
        now = datetime.datetime.now(datetime.UTC).timestamp()
        if now > expires_at:
            self._session_affinity.pop(session_id, None)
            return None
        return platform, key_hash, expires_at

    def set_session_affinity(
        self,
        session_id: str,
        platform: str,
        key_hash: str,
        ttl_seconds: float | None = None,
    ) -> None:
        """Binds a session to a specific platform and key_hash with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else settings.session_affinity_ttl_seconds
        now = datetime.datetime.now(datetime.UTC).timestamp()
        self._session_affinity[session_id] = (platform, key_hash, now + ttl)

    def evict_session_affinity(self, session_id: str) -> None:
        """Evicts a specific session affinity."""
        self._session_affinity.pop(session_id, None)

    def evict_affinities_for_key(self, key_hash: str) -> None:
        """Evicts all session affinities bound to a specific key_hash."""
        expired = [s for s, (_, kh, _) in self._session_affinity.items() if kh == key_hash]
        for s in expired:
            self._session_affinity.pop(s, None)

    def get_key_concurrency_limit(self, key_hash: str, meta: Any = None) -> int:
        """Determines the maximum concurrency limit for a key from meta or settings."""
        if key_hash in self._key_concurrency_limits and meta is None:
            return self._key_concurrency_limits[key_hash]

        limit = settings.default_key_concurrency
        if meta:
            parsed_meta: dict[str, Any] = {}
            if isinstance(meta, str):
                try:
                    parsed_meta = json.loads(meta)
                except Exception:
                    parsed_meta = {}
            elif isinstance(meta, dict):
                parsed_meta = meta

            val = parsed_meta.get("max_concurrency") or parsed_meta.get("concurrency_limit")
            if val is not None:
                with suppress(ValueError, TypeError):
                    limit = int(val)

        self._key_concurrency_limits[key_hash] = limit
        return limit

    def _get_semaphore(self, key_hash: str, limit: int | None = None) -> asyncio.Semaphore:
        if key_hash not in self._semaphores:
            effective_limit = (
                limit if limit is not None else self.get_key_concurrency_limit(key_hash)
            )
            self._semaphores[key_hash] = asyncio.Semaphore(effective_limit)
        return self._semaphores[key_hash]

    def get_in_flight_count(self, key_hash: str) -> int:
        return self._in_flight.get(key_hash, 0)

    def is_key_busy(self, key_hash: str) -> bool:
        return self.get_in_flight_count(key_hash) > 0

    def is_key_at_capacity(self, key_hash: str, limit: int | None = None) -> bool:
        effective_limit = limit if limit is not None else self.get_key_concurrency_limit(key_hash)
        return self.get_in_flight_count(key_hash) >= effective_limit

    async def get_next_key(self, platform: str, session_id: str | None = None) -> dict[str, Any]:
        """
        Retrieves the best available key for a given platform.
        Prioritizes session affinity, then highest `priority`, then round-robin / least-loaded.
        Skips keys currently in cooldown (try_after).
        """
        now = datetime.datetime.now(datetime.UTC).timestamp()

        async with self.db.execute(
            """
            SELECT key_hash, key_value, priority, try_after, last_used_at, meta
            FROM api_keys
            WHERE platform = ?
            ORDER BY priority DESC, last_used_at ASC
            """,
            (platform,),
        ) as cursor:
            rows = list(await cursor.fetchall())

        if not rows:
            raise NoKeysAvailableError(f"No keys configured for platform: {platform}")

        valid_rows = []
        for row in rows:
            key_hash, _, _, try_after, _, meta = row

            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            if try_after:
                try_after_ts = datetime.datetime.fromisoformat(try_after).timestamp()
                if try_after_ts > now:
                    self._cooldowns[key_hash] = try_after_ts
                    continue
                else:
                    self._cooldowns.pop(key_hash, None)
                    await self.db.execute(
                        "UPDATE api_keys SET try_after = NULL WHERE key_hash = ?", (key_hash,)
                    )
                    await self.db.commit()

            self.get_key_concurrency_limit(key_hash, meta)
            valid_rows.append(row)

        if not valid_rows:
            raise NoKeysAvailableError(
                f"All keys for {platform} are currently exhausted or in cooldown."
            )

        if session_id:
            affinity = self.get_session_affinity(session_id)
            if affinity:
                aff_platform, aff_key_hash, _ = affinity
                bound_row = next(
                    (r for r in valid_rows if aff_platform == platform and r[0] == aff_key_hash),
                    None,
                )
                if bound_row:
                    self.set_session_affinity(session_id, platform, aff_key_hash)
                    return {
                        "key_hash": bound_row[0],
                        "key_value": bound_row[1],
                        "priority": bound_row[2],
                        "meta": bound_row[5],
                    }
                else:
                    self.evict_session_affinity(session_id)

        rr_offset = self._rr_index.get(platform, 0)

        # Pass 1: select from idle keys (in_flight == 0)
        idle_rows = [r for r in valid_rows if not self.is_key_busy(r[0])]
        if idle_rows:
            highest_priority = max(r[2] for r in idle_rows)
            best_idle = [r for r in idle_rows if r[2] == highest_priority]
            best_idle.sort(key=lambda r: r[4] or "")
            oldest_ts = best_idle[0][4] or ""
            same_ts = [r for r in best_idle if (r[4] or "") == oldest_ts]
            selected = same_ts[rr_offset % len(same_ts)]
            self._rr_index[platform] = rr_offset + 1
            if session_id:
                self.set_session_affinity(session_id, platform, selected[0])
            return {
                "key_hash": selected[0],
                "key_value": selected[1],
                "priority": selected[2],
                "meta": selected[5],
            }

        # Pass 2: fallback to busy keys (least-loaded under capacity first)
        under_capacity = [r for r in valid_rows if not self.is_key_at_capacity(r[0])]
        candidates = under_capacity if under_capacity else valid_rows

        highest_priority = max(r[2] for r in candidates)
        best_candidates = [r for r in candidates if r[2] == highest_priority]
        best_candidates.sort(key=lambda r: (self.get_in_flight_count(r[0]), r[4] or ""))

        min_load = self.get_in_flight_count(best_candidates[0][0])
        min_load_candidates = [
            r for r in best_candidates if self.get_in_flight_count(r[0]) == min_load
        ]
        oldest_ts = min_load_candidates[0][4] or ""
        same_load_ts = [r for r in min_load_candidates if (r[4] or "") == oldest_ts]
        selected = same_load_ts[rr_offset % len(same_load_ts)]
        self._rr_index[platform] = rr_offset + 1
        if session_id:
            self.set_session_affinity(session_id, platform, selected[0])

        return {
            "key_hash": selected[0],
            "key_value": selected[1],
            "priority": selected[2],
            "meta": selected[5],
        }

    @asynccontextmanager
    async def use_key(self, key_hash: str, limit: int | None = None) -> AsyncGenerator[None, None]:
        """Context manager to track active in-flight requests and update last_used_at."""
        sem = self._get_semaphore(key_hash, limit)
        async with sem:
            self._in_flight[key_hash] = self._in_flight.get(key_hash, 0) + 1
            try:
                yield
            finally:
                self._in_flight[key_hash] = max(0, self._in_flight.get(key_hash, 1) - 1)
                await self.mark_key_used(key_hash)

    async def get_best_platform_and_key(
        self, candidate_platforms: list[str], session_id: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Retrieves the best available platform and key from a list of candidates.
        Prioritizes session affinity, then highest `priority`, caller order, least in-flight load,
        and round-robin.
        """
        now_dt = datetime.datetime.now(datetime.UTC)
        now = now_dt.timestamp()

        if not candidate_platforms:
            raise NoKeysAvailableError("No candidate platforms provided for dynamic routing")

        placeholders = ",".join("?" for _ in candidate_platforms)
        query = f"""
            SELECT platform, key_hash, key_value, priority, try_after, last_used_at, meta
            FROM api_keys
            WHERE platform IN ({placeholders})
        """

        async with self.db.execute(query, candidate_platforms) as cursor:
            rows = list(await cursor.fetchall())

        if not rows:
            raise NoKeysAvailableError(
                f"No keys configured for candidate platforms: {candidate_platforms}"
            )

        valid_rows = []
        for row in rows:
            _, key_hash, _, _, try_after, _, meta = row

            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            if try_after:
                try_after_ts = datetime.datetime.fromisoformat(try_after).timestamp()
                if try_after_ts > now:
                    self._cooldowns[key_hash] = try_after_ts
                    continue
                else:
                    self._cooldowns.pop(key_hash, None)
                    await self.db.execute(
                        "UPDATE api_keys SET try_after = NULL WHERE key_hash = ?", (key_hash,)
                    )
                    await self.db.commit()

            self.get_key_concurrency_limit(key_hash, meta)
            valid_rows.append(row)

        if not valid_rows:
            raise NoKeysAvailableError(
                f"All keys for candidate platforms {candidate_platforms} are in cooldown."
            )

        if session_id:
            affinity = self.get_session_affinity(session_id)
            if affinity:
                aff_platform, aff_key_hash, _ = affinity
                bound_row = next(
                    (r for r in valid_rows if r[0] == aff_platform and r[1] == aff_key_hash),
                    None,
                )
                if bound_row:
                    self.set_session_affinity(session_id, aff_platform, aff_key_hash)
                    await self.db.execute(
                        "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                        (now_dt.isoformat(), aff_key_hash),
                    )
                    await self.db.commit()
                    return aff_platform, {
                        "key_hash": bound_row[1],
                        "key_value": bound_row[2],
                        "meta": bound_row[6],
                    }
                else:
                    self.evict_session_affinity(session_id)

        platform_order = {p: i for i, p in enumerate(candidate_platforms)}

        # Pass 1: idle keys (in_flight == 0)
        idle_rows = [r for r in valid_rows if not self.is_key_busy(r[1])]
        if idle_rows:
            idle_rows.sort(key=lambda r: (-r[3], platform_order.get(r[0], 9999), r[5] or ""))
            top = idle_rows[0]
            top_rank_candidates = [
                r
                for r in idle_rows
                if r[3] == top[3]
                and platform_order.get(r[0], 9999) == platform_order.get(top[0], 9999)
            ]
            oldest_ts = top_rank_candidates[0][5] or ""
            same_ts = [r for r in top_rank_candidates if (r[5] or "") == oldest_ts]
            rr_offset = self._rr_index.get(top[0], 0)
            selected = same_ts[rr_offset % len(same_ts)]
            self._rr_index[top[0]] = rr_offset + 1

            await self.db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (now_dt.isoformat(), selected[1]),
            )
            await self.db.commit()

            if session_id:
                self.set_session_affinity(session_id, selected[0], selected[1])

            return selected[0], {
                "key_hash": selected[1],
                "key_value": selected[2],
                "meta": selected[6],
            }

        # Pass 2: fallback to busy keys (least-loaded under capacity first)
        under_capacity = [r for r in valid_rows if not self.is_key_at_capacity(r[1])]
        candidates = under_capacity if under_capacity else valid_rows

        candidates.sort(
            key=lambda r: (
                -r[3],
                platform_order.get(r[0], 9999),
                self.get_in_flight_count(r[1]),
                r[5] or "",
            )
        )
        top = candidates[0]
        top_candidates = [
            r
            for r in candidates
            if r[3] == top[3]
            and platform_order.get(r[0], 9999) == platform_order.get(top[0], 9999)
            and self.get_in_flight_count(r[1]) == self.get_in_flight_count(top[1])
        ]
        oldest_ts = top_candidates[0][5] or ""
        same_ts = [r for r in top_candidates if (r[5] or "") == oldest_ts]
        rr_offset = self._rr_index.get(top[0], 0)
        selected = same_ts[rr_offset % len(same_ts)]
        self._rr_index[top[0]] = rr_offset + 1

        await self.db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (now_dt.isoformat(), selected[1]),
        )
        await self.db.commit()

        if session_id:
            self.set_session_affinity(session_id, selected[0], selected[1])

        return selected[0], {
            "key_hash": selected[1],
            "key_value": selected[2],
            "meta": selected[6],
        }

    async def mark_key_exhausted(self, key_hash: str, duration_seconds: float) -> None:
        """Puts a key in cooldown for a specified duration and evicts any session affinities."""
        now = datetime.datetime.now(datetime.UTC)
        try_after = now + datetime.timedelta(seconds=duration_seconds)
        self._cooldowns[key_hash] = try_after.timestamp()
        self.evict_affinities_for_key(key_hash)

        await self.db.execute(
            "UPDATE api_keys SET try_after = ? WHERE key_hash = ?",
            (try_after.isoformat(), key_hash),
        )
        await self.db.commit()
        logger.warning(f"Key {key_hash} exhausted. Cooldown until {try_after.isoformat()}")

    async def mark_key_used(self, key_hash: str) -> None:
        """Updates the last_used_at timestamp for a key."""
        now = datetime.datetime.now(datetime.UTC)
        await self.db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (now.isoformat(), key_hash),
        )
        await self.db.commit()
