import asyncio
import datetime
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

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

    def _get_semaphore(self, key_hash: str, limit: int = 10) -> asyncio.Semaphore:
        if key_hash not in self._semaphores:
            self._semaphores[key_hash] = asyncio.Semaphore(limit)
        return self._semaphores[key_hash]

    def is_key_busy(self, key_hash: str) -> bool:
        return self._in_flight.get(key_hash, 0) > 0

    async def get_next_key(self, platform: str) -> dict[str, Any]:
        """
        Retrieves the best available key for a given platform.
        Prioritizes highest `priority`, then oldest `last_used_at`.
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
            rows = await cursor.fetchall()

        if not rows:
            raise NoKeysAvailableError(f"No keys configured for platform: {platform}")

        for row in rows:
            key_hash, key_value, priority, try_after, _, meta = row

            # Check in-memory fast path cooldown
            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            # DB level cooldown check
            if try_after:
                try_after_ts = datetime.datetime.fromisoformat(try_after).timestamp()
                if try_after_ts > now:
                    self._cooldowns[key_hash] = try_after_ts
                    continue
                else:
                    # Cooldown expired, clear it
                    self._cooldowns.pop(key_hash, None)
                    await self.db.execute(
                        "UPDATE api_keys SET try_after = NULL WHERE key_hash = ?", (key_hash,)
                    )
                    await self.db.commit()

            # If we get here, the key is not in cooldown.
            # Check if the key is currently busy. Skip to next to maximize concurrency.
            if self.is_key_busy(key_hash):
                continue

            return {
                "key_hash": key_hash,
                "key_value": key_value,
                "priority": priority,
                "meta": meta,
            }

        # Second pass: if all unlocked keys are in cooldown, can we wait on a busy key?
        # We return the first non-cooldown key, even if busy.
        for row in rows:
            key_hash, key_value, priority, try_after, _, meta = row
            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            return {
                "key_hash": key_hash,
                "key_value": key_value,
                "priority": priority,
                "meta": meta,
            }

        raise NoKeysAvailableError(
            f"All keys for {platform} are currently exhausted or in cooldown."
        )

    @asynccontextmanager
    async def use_key(self, key_hash: str) -> AsyncGenerator[None, None]:
        """Context manager to track active in-flight requests and update last_used_at."""
        sem = self._get_semaphore(key_hash)
        async with sem:
            self._in_flight[key_hash] = self._in_flight.get(key_hash, 0) + 1
            try:
                yield
            finally:
                self._in_flight[key_hash] = max(0, self._in_flight.get(key_hash, 1) - 1)
                await self.mark_key_used(key_hash)

    async def get_best_platform_and_key(
        self, candidate_platforms: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """
        Retrieves the best available platform and key from a list of candidates.
        Prioritizes highest `priority`, then caller candidate order, then oldest `last_used_at`.
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

        # Prioritize candidates by caller's specified preference order
        # before sorting by last_used_at
        platform_order = {p: i for i, p in enumerate(candidate_platforms)}
        rows.sort(key=lambda r: (-r[3], platform_order.get(r[0], 9999), r[5] or ""))

        # Pass 1: skip keys in cooldown and busy keys
        for row in rows:
            platform, key_hash, key_value, _priority, try_after, _, meta = row

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

            if self.is_key_busy(key_hash):
                continue

            await self.db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (now_dt.isoformat(), key_hash),
            )
            await self.db.commit()

            return platform, {
                "key_hash": key_hash,
                "key_value": key_value,
                "meta": meta,
            }

        # Pass 2: fallback to busy key if not in cooldown
        for row in rows:
            platform, key_hash, key_value, _priority, try_after, _, meta = row
            if key_hash in self._cooldowns and self._cooldowns[key_hash] > now:
                continue

            await self.db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (now_dt.isoformat(), key_hash),
            )
            await self.db.commit()

            return platform, {
                "key_hash": key_hash,
                "key_value": key_value,
                "meta": meta,
            }

        raise NoKeysAvailableError(
            f"All keys for candidate platforms {candidate_platforms} are in cooldown."
        )

    async def mark_key_exhausted(self, key_hash: str, duration_seconds: float) -> None:
        """Puts a key in cooldown for a specified duration."""
        now = datetime.datetime.now(datetime.UTC)
        try_after = now + datetime.timedelta(seconds=duration_seconds)
        self._cooldowns[key_hash] = try_after.timestamp()

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
