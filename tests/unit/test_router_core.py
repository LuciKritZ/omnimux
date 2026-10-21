import asyncio
import datetime
from collections.abc import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio

from omnimux.services.db import init_db
from omnimux.services.router_core import NoKeysAvailableError, RouterCore


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        # Insert some test data
        await db.executemany(
            """
            INSERT INTO api_keys (platform, key_hash, key_value, priority, last_used_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("openrouter", "hash1", "val1", 1, "2023-01-01T00:00:00+00:00"),
                ("openrouter", "hash2", "val2", 2, "2023-01-01T00:00:00+00:00"),
                ("openrouter", "hash3", "val3", 2, "2023-01-01T00:00:01+00:00"),
                ("gemini", "hash4", "val4", 1, "2023-01-01T00:00:00+00:00"),
            ],
        )
        await db.commit()
        yield db


@pytest.mark.asyncio
async def test_get_next_key_priority(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # Priority 2 keys should come first. hash2 was used earlier than hash3.
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash2"

    # Mark hash2 as used, meaning hash3 is now the oldest priority 2 key
    await router.mark_key_used("hash2")
    key2 = await router.get_next_key("openrouter")
    assert key2["key_hash"] == "hash3"


@pytest.mark.asyncio
async def test_mark_key_exhausted(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # Mark hash2 and hash3 as exhausted
    await router.mark_key_exhausted("hash2", 3600.0)
    await router.mark_key_exhausted("hash3", 3600.0)

    # It should fallback to hash1 (priority 1) because Priority 2s are in cooldown
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash1"


@pytest.mark.asyncio
async def test_no_keys_available(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    with pytest.raises(NoKeysAvailableError):
        await router.get_next_key("unknown")


@pytest.mark.asyncio
async def test_get_best_platform_and_key(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # 1. No candidates provided
    with pytest.raises(NoKeysAvailableError):
        await router.get_best_platform_and_key([])

    # 2. No keys configured for the candidate platforms
    with pytest.raises(NoKeysAvailableError):
        await router.get_best_platform_and_key(["non-existent-platform"])

    # 3. Successful retrieval
    platform, key = await router.get_best_platform_and_key(["openrouter", "gemini"])
    assert platform == "openrouter"
    assert key["key_hash"] == "hash2"  # priority 2

    # 4. Memory cooldown check
    router._cooldowns["hash2"] = datetime.datetime.now(datetime.UTC).timestamp() + 3600
    platform, key = await router.get_best_platform_and_key(["openrouter"])
    assert key["key_hash"] == "hash3"  # priority 1

    # 5. DB cooldown check
    now = datetime.datetime.now(datetime.UTC)
    future = (now + datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute(
        "UPDATE api_keys SET try_after = ? WHERE key_hash IN ('hash3', 'hash1')", (future,)
    )
    await memory_db.commit()

    with pytest.raises(NoKeysAvailableError):
        await router.get_best_platform_and_key(["openrouter"])

    # 6. Cooldown expired (testing db cooldown clearing)
    past = (now - datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute("UPDATE api_keys SET try_after = ? WHERE key_hash = 'hash3'", (past,))
    await memory_db.commit()
    router._cooldowns["hash3"] = now.timestamp() - 3600

    platform, key = await router.get_best_platform_and_key(["openrouter"])
    assert platform == "openrouter"
    assert key["key_hash"] == "hash3"

    await router.mark_key_exhausted("hash4", 3600.0)
    with pytest.raises(NoKeysAvailableError, match="All keys for gemini are currently exhausted"):
        await router.get_next_key("gemini")


@pytest.mark.asyncio
async def test_expired_cooldown(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # Put it in cooldown but make it negative (expired)
    await router.mark_key_exhausted("hash2", -10.0)

    # It should immediately clear the cooldown and be available
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash2"


@pytest.mark.asyncio
async def test_concurrency_locks(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # Acquire the lock for hash2 via context manager
    # We will simulate a long running task
    async def long_task() -> None:
        async with router.use_key("hash2"):
            await asyncio.sleep(0.1)

    # Start task in background
    task = asyncio.create_task(long_task())

    # Wait a tiny bit to ensure lock is acquired
    await asyncio.sleep(0.01)

    # Requesting a key should skip hash2 because it's locked, and yield hash3 instead!
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash3"

    # Wait for task to finish
    await task

    # Now hash2 is free again, but its last_used_at was updated by the context manager!
    # So hash3 is actually older than hash2 now.
    key2 = await router.get_next_key("openrouter")
    assert key2["key_hash"] == "hash3"


@pytest.mark.asyncio
async def test_db_cooldown_sync(memory_db: aiosqlite.Connection) -> None:
    # Set a future try_after directly in DB to simulate another instance exhausting it
    now = datetime.datetime.now(datetime.UTC)
    future = (now + datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute("UPDATE api_keys SET try_after = ? WHERE key_hash = 'hash2'", (future,))
    await memory_db.commit()

    router = RouterCore(memory_db)

    # Priority 2 key 'hash2' is in DB cooldown, so it should be skipped
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash3"

    # Check that it synced to memory
    assert "hash2" in router._cooldowns


@pytest.mark.asyncio
async def test_second_pass_cooldown(memory_db: aiosqlite.Connection) -> None:
    # Make all keys except hash2 have a future try_after in DB
    now = datetime.datetime.now(datetime.UTC)
    future = (now + datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute(
        "UPDATE api_keys SET try_after = ? WHERE key_hash != 'hash2'", (future,)
    )
    await memory_db.commit()

    router = RouterCore(memory_db)

    # Lock hash2
    async def lock_hash2() -> None:
        async with router.use_key("hash2"):
            await asyncio.sleep(0.1)

    task = asyncio.create_task(lock_hash2())
    await asyncio.sleep(0.01)

    # Now all unlocked keys (hash1, hash3) are in DB cooldown
    # (not in memory yet for the second pass)
    # The first pass skips hash2 (locked).
    # The second pass will check hash1, hash3 and see they are in DB cooldown, skipping them.
    # It will find hash2 (locked but not in cooldown) and return it!
    key = await router.get_next_key("openrouter")
    assert key["key_hash"] == "hash2"

    await task


@pytest.mark.asyncio
async def test_standardized_iso_timestamps(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority)
        VALUES ('openrouter', 'h1', 'k1', 10)
        """
    )
    await memory_db.commit()

    platform, key = await router.get_best_platform_and_key(["openrouter"])
    assert platform == "openrouter"
    assert key["key_hash"] == "h1"
    async with memory_db.execute("SELECT last_used_at FROM api_keys WHERE key_hash = 'h1'") as cur:
        row = await cur.fetchone()
        assert row is not None
        ts = row[0]
        assert "T" in ts, f"Timestamp {ts} must be ISO format with T for proper SQLite ordering"


@pytest.mark.asyncio
async def test_candidate_platform_preference_order(memory_db: aiosqlite.Connection) -> None:
    # Insert a priority 2 key for gemini so both openrouter and gemini have top priority 2
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority)
        VALUES ('gemini', 'hash5', 'val5', 2)
        """
    )
    await memory_db.commit()

    router = RouterCore(memory_db)
    # When gemini is first in candidate_platforms, gemini should be chosen.
    platform, key = await router.get_best_platform_and_key(["gemini", "openrouter"])
    assert platform == "gemini"
    assert key["key_hash"] == "hash5"

    # When openrouter is first, openrouter should be chosen.
    platform2, key2 = await router.get_best_platform_and_key(["openrouter", "gemini"])
    assert platform2 == "openrouter"
    assert key2["key_hash"] == "hash2"


@pytest.mark.asyncio
async def test_router_concurrent_semaphores(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    concurrent_count = 0
    max_concurrent = 0

    async def worker() -> None:
        nonlocal concurrent_count, max_concurrent
        async with router.use_key("hash2"):
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert max_concurrent == 5
    assert router.is_key_busy("hash2") is False


@pytest.mark.asyncio
async def test_get_best_platform_and_key_busy_fallback(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Mark all openrouter keys as busy
    router._in_flight["hash1"] = 1
    router._in_flight["hash2"] = 1
    router._in_flight["hash3"] = 1

    # Pass 1 skips all busy keys, Pass 2 returns the highest priority busy key
    platform, key = await router.get_best_platform_and_key(["openrouter"])
    assert platform == "openrouter"
    assert key["key_hash"] == "hash2"

    # Now put all keys in future cooldown via DB try_after
    now = datetime.datetime.now(datetime.UTC)
    future = (now + datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute("UPDATE api_keys SET try_after = ?", (future,))
    await memory_db.commit()

    with pytest.raises(NoKeysAvailableError, match="All keys for candidate platforms"):
        await router.get_best_platform_and_key(["openrouter"])


@pytest.mark.asyncio
async def test_get_next_key_db_cooldown_pass2(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Make hash2 busy so pass 1 skips it
    router._in_flight["hash1"] = 1
    router._in_flight["hash2"] = 1
    router._in_flight["hash3"] = 1

    now = datetime.datetime.now(datetime.UTC)
    future = (now + datetime.timedelta(seconds=3600)).isoformat()
    await memory_db.execute(
        "UPDATE api_keys SET try_after = ? WHERE platform = 'openrouter'", (future,)
    )
    await memory_db.commit()

    with pytest.raises(
        NoKeysAvailableError, match="All keys for openrouter are currently exhausted"
    ):
        await router.get_next_key("openrouter")


@pytest.mark.asyncio
async def test_router_key_concurrency_limit_meta(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Default limit
    assert router.get_key_concurrency_limit("hash1") == 10

    # Parse limit from dict meta
    assert router.get_key_concurrency_limit("hash2", {"max_concurrency": 5}) == 5
    assert router.get_key_concurrency_limit("hash3", {"concurrency_limit": "3"}) == 3

    # Parse limit from JSON string meta
    assert router.get_key_concurrency_limit("hash4", '{"max_concurrency": 2}') == 2
    # Invalid JSON string falls back to default
    assert router.get_key_concurrency_limit("hash5", "invalid-json") == 10
    # Invalid int string falls back to default
    assert router.get_key_concurrency_limit("hash6", {"max_concurrency": "not-an-int"}) == 10

    # is_key_at_capacity
    router._in_flight["hash4"] = 1
    assert router.is_key_at_capacity("hash4") is False
    router._in_flight["hash4"] = 2
    assert router.is_key_at_capacity("hash4") is True


@pytest.mark.asyncio
async def test_router_round_robin_equal_keys(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Insert two keys with identical priority and last_used_at for test platform
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority, last_used_at)
        VALUES
            ('testplat', 'thash1', 'tval1', 1, '2023-01-01T00:00:00+00:00'),
            ('testplat', 'thash2', 'tval2', 1, '2023-01-01T00:00:00+00:00')
        """
    )
    await memory_db.commit()

    # get_next_key should alternate between thash1 and thash2 via round-robin
    k1 = await router.get_next_key("testplat")
    k2 = await router.get_next_key("testplat")
    assert {k1["key_hash"], k2["key_hash"]} == {"thash1", "thash2"}
    assert k1["key_hash"] != k2["key_hash"]

    # get_best_platform_and_key should alternate between thash1 and thash2
    # Reset last_used_at to identical timestamp
    await memory_db.execute(
        "UPDATE api_keys SET last_used_at = '2023-01-01T00:00:00+00:00' WHERE platform = 'testplat'"
    )
    await memory_db.commit()

    p1, b1 = await router.get_best_platform_and_key(["testplat"])
    p2, b2 = await router.get_best_platform_and_key(["testplat"])
    assert p1 == "testplat" and p2 == "testplat"
    # Even if last_used_at got updated by b1, if they have identical rank they alternate
    assert b1["key_hash"] != b2["key_hash"]


@pytest.mark.asyncio
async def test_router_least_loaded_selection(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Mark hash2 with in_flight = 3 and hash3 with in_flight = 1
    router._in_flight["hash1"] = 10
    router._in_flight["hash2"] = 3
    router._in_flight["hash3"] = 1

    # Pass 2 should select hash3 as least-loaded among priority 2 keys
    k = await router.get_next_key("openrouter")
    assert k["key_hash"] == "hash3"

    p, best = await router.get_best_platform_and_key(["openrouter"])
    assert p == "openrouter"
    assert best["key_hash"] == "hash3"

    # If hash3 reaches capacity (e.g. limit 1) and hash2 has capacity
    router._key_concurrency_limits["hash3"] = 1
    router._key_concurrency_limits["hash2"] = 5
    # Now hash3 is at capacity, so hash2 is selected as under capacity
    k2 = await router.get_next_key("openrouter")
    assert k2["key_hash"] == "hash2"


@pytest.mark.asyncio
async def test_session_affinity_lifecycle(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # 1. Unknown session returns None
    assert router.get_session_affinity("unknown") is None

    # 2. Set affinity and get it
    router.set_session_affinity("sess1", "openrouter", "hash2", ttl_seconds=100.0)
    aff = router.get_session_affinity("sess1")
    assert aff is not None
    assert aff[0] == "openrouter"
    assert aff[1] == "hash2"

    # 3. Expired affinity returns None and cleans up
    router.set_session_affinity("sess2", "openrouter", "hash2", ttl_seconds=-10.0)
    assert router.get_session_affinity("sess2") is None
    assert "sess2" not in router._session_affinity

    # 4. Explicit eviction
    router.evict_session_affinity("sess1")
    assert router.get_session_affinity("sess1") is None

    # 5. Evict affinities for key
    router.set_session_affinity("sessA", "openrouter", "hash2")
    router.set_session_affinity("sessB", "openrouter", "hash3")
    router.evict_affinities_for_key("hash2")
    assert router.get_session_affinity("sessA") is None
    assert router.get_session_affinity("sessB") is not None


@pytest.mark.asyncio
async def test_router_best_platform_session_affinity(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    # First turn: selects best key (hash2) and binds sess1 to it
    p1, k1 = await router.get_best_platform_and_key(["openrouter"], session_id="sess1")
    assert p1 == "openrouter"
    assert k1["key_hash"] == "hash2"

    # Second turn without session_id: round-robin might switch or pick another key
    # But with session_id="sess1": MUST return hash2 again!
    p2, k2 = await router.get_best_platform_and_key(["openrouter"], session_id="sess1")
    assert p2 == "openrouter"
    assert k2["key_hash"] == "hash2"

    # Turn with session_id="sess2": binds to a new key
    p3, _ = await router.get_best_platform_and_key(["openrouter"], session_id="sess2")
    assert p3 == "openrouter"

    # If bound key is marked exhausted, session affinity is cleared immediately
    await router.mark_key_exhausted("hash2", duration_seconds=3600.0)
    assert router.get_session_affinity("sess1") is None

    # Subsequent call with sess1 seamlessly routes to the next healthy key (hash3)
    p4, k4 = await router.get_best_platform_and_key(["openrouter"], session_id="sess1")
    assert p4 == "openrouter"
    assert k4["key_hash"] == "hash3"


@pytest.mark.asyncio
async def test_router_get_next_key_session_affinity(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)

    k1 = await router.get_next_key("openrouter", session_id="s1")
    assert k1["key_hash"] == "hash2"

    # Successive call returns same key
    k2 = await router.get_next_key("openrouter", session_id="s1")
    assert k2["key_hash"] == "hash2"

    # If key is exhausted, session affinity evicted and fallback picked
    await router.mark_key_exhausted("hash2", 3600.0)
    k3 = await router.get_next_key("openrouter", session_id="s1")
    assert k3["key_hash"] == "hash3"


@pytest.mark.asyncio
async def test_router_session_affinity_pass2_busy_keys(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Mark all keys busy so Pass 1 is skipped
    router._in_flight["hash1"] = 1
    router._in_flight["hash2"] = 1
    router._in_flight["hash3"] = 1
    router._in_flight["hash4"] = 1

    # 1. get_next_key Pass 2 with session_id sets affinity
    k = await router.get_next_key("openrouter", session_id="busy_sess_1")
    assert k["key_hash"] in ("hash2", "hash3")
    aff1 = router.get_session_affinity("busy_sess_1")
    assert aff1 is not None
    assert aff1[1] == k["key_hash"]

    # 2. get_best_platform_and_key Pass 2 with session_id sets affinity
    p, best = await router.get_best_platform_and_key(["openrouter"], session_id="busy_sess_2")
    assert p == "openrouter"
    aff2 = router.get_session_affinity("busy_sess_2")
    assert aff2 is not None
    assert aff2[1] == best["key_hash"]


@pytest.mark.asyncio
async def test_router_session_affinity_stale_key_eviction(memory_db: aiosqlite.Connection) -> None:
    router = RouterCore(memory_db)
    # Set affinity to key not in valid rows
    router.set_session_affinity("stale1", "openrouter", "deleted_key_hash")
    k = await router.get_next_key("openrouter", session_id="stale1")
    assert k["key_hash"] != "deleted_key_hash"
    aff1 = router.get_session_affinity("stale1")
    assert aff1 is not None
    assert aff1[1] == k["key_hash"]

    router.set_session_affinity("stale2", "gemini", "deleted_gemini_hash")
    p, _ = await router.get_best_platform_and_key(["openrouter"], session_id="stale2")
    assert p == "openrouter"
    aff2 = router.get_session_affinity("stale2")
    assert aff2 is not None
    assert aff2[0] == "openrouter"
