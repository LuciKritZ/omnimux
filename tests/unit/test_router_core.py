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
