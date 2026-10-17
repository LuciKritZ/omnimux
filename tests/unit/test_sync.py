import json
import os
import tempfile
from collections.abc import AsyncGenerator, Generator

import aiosqlite
import pytest
import pytest_asyncio

from omnimux.services.db import init_db
from omnimux.services.sync import sync_keys_from_json


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


@pytest.fixture
def mock_keys_json() -> Generator[str, None, None]:
    data = {
        "keys": {
            "openrouter": [{"key_value": "test-key-1", "priority": 1, "meta": {"name": "test"}}]
        }
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(data, f)
        temp_name = f.name
    yield temp_name
    os.remove(temp_name)


@pytest.mark.asyncio
async def test_sync_keys_from_json(memory_db: aiosqlite.Connection, mock_keys_json: str) -> None:
    await sync_keys_from_json(memory_db, mock_keys_json)

    # Check if key was inserted
    query = "SELECT platform, key_value, priority, meta FROM api_keys"
    async with memory_db.execute(query) as cursor:
        rows = list(await cursor.fetchall())
        assert len(rows) == 1
        assert rows[0][0] == "openrouter"
        assert rows[0][1] == "test-key-1"
        assert rows[0][2] == 1
        assert "test" in rows[0][3]

    # Run again to ensure no duplicates (upsert behavior)
    await sync_keys_from_json(memory_db, mock_keys_json)
    async with memory_db.execute("SELECT COUNT(*) FROM api_keys") as cursor:
        count = await cursor.fetchone()
        assert count is not None
        assert count[0] == 1


@pytest.mark.asyncio
async def test_sync_keys_from_json_file_not_found(memory_db: aiosqlite.Connection) -> None:
    await sync_keys_from_json(memory_db, "non_existent_file_12345.json")


@pytest.mark.asyncio
async def test_sync_keys_from_json_invalid_json(memory_db: aiosqlite.Connection) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("{invalid_json: true")
        temp_name = f.name
    try:
        await sync_keys_from_json(memory_db, temp_name)
    finally:
        os.remove(temp_name)


@pytest.fixture
def mock_keys_json_missing_key_value() -> Generator[str, None, None]:
    data = {"keys": {"openrouter": [{"priority": 1, "meta": {"name": "test"}}]}}
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(data, f)
        temp_name = f.name
    yield temp_name
    os.remove(temp_name)


@pytest.mark.asyncio
async def test_sync_keys_from_json_missing_key_value(
    memory_db: aiosqlite.Connection, mock_keys_json_missing_key_value: str
) -> None:
    await sync_keys_from_json(memory_db, mock_keys_json_missing_key_value)
    async with memory_db.execute("SELECT COUNT(*) FROM api_keys") as cursor:
        count = await cursor.fetchone()
        assert count is not None
        assert count[0] == 0


@pytest.fixture
def mock_keys_json_invalid_types() -> Generator[str, None, None]:
    data = {"keys": {"not-a-list": "this is a string", "bad-key-obj": ["not a dict"]}}
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(data, f)
        temp_name = f.name
    yield temp_name
    os.remove(temp_name)


@pytest.mark.asyncio
async def test_sync_keys_from_json_invalid_types(
    memory_db: aiosqlite.Connection, mock_keys_json_invalid_types: str
) -> None:
    await sync_keys_from_json(memory_db, mock_keys_json_invalid_types)
    async with memory_db.execute("SELECT COUNT(*) FROM api_keys") as cursor:
        count = await cursor.fetchone()
        assert count is not None
        assert count[0] == 0


@pytest.mark.asyncio
async def test_sync_keys_preserves_existing_db_keys(
    memory_db: aiosqlite.Connection,
) -> None:
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority, meta)
        VALUES ('gemini', 'hash-existing-1', 'gemini-key-val', 1, '{}')
        """
    )
    await memory_db.commit()

    payload = {
        "keys": {"openrouter": [{"key_value": "new-openrouter-key", "priority": 1, "meta": {}}]}
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(payload, f)
        temp_name = f.name

    try:
        await sync_keys_from_json(memory_db, temp_name)
    finally:
        os.remove(temp_name)

    async with memory_db.execute("SELECT platform FROM api_keys ORDER BY platform") as cursor:
        rows = list(await cursor.fetchall())
        platforms = [r[0] for r in rows]
        assert "gemini" in platforms
        assert "openrouter" in platforms
