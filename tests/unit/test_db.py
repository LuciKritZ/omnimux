import hashlib
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from omnimux.services.db import (
    auto_import_from_json_if_empty,
    create_client_key,
    export_configuration,
    get_all_platforms_and_models,
    init_db,
    list_client_keys,
    revoke_client_key,
    save_platform_and_models,
    validate_client_key,
)


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provides a fresh in-memory SQLite connection."""
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


@pytest.mark.asyncio
async def test_init_db_creates_all_tables(memory_db: aiosqlite.Connection) -> None:
    for tbl in ["api_keys", "client_keys", "platforms", "models"]:
        async with memory_db.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl}';"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == tbl


@pytest.mark.asyncio
async def test_client_keys_crud(memory_db: aiosqlite.Connection) -> None:
    # 1. Create
    raw_key1, info1 = await create_client_key(memory_db, "claude-code")
    assert raw_key1.startswith("om_")
    assert info1["name"] == "claude-code"

    raw_key2, _info2 = await create_client_key(memory_db, "web-app")
    assert raw_key2.startswith("om_")

    # 2. Validate
    assert await validate_client_key(memory_db, raw_key1) is True
    assert await validate_client_key(memory_db, raw_key2) is True
    assert await validate_client_key(memory_db, "om_invalid_token") is False

    # 3. List
    keys = await list_client_keys(memory_db)
    assert len(keys) == 2
    assert keys[0]["status"] == "active"

    # 4. Revoke by ID
    revoked_id = await revoke_client_key(memory_db, str(keys[0]["id"]))
    assert revoked_id is True
    assert await validate_client_key(memory_db, raw_key2) is False

    # Revoke already revoked
    assert await revoke_client_key(memory_db, str(keys[0]["id"])) is False

    # Revoke by prefix
    prefix = info1["prefix"].replace("...", "")
    revoked_pref = await revoke_client_key(memory_db, prefix)
    assert revoked_pref is True
    assert await validate_client_key(memory_db, raw_key1) is False


@pytest.mark.asyncio
async def test_save_and_get_platforms_and_models(memory_db: aiosqlite.Connection) -> None:
    models_to_save = [
        {
            "id": "qwen2.5-coder",
            "alias": "coder",
            "max_input_tokens": 128000,
            "supports_vision": False,
        },
        {"id": "claude-3-5-sonnet", "supports_vision": True},
    ]

    await save_platform_and_models(
        memory_db, "openrouter", "https://openrouter.ai/api", "anthropic", models_to_save
    )

    platforms_dict = await get_all_platforms_and_models(memory_db)
    assert "openrouter/qwen2.5-coder" in platforms_dict
    m1 = platforms_dict["openrouter/qwen2.5-coder"]
    assert m1["apiBase"] == "https://openrouter.ai/api"
    assert m1["protocol"] == "anthropic"
    assert m1["max_input_tokens"] == 128000
    assert m1["supports_vision"] is False
    assert m1["alias"] == "coder"

    m2 = platforms_dict["openrouter/claude-3-5-sonnet"]
    assert m2["supports_vision"] is True
    assert m2["max_input_tokens"] == 200000


@pytest.mark.asyncio
async def test_export_configuration(memory_db: aiosqlite.Connection, tmp_path: Path) -> None:
    # Insert API key
    await memory_db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority, meta)
        VALUES ('openrouter', 'hash1', 'val1', 2, '{"api_base": "https://openrouter.ai"}')
    """
    )
    # Save model
    await save_platform_and_models(
        memory_db, "openrouter", "https://openrouter.ai", "openai", [{"id": "model-1"}]
    )

    out_file = tmp_path / "subdir" / "exported.json"
    exported = await export_configuration(memory_db, str(out_file))

    assert out_file.exists()
    assert "openrouter" in exported["keys"]
    assert exported["keys"]["openrouter"][0]["key_value"] == "val1"
    assert "openrouter/model-1" in exported["models"]


@pytest.mark.asyncio
async def test_auto_import_from_json_if_empty(
    memory_db: aiosqlite.Connection, tmp_path: Path
) -> None:
    sample_json = {
        "keys": {
            "tokenharbor": [
                {
                    "key_value": "tok_123",
                    "meta": {
                        "api_base": "https://api.tokenharbor.com",
                        "protocol": "anthropic",
                    },
                }
            ],
            "anotherplat": [
                {
                    "key_value": "tok_456",
                    "meta": {
                        "api_base": "https://api.another.com",
                        "protocol": "openai",
                    },
                }
            ],
        },
        "models": {
            "tokenharbor/sonnet": {
                "max_input_tokens": 100000,
                "supports_vision": True,
                "apiBase": "https://api.tokenharbor.com",
            },
            "anotherplat/model1": {
                "max_input_tokens": 80000,
                "supports_vision": False,
            },
            "invalid_no_slash": "not_dict",
            "no_slash_dict": {"max_input_tokens": 1000},
        },
    }
    json_path = tmp_path / "models.json"
    json_path.write_text(json.dumps(sample_json), encoding="utf-8")

    # 1. Non-existent file does nothing
    await auto_import_from_json_if_empty(memory_db, str(tmp_path / "missing.json"))

    # 2. Valid file imports models
    await auto_import_from_json_if_empty(memory_db, str(json_path))
    platforms = await get_all_platforms_and_models(memory_db)
    assert "tokenharbor/sonnet" in platforms
    assert platforms["tokenharbor/sonnet"]["apiBase"] == "https://api.tokenharbor.com"
    assert platforms["tokenharbor/sonnet"]["protocol"] == "anthropic"

    # 3. Running again when DB is not empty skips import
    await auto_import_from_json_if_empty(memory_db, str(json_path))


@pytest.mark.asyncio
async def test_keyless_platform_hash_uniqueness(memory_db: aiosqlite.Connection) -> None:
    def get_hash(platform: str, key: str) -> str:
        return hashlib.sha256(f"{platform}:{key}".encode()).hexdigest()

    ollama_hash = get_hash("ollama", "")
    vllm_hash = get_hash("vllm", "")

    assert ollama_hash != vllm_hash

    await memory_db.execute(
        "INSERT INTO api_keys (platform, key_hash, key_value, priority) VALUES (?, ?, ?, 1)",
        ("ollama", ollama_hash, ""),
    )
    await memory_db.execute(
        "INSERT INTO api_keys (platform, key_hash, key_value, priority) VALUES (?, ?, ?, 1)",
        ("vllm", vllm_hash, ""),
    )
    await memory_db.commit()

    async with memory_db.execute("SELECT COUNT(*) FROM api_keys") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 2


@pytest.mark.asyncio
async def test_platforms_without_models_visible(memory_db: aiosqlite.Connection) -> None:
    await save_platform_and_models(
        memory_db,
        platform="openrouter",
        api_base="https://openrouter.ai/api",
        protocol="openai",
        models_list=[],
    )

    platforms_data = await get_all_platforms_and_models(memory_db)
    assert "openrouter" in platforms_data
    assert platforms_data["openrouter"]["apiBase"] == "https://openrouter.ai/api"
    async with memory_db.execute("SELECT name, api_base FROM platforms") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "openrouter"


@pytest.mark.asyncio
async def test_revoke_client_key_escapes_wildcards(memory_db: aiosqlite.Connection) -> None:
    _, info1 = await create_client_key(memory_db, "key1")
    _, _info2 = await create_client_key(memory_db, "key2")

    id1 = info1["prefix"][:8]
    await revoke_client_key(memory_db, id1)

    async with memory_db.execute(
        "SELECT name, revoked_at FROM client_keys ORDER BY name"
    ) as cursor:
        rows = list(await cursor.fetchall())
        assert rows[0][1] is not None
        assert rows[1][1] is None
