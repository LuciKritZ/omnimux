import time
import typing
from pathlib import Path

import pytest

from omnimux.exceptions import CacheError
from omnimux.services.cache import ImageCache


def test_cache_creation_success(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=cache_dir)
    assert cache.cache_dir.exists()


def test_cache_creation_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def mock_mkdir(*args: typing.Any, **kwargs: typing.Any) -> None:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    with pytest.raises(CacheError, match="Failed to create cache directory"):
        ImageCache(cache_dir=tmp_path / "cache")


def test_hash_image(tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)
    hash1 = cache.hash_image("some_base64_data")
    hash2 = cache.hash_image("some_base64_data")
    hash3 = cache.hash_image("different_data")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_store_and_get_path(tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)

    assert cache.get_path("missing_hash") is None

    file_path = cache.store("my_hash", b"image bytes")
    assert file_path.exists()
    assert file_path.read_bytes() == b"image bytes"

    retrieved = cache.get_path("my_hash")
    assert retrieved == file_path

    file_path2 = cache.store("my_hash", b"new bytes that should not overwrite")
    assert file_path2 == file_path
    assert file_path.read_bytes() == b"image bytes"


def test_store_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)

    def mock_write_bytes(*args: typing.Any, **kwargs: typing.Any) -> None:
        raise OSError("Disk full")

    monkeypatch.setattr(Path, "write_bytes", mock_write_bytes)

    with pytest.raises(CacheError, match="Failed to write image to cache"):
        cache.store("hash", b"bytes")


def test_clear_cache(tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)
    cache.store("file1", b"1")
    cache.store("file2", b"2")

    assert len(list(tmp_path.iterdir())) == 2

    cache.clear()

    assert len(list(tmp_path.iterdir())) == 0


def test_clear_cache_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)
    cache.store("file1", b"1")

    def mock_iterdir(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise OSError("Read error")

    monkeypatch.setattr(Path, "iterdir", mock_iterdir)

    with pytest.raises(CacheError, match="Failed to clear cache"):
        cache.clear()


def test_offload_logic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Reduce max size to 10 bytes for testing
    import omnimux.services.cache as cache_module

    monkeypatch.setattr(cache_module, "MAX_CACHE_SIZE_BYTES", 10)

    cache = ImageCache(cache_dir=tmp_path)

    # Store 4 bytes -> total 4 bytes
    cache.store("file1", b"1234")
    time.sleep(0.01)  # ensure mtime is different

    # Store 4 bytes -> total 8 bytes
    cache.store("file2", b"5678")
    time.sleep(0.01)

    # Store 4 bytes -> total 12 bytes (> 10 max). Should offload until <= 8 (10 * 0.8)
    # Target size: 8 bytes.
    # Current size before file3 write logic triggers is 8.
    # Wait, store() runs offloading *before* writing.
    # So before writing file3, size is 8. Offloading is not triggered.
    # Then file3 is written, total size = 12.
    cache.store("file3", b"9012")

    assert len(list(tmp_path.iterdir())) == 3

    # Store file4 -> offloading triggered before writing file4.
    # Size is 12 > 10. Offloads file1 (oldest, size 4). Size becomes 8 <= 8. Stop.
    # Writes file4 (size 4). Total size 12 again.
    cache.store("file4", b"abcd")

    files = list(tmp_path.iterdir())
    assert len(files) == 3
    names = [f.name for f in files]
    assert "file1" not in names
    assert "file2" in names
    assert "file3" in names
    assert "file4" in names


def test_offload_read_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = ImageCache(cache_dir=tmp_path)

    def mock_iterdir(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise OSError("Cannot read")

    monkeypatch.setattr(Path, "iterdir", mock_iterdir)

    with pytest.raises(CacheError, match="Failed to read cache directory"):
        cache.offload_if_needed()


def test_offload_unlink_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import omnimux.services.cache as cache_module

    monkeypatch.setattr(cache_module, "MAX_CACHE_SIZE_BYTES", 10)

    cache = ImageCache(cache_dir=tmp_path)
    cache.store("file1", b"123456")
    time.sleep(0.01)
    cache.store("file2", b"789012")  # triggers offloading of file1

    # Store one more, but mock unlink to fail
    def mock_unlink(*args: typing.Any, **kwargs: typing.Any) -> None:
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "unlink", mock_unlink)

    # Offloading triggers again. Will try to unlink file2, fail, ignore it,
    # and finish without crashing.
    cache.store("file3", b"345678")

    # All files still exist because unlink failed
    assert len(list(tmp_path.iterdir())) == 3
