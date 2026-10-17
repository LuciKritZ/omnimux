import hashlib
from pathlib import Path

from omnimux.exceptions import CacheError

# 500 MB in bytes
MAX_CACHE_SIZE_BYTES = 500 * 1024 * 1024


class ImageCache:
    """Handles local disk caching for multimodal images with LRU eviction."""

    def __init__(self, cache_dir: Path | str = "~/.omnimux/cache") -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CacheError(f"Failed to create cache directory: {e}") from e

    def hash_image(self, base64_data: str) -> str:
        """Returns a deterministic SHA-256 hash of the base64 string."""
        return hashlib.sha256(base64_data.encode("utf-8")).hexdigest()

    def store(self, image_hash: str, raw_bytes: bytes) -> Path:
        """Stores the image to disk. If cache exceeds limit, offloads oldest files."""
        existing = self.get_path(image_hash)
        if existing is not None:
            return existing
        self.offload_if_needed()
        file_path = self.cache_dir / image_hash
        try:
            file_path.write_bytes(raw_bytes)
        except OSError as e:
            raise CacheError(f"Failed to write image to cache: {e}") from e
        return file_path

    def get_path(self, image_hash: str) -> Path | None:
        """Returns the file path if it exists in the cache, updating its modified time for LRU."""
        file_path = self.cache_dir / image_hash
        if file_path.exists():
            # Update modification time (touch) for LRU mechanism
            file_path.touch(exist_ok=True)
            return file_path
        return None

    def offload_if_needed(self) -> None:
        """Enforces the 500MB max capacity by deleting the oldest files."""
        try:
            files = [f for f in self.cache_dir.iterdir() if f.is_file()]
        except OSError as e:
            raise CacheError(f"Failed to read cache directory: {e}") from e

        total_size = sum(f.stat().st_size for f in files)

        if total_size > MAX_CACHE_SIZE_BYTES:
            # Sort files by modification time (oldest first)
            files.sort(key=lambda x: x.stat().st_mtime)

            # Delete files until we're under 80% of max capacity
            target_size = MAX_CACHE_SIZE_BYTES * 0.8
            current_size = total_size

            for f in files:
                if current_size <= target_size:
                    break
                size = f.stat().st_size
                try:
                    f.unlink()
                    current_size -= size
                except OSError:
                    # Ignore deletion errors during offloading, keep going
                    pass

    def clear(self) -> None:
        """Deletes all files in the cache."""
        try:
            for f in self.cache_dir.iterdir():
                if f.is_file():
                    f.unlink()
        except OSError as e:
            raise CacheError(f"Failed to clear cache: {e}") from e
