"""Fetching external datasets into a local cache, with integrity checking.

Every download is verified against a pinned SHA-256. A benchmark whose data can
change underneath it is not a benchmark: results computed against a silently
updated file are not comparable to results computed against the old one, and
nothing in the run would reveal the difference. A mismatch is therefore an error,
not a warning.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

#: Environment variable overriding where datasets are cached.
CACHE_ENV_VAR = "EVOFLOWNET_DATA_DIR"

#: Read size for hashing and copying. Large enough to be efficient, small enough
#: that a multi-gigabyte file never has to be held in memory.
_CHUNK_BYTES = 1 << 20


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match its expected checksum.

    This means the remote file changed, the download was corrupted, or the URL
    now serves something else entirely. All three invalidate any result computed
    from it, so none of them are recoverable by retrying silently.
    """


def cache_dir() -> Path:
    """Directory where datasets are stored.

    Respects ``EVOFLOWNET_DATA_DIR``; otherwise follows ``XDG_CACHE_HOME``, and
    falls back to ``~/.cache``. Deliberately outside the repository, so a large
    download is shared between checkouts and never lands in a commit.

    Returns:
        The cache directory. It may not exist yet.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "evogfn"


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, *, sha256: str, filename: str, force: bool = False) -> Path:
    """Return a local path to ``url``, downloading it if necessary.

    A cached file is re-verified against the checksum on every call rather than
    trusted because it exists. Verifying costs a hash of a file already on disk;
    not verifying risks running an experiment against a truncated or tampered
    copy.

    Args:
        url: Where to download from.
        sha256: Expected SHA-256 of the file, as lowercase hex.
        filename: Name to store it under in the cache.
        force: Re-download even if a valid cached copy exists.

    Returns:
        Path to the verified local file.

    Raises:
        ChecksumMismatchError: If the downloaded or cached bytes do not match
            ``sha256``.
    """
    destination = cache_dir() / filename
    if destination.exists() and not force:
        actual = sha256_of(destination)
        if actual == sha256:
            return destination
        raise ChecksumMismatchError(
            f"cached file {destination} has checksum {actual}, expected {sha256}; "
            f"delete it to re-download, or pass force=True"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file in the same directory, then move into place.
    # An interrupted download must not leave a truncated file that a later run
    # would find and, without the checksum, would have trusted.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as scratch:
        partial = Path(scratch.name)
        try:
            # Only http(s) is ever passed here; the URLs are constants in this
            # package, not user input.
            with urllib.request.urlopen(url) as response:  # noqa: S310
                shutil.copyfileobj(response, scratch, _CHUNK_BYTES)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"downloaded {url} with checksum {actual}, expected {sha256}; "
            f"the remote file has changed and results would not be comparable"
        )

    partial.replace(destination)
    return destination
