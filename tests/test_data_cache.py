"""Tests for dataset caching and integrity checking.

These use ``file://`` URLs so they exercise the real download path without
needing the network.
"""

import hashlib

import pytest

from evogfn.data.cache import (
    CACHE_ENV_VAR,
    ChecksumMismatchError,
    cache_dir,
    fetch,
    sha256_of,
)


@pytest.fixture
def source(tmp_path):
    """A local file to 'download', with its checksum."""
    path = tmp_path / "source.bin"
    payload = b"fitness,sequence\n1.0,VDGV\n"
    path.write_bytes(payload)
    return path.as_uri(), hashlib.sha256(payload).hexdigest()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the cache at a scratch directory."""
    target = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV_VAR, str(target))
    return target


class TestCacheLocation:
    def test_the_environment_variable_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "elsewhere"))
        assert cache_dir() == tmp_path / "elsewhere"

    def test_it_falls_back_to_xdg_cache_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cache_dir() == tmp_path / "evogfn"

    def test_the_cache_lives_outside_the_repository(self, monkeypatch):
        # A multi-megabyte dataset inside the checkout eventually gets committed
        # by someone running `git add -A`.
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert ".cache" in cache_dir().parts


@pytest.mark.usefixtures("cache")
class TestFetch:
    def test_a_matching_file_is_downloaded_and_kept(self, source, cache):
        url, digest = source
        path = fetch(url, sha256=digest, filename="data.bin")
        assert path.exists()
        assert path.parent == cache
        assert sha256_of(path) == digest

    def test_a_second_call_reuses_the_cached_copy(self, source):
        url, digest = source
        first = fetch(url, sha256=digest, filename="data.bin")
        stamp = first.stat().st_mtime_ns
        second = fetch(url, sha256=digest, filename="data.bin")
        assert second == first
        assert second.stat().st_mtime_ns == stamp, "file was re-downloaded"

    def test_a_wrong_checksum_is_an_error_not_a_warning(self, source):
        # Results computed against a silently changed file are not comparable to
        # results computed against the old one, and nothing in the run would say so.
        url, _ = source
        with pytest.raises(ChecksumMismatchError, match="remote file has changed"):
            fetch(url, sha256="0" * 64, filename="data.bin")

    def test_nothing_is_left_behind_when_the_checksum_fails(self, source, cache):
        # A truncated or wrong file left in the cache would be found by a later
        # run, which without re-verification would have trusted it.
        url, _ = source
        with pytest.raises(ChecksumMismatchError):
            fetch(url, sha256="0" * 64, filename="data.bin")
        assert list(cache.glob("*")) == []

    def test_a_corrupted_cached_file_is_detected(self, source):
        url, digest = source
        path = fetch(url, sha256=digest, filename="data.bin")
        path.write_bytes(b"truncated")
        with pytest.raises(ChecksumMismatchError, match="cached file"):
            fetch(url, sha256=digest, filename="data.bin")

    def test_force_replaces_a_corrupted_cached_file(self, source):
        url, digest = source
        path = fetch(url, sha256=digest, filename="data.bin")
        path.write_bytes(b"truncated")
        assert sha256_of(fetch(url, sha256=digest, filename="data.bin", force=True)) == digest
