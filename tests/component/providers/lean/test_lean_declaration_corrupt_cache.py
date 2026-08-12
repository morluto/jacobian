"""Portable Lean declaration-cache corruption recovery."""

from pathlib import Path

import pytest

import jacobian.lean_frontend.declarations as declarations

_ENVIRONMENT_DIGEST = "sha256:" + "c" * 64


def _session(
    tmp_path: Path, cache_path: Path
) -> declarations._ReusableLeanQuerySession:
    return declarations._ReusableLeanQuerySession(
        command=[str(tmp_path / "lean")],
        cwd=tmp_path,
        process_environment={},
        source="import Init.Prelude",
        memory_mb="1024",
        isolated_home=True,
        environment_digest=_ENVIRONMENT_DIGEST,
        index_cache_path=cache_path,
        cache_results=True,
    )


def test_corrupt_regular_cache_is_removed_after_failed_restore(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_text(
        f"{_ENVIRONMENT_DIGEST}\nmalformed-row\n",
        encoding="utf-8",
    )

    session = _session(tmp_path, cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    assert not cache_path.exists()
    session.close()


def test_oversized_regular_cache_is_removed_before_next_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"x" * 65)
    monkeypatch.setattr(declarations, "_MAX_INDEX_BYTES", 64)

    session = _session(tmp_path, cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    assert not cache_path.exists()
    session.close()


def test_cache_symlink_is_rejected_without_deleting_its_target(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    target = tmp_path / "external.index"
    target.write_text("external cache", encoding="utf-8")
    cache_path.symlink_to(target)

    session = _session(tmp_path, cache_path)

    assert session._index_digest is None
    assert not session._index_path.exists()
    assert cache_path.is_symlink()
    assert target.read_text(encoding="utf-8") == "external cache"
    session.close()


def test_concurrent_atomic_replacement_is_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "state" / "core.index"
    cache_path.parent.mkdir()
    cache_path.write_text("stale cache", encoding="utf-8")

    def replace_then_fail(
        source: Path,
        destination: Path,
        *,
        max_bytes: int,
    ) -> None:
        del destination, max_bytes
        replacement = source.with_name("replacement.index")
        replacement.write_text("new cache", encoding="utf-8")
        replacement.replace(source)
        raise ValueError("simulated corrupt source")

    monkeypatch.setattr(declarations, "_copy_bounded_file", replace_then_fail)

    session = _session(tmp_path, cache_path)

    assert session._index_digest is None
    assert cache_path.read_text(encoding="utf-8") == "new cache"
    session.close()
