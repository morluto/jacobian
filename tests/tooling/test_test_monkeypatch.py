"""Focused tests for the test monkeypatch-evidence guard."""

from __future__ import annotations

from pathlib import Path

from tools.check_test_monkeypatch import check


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _violations(root: Path) -> list[str]:
    return [violation.path for violation in check(root).violations]


def test_patch_without_any_assertion_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_bad.py",
        "import pytest\n"
        "def test_patches_but_proves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "    do_something()\n",
    )

    assert _violations(tmp_path) == ["tests/example/test_bad.py"]


def test_plain_assert_counts_as_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_ok.py",
        "import pytest\n"
        "def test_patches_and_asserts(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: 1)\n"
        "    assert compute() == 1\n",
    )

    assert _violations(tmp_path) == []


def test_typed_error_counts_as_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_raises.py",
        "import pytest\n"
        "def test_patch_injects_failure(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "    with pytest.raises(ValueError):\n"
        "        compute()\n",
    )

    assert _violations(tmp_path) == []


def test_assert_helper_counts_as_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_helper.py",
        "import pytest\n"
        "def _assert_rejected() -> None:\n"
        "    ...\n"
        "def test_patch_uses_helper(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "    _assert_rejected()\n",
    )

    assert _violations(tmp_path) == []


def test_sentinel_replacement_counts_as_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_sentinel.py",
        "import pytest\n"
        "def test_patch_forbids_a_call(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    def fail(*_a: object, **_k: object) -> None:\n"
        "        raise AssertionError('must not run')\n"
        "    monkeypatch.setattr('x.y', fail)\n"
        "    compute()\n",
    )

    assert _violations(tmp_path) == []


def test_waiver_marker_suppresses_the_violation(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_waived.py",
        "import pytest\n"
        "def test_patch_with_rationale(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    # monkeypatch-evidence: the replacement is the observable behavior\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "    compute()\n",
    )

    assert _violations(tmp_path) == []


def test_non_test_functions_and_other_methods_are_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_helper_only.py",
        "import pytest\n"
        "def _helper(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "def test_uses_helper(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    _helper(monkeypatch)\n",
    )

    assert _violations(tmp_path) == []


def test_environment_patches_are_covered(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/example/test_env.py",
        "import pytest\n"
        "def test_sets_env_without_evidence(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setenv('X', '1')\n"
        "    read_env()\n",
    )

    assert _violations(tmp_path) == ["tests/example/test_env.py"]


def test_report_renders_ok_and_failure(tmp_path: Path) -> None:
    _write(
        tmp_path, "tests/example/test_ok.py", "def test_a() -> None:\n    assert True\n"
    )
    report = check(tmp_path)
    assert report.ok and report.failed is False
    assert "test-hygiene: OK" in report.render()

    _write(
        tmp_path,
        "tests/example/test_bad.py",
        "import pytest\n"
        "def test_bad(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setattr('x.y', lambda *a, **k: None)\n"
        "    compute()\n",
    )
    failing = check(tmp_path)
    assert failing.failed
    rendered = failing.render()
    assert "monkeypatch-without-evidence" in rendered
    assert "tests/example/test_bad.py" in rendered
