from __future__ import annotations

from pathlib import Path

from tools.check_doc_commands import validate_documents


def _write_fixture(
    root: Path,
    document: str,
    makefile: str = "check:\n\nunit:\n",
) -> tuple[Path, Path]:
    (root / "tests/unit").mkdir(parents=True)
    test_path = root / "tests/unit/test_example.py"
    test_path.write_text(
        'def test_example():\n    raise AssertionError("this fixture must not execute")\n',
        encoding="utf-8",
    )
    (root / "Makefile").write_text(makefile, encoding="utf-8")
    doc_path = root / "guide.md"
    doc_path.write_text(document, encoding="utf-8")
    return root, doc_path


def test_validates_targets_and_test_paths_without_running_them(tmp_path: Path) -> None:
    root, document = _write_fixture(
        tmp_path,
        """```sh
make unit TESTS=tests/unit/test_example.py
```
""",
    )

    assert validate_documents(root, (document,)) == []


def test_reports_unknown_target_and_missing_test_path(tmp_path: Path) -> None:
    root, document = _write_fixture(
        tmp_path,
        "`make missing TESTS=tests/unit/test_missing.py`\n",
    )

    failures = validate_documents(root, (document,))

    assert any("unknown Make target: missing" in failure for failure in failures)
    assert any("TESTS path does not exist" in failure for failure in failures)


def test_accepts_multiple_make_targets_and_test_nodes(tmp_path: Path) -> None:
    root, document = _write_fixture(
        tmp_path,
        "`make check unit TESTS=tests/unit/test_example.py::test_example`\n",
    )

    assert validate_documents(root, (document,)) == []


def test_flags_fenced_command_missing_required_variable(tmp_path: Path) -> None:
    makefile = (
        "deploy:\n"
        '\t@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }\n'
        "\techo done\n"
    )
    root, document = _write_fixture(
        tmp_path,
        "```sh\nmake deploy\n```\n",
        makefile=makefile,
    )

    failures = validate_documents(root, (document,))

    assert any("make target deploy requires DATASET" in f for f in failures)


def test_accepts_fenced_command_with_required_variable(tmp_path: Path) -> None:
    makefile = (
        "deploy:\n"
        '\t@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }\n'
        "\techo done\n"
    )
    root, document = _write_fixture(
        tmp_path,
        "```sh\nmake deploy DATASET=mathematical-benchmarks-v1\n```\n",
        makefile=makefile,
    )

    assert validate_documents(root, (document,)) == []


def test_does_not_flag_inline_prose_mention_of_required_variable_target(
    tmp_path: Path,
) -> None:
    makefile = (
        "deploy:\n"
        '\t@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }\n'
        "\techo done\n"
    )
    root, document = _write_fixture(
        tmp_path,
        "Use `make deploy` for shared tooling.\n",
        makefile=makefile,
    )

    assert validate_documents(root, (document,)) == []


def test_flags_chained_required_variables(tmp_path: Path) -> None:
    makefile = (
        "publish:\n"
        '\t@test -n "$(LOCK)" -a -n "$(DEST)" || '
        '{ echo "LOCK and DEST are required" >&2; exit 2; }\n'
        "\techo done\n"
    )
    root, document = _write_fixture(
        tmp_path,
        "```sh\nmake publish LOCK=snapshot.json\n```\n",
        makefile=makefile,
    )

    failures = validate_documents(root, (document,))

    assert any("make target publish requires DEST" in f for f in failures)
    assert not any("make target publish requires LOCK" in f for f in failures)


def test_flags_required_variables_in_make_prerequisites(tmp_path: Path) -> None:
    makefile = (
        "harbor-check-task:\n"
        '\t@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }\n'
        '\t@test -n "$(TASKS)" || { echo "TASKS is required" >&2; exit 2; }\n'
        "harbor-oracle-task: harbor-check-task\n"
        "\techo run\n"
    )
    root, document = _write_fixture(
        tmp_path,
        "```sh\nmake harbor-oracle-task\n```\n",
        makefile=makefile,
    )

    failures = validate_documents(root, (document,))

    assert any("make target harbor-oracle-task requires DATASET" in f for f in failures)
    assert any("make target harbor-oracle-task requires TASKS" in f for f in failures)
