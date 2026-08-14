"""Architecture checker contracts for unsupported-surface scanning."""

from __future__ import annotations

from pathlib import Path

from tools.check_architecture import check_architecture


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_K = "knowledge"
_S = "search"
_RM = "Research" + "Memory"
_RE = "Research" + "Episode"
_KS_DOT = _K + "." + _S
_CAP = "capab" + "ility"


def test_research_memory_import_in_src_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/operation_dispatcher.py",
        f"from jacobian.memory import {_RM}\n",
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1


def test_research_memory_import_in_tests_is_flagged(tmp_path: Path) -> None:
    """Unsupported surfaces are scanned in tests too, not just src."""
    _write(
        tmp_path,
        "tests/composition/test_memory.py",
        f"from jacobian.memory import {_RM}\n",
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1
    assert surf[0].path == "tests/composition/test_memory.py"


def test_knowledge_search_string_in_src_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/builtin_operations.py",
        f'operation_id = "{_KS_DOT}"\n',
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert _KS_DOT in surf[0].message


def test_knowledge_search_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/search.md",
        f"# Search\n\nUse {_KS_DOT} for retrieval.\n",
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert surf[0].path == "docs/explanation/search.md"


def test_knowledge_search_in_schema_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/schemas/test-schema.json",
        f'{{"operation_id": "{_KS_DOT}"}}\n',
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) == 1
    assert surf[0].path == "benchmarks/schemas/test-schema.json"


def test_research_episode_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/episodes.md",
        f"# Episodes\n\nThe {_RE} store records past work.\n",
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1
    assert surf[0].path == "docs/explanation/episodes.md"


def test_research_memory_prose_in_docs_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/memory.md",
        "# Memory\n\nThe Research memory store is deprecated.\n",
    )
    report = check_architecture(tmp_path)
    surf = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(surf) >= 1


def test_generic_memory_word_is_not_flagged(tmp_path: Path) -> None:
    """Legitimate generic words like 'memory limits' must not trigger."""
    _write(
        tmp_path,
        "src/jacobian/bounded_process.py",
        "import os\n\n# Configure memory limits for child processes.\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)


def test_changelog_is_excluded_from_surface_scan(tmp_path: Path) -> None:
    _write(tmp_path, "CHANGELOG.md", f"# Changelog\n\nAdded {_KS_DOT} operation.\n")
    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)


def test_generated_results_and_local_evidence_are_excluded(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "benchmarks/results/run/config.json",
        f'{{"tool_name_profile": "{_CAP}"}}\n',
    )
    _write(tmp_path, "tmp/rootcause-analysis/report.md", f"# {_CAP}\n")
    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)


def test_removed_capability_vocabulary_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/jacobian/contracts/old.py",
        f"class Legacy{_CAP.title()}:\n    pass\n",
    )
    report = check_architecture(tmp_path)
    violations = [v for v in report.violations if v.code == "unsupported-surface"]
    assert len(violations) == 1
    assert violations[0].path == "src/jacobian/contracts/old.py"


def test_operation_migration_design_may_name_removed_contracts(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/explanation/operation-runtime-target.md",
        f"# Migration\n\nThe old `{_CAP}_id` field is removed.\n",
    )
    report = check_architecture(tmp_path)
    assert all(v.code != "unsupported-surface" for v in report.violations)
