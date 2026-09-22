"""Semantic test tiers collect only files that explicitly own their markers."""

from pathlib import Path

import pytest
from tools.marker_test_roots import SEMANTIC_MARKERS, marker_test_roots


def test_marker_roots_find_decorators_module_marks_and_parameter_marks(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    decorated = tests / "test_decorated.py"
    decorated.write_text(
        "import pytest\n@pytest.mark.property\ndef test_case(): pass\n",
        encoding="utf-8",
    )
    module_marked = tests / "test_module.py"
    module_marked.write_text(
        "import pytest\npytestmark = pytest.mark.exhaustive\n",
        encoding="utf-8",
    )
    parameter_marked = tests / "test_parameter.py"
    parameter_marked.write_text(
        "import pytest\nCASE = pytest.param(1, marks=pytest.mark.scale)\n",
        encoding="utf-8",
    )
    unmarked = tests / "test_unmarked.py"
    unmarked.write_text("def test_case(): pass\n", encoding="utf-8")

    assert marker_test_roots(tests, "property") == (decorated,)
    assert marker_test_roots(tests, "exhaustive") == (module_marked,)
    assert marker_test_roots(tests, "scale") == (parameter_marked,)


def test_live_semantic_markers_have_narrow_owners() -> None:
    for marker in SEMANTIC_MARKERS:
        roots = marker_test_roots(Path("tests"), marker)
        assert roots
        assert all(path.name.startswith("test_") for path in roots)
        assert len(roots) < len(tuple(Path("tests").rglob("test_*.py")))


def test_unknown_semantic_marker_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported semantic marker"):
        marker_test_roots(tmp_path, "slow")
