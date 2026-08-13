from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.matrix_operations import MatrixDeterminantRequest

ROOT = Path(__file__).parents[3]


def test_exact_replay_declaration_requires_provider_runtime_factory() -> None:
    with pytest.raises(ValueError, match="provider runtime factory"):
        ExactReplayCheckerDeclaration(
            "matrix.determinant.compute",
            MatrixDeterminantRequest,
            "check_matrix_determinant",
            "matrix.determinant.flint-replay",
        )


def test_exact_domain_checkers_has_no_central_semantic_maps() -> None:
    source = (ROOT / "src/jacobian/exact_domain_checkers.py").read_text(
        encoding="utf-8"
    )

    assert "_ENTRYPOINT_PROVIDER_RUNTIME_KEYS" not in source
    assert "def _checker_supports" not in source
    assert "def _provider_runtime_key" not in source
