"""Public native oriented-matroid API contract."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids import oriented as oriented_matroids
from jacobian.math.combinatorics.matroids.oriented import (
    BracketPolynomial,
    bracket_syzygy_residual,
    grassmann_pluecker_relation,
)


def test_public_api_exports_bracket_operations_and_values() -> None:
    assert oriented_matroids.__all__ == [
        "BracketMonomial",
        "BracketPolynomial",
        "BracketPolynomialTerm",
        "BracketSyzygyResidualRequest",
        "CanonicalBracket",
        "GrassmannPlueckerRelation",
        "GrassmannPlueckerRelationRequest",
        "GrassmannPlueckerRelationResult",
        "bracket_syzygy_residual",
        "check_chirotope",
        "grassmann_pluecker_relation",
        "verify_chirotope_check",
    ]


def test_native_pluecker_and_syzygy_compose_without_private_imports() -> None:
    relation = grassmann_pluecker_relation(6, (0, 1, 2, 3, 4, 5), "FOUR_TERM")
    residual = bracket_syzygy_residual(BracketPolynomial(ground_size=6, terms=()), ())
    assert relation.polynomial.ground_size == 6
    assert residual.terms == ()
