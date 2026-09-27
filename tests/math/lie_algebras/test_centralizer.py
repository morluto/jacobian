"""Exact common centralizers of finite-dimensional Lie algebras."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieCentralizerResult,
    LieSubspace,
)
from jacobian.math.lie_algebras.operations import lie_subalgebra_centralizer


def _algebra(
    basis: tuple[str, ...], constants: tuple[tuple[int, int, int, int], ...]
) -> FiniteDimensionalLieAlgebra:
    return FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": list(basis),
            "structure_constants": [
                {"i": i, "j": j, "k": k, "coefficient": {"num": c, "den": 1}}
                for i, j, k, c in constants
            ],
        }
    )


def _element(basis: tuple[str, ...], coordinates: tuple[int, ...]) -> LieAlgebraElement:
    return LieAlgebraElement.model_validate(
        {
            "basis": list(basis),
            "coordinates": [
                {"num": coordinate, "den": 1} for coordinate in coordinates
            ],
        }
    )


def _rows(result: LieCentralizerResult) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row)
        for row in result.centralizer.generators.entries
    )


HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))


def test_heisenberg_centralizer_of_x_is_span_x_z() -> None:
    result = lie_subalgebra_centralizer(
        HEISENBERG, (_element(HEISENBERG.basis, (1, 0, 0)),)
    )
    assert _rows(result) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    assert result.centralizer.algebra == HEISENBERG


def test_common_centralizer_of_heisenberg_generators_is_center() -> None:
    result = lie_subalgebra_centralizer(
        HEISENBERG,
        (
            _element(HEISENBERG.basis, (1, 0, 0)),
            _element(HEISENBERG.basis, (0, 1, 0)),
        ),
    )
    assert _rows(result) == ((Fraction(0), Fraction(0), Fraction(1)),)


def test_centralizer_of_empty_family_is_whole_source_algebra() -> None:
    result = lie_subalgebra_centralizer(HEISENBERG, ())
    assert _rows(result) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_affine_centralizer_not_assumed_to_be_an_ideal() -> None:
    # C_g(a)=span(a), a subalgebra that is not an ideal since [b,a]=-b escapes.
    result = lie_subalgebra_centralizer(AFFINE, (_element(AFFINE.basis, (1, 0)),))
    assert _rows(result) == ((Fraction(1), Fraction(0)),)


def test_every_returned_row_satisfies_each_defining_equation() -> None:
    from jacobian.math.lie_algebras.operations import lie_bracket

    elements = (
        _element(HEISENBERG.basis, (1, 2, 0)),
        _element(HEISENBERG.basis, (0, 1, 3)),
    )
    result = lie_subalgebra_centralizer(HEISENBERG, elements)
    # Direct coordinate solution in [x,y]=z: commuting with x+2y forces
    # beta=2*alpha, while commuting with y+3z forces alpha=0; only z remains.
    assert _rows(result) == ((Fraction(0), Fraction(0), Fraction(1)),)
    for row in _rows(result):
        vector = _element(HEISENBERG.basis, tuple(int(x) for x in row))
        for element in elements:
            bracket = lie_bracket(HEISENBERG, vector, element).bracket
            assert all(value.as_fraction() == 0 for value in bracket.coordinates)


def test_common_centralizer_is_closed_under_bracket() -> None:
    from jacobian.math.lie_algebras.operations import lie_bracket

    result = lie_subalgebra_centralizer(
        HEISENBERG, (_element(HEISENBERG.basis, (1, 1, 0)),)
    )
    rows = _rows(result)
    for left in rows:
        for right in rows:
            bracket = lie_bracket(
                HEISENBERG,
                _element(HEISENBERG.basis, tuple(int(x) for x in left)),
                _element(HEISENBERG.basis, tuple(int(x) for x in right)),
            ).bracket
            assert all(value.as_fraction() == 0 for value in bracket.coordinates)


def test_source_bound_wire_value_round_trips() -> None:
    result = lie_subalgebra_centralizer(
        HEISENBERG, (_element(HEISENBERG.basis, (1, 0, 0)),)
    )
    assert LieCentralizerResult.model_validate_json(result.model_dump_json()) == result


def test_result_composes_as_a_subspace_with_the_quotient_operation() -> None:
    from jacobian.math.lie_algebras.operations import lie_quotient

    result = lie_subalgebra_centralizer(
        HEISENBERG, (_element(HEISENBERG.basis, (1, 0, 0)),)
    )
    assert isinstance(result.centralizer, LieSubspace)
    quotient = lie_quotient(result.algebra, result.centralizer, ("ybar",))
    assert quotient.quotient.basis == ("ybar",)
    assert quotient.quotient.structure_constants == ()


def test_element_basis_must_match_source() -> None:
    alien = _element(("u", "v", "w"), (1, 0, 0))
    with pytest.raises(ValueError, match="centralizer elements must use"):
        lie_subalgebra_centralizer(HEISENBERG, (alien,))


def test_above_family_size_bound_is_rejected() -> None:
    vector = _element(HEISENBERG.basis, (1, 0, 0))
    with pytest.raises((ValueError, OperationDomainValidationError)):
        lie_subalgebra_centralizer(HEISENBERG, (vector,) * 4)


def test_entry_height_bound_rejects_before_exact_row_construction() -> None:
    denominator = 10**63
    basis = tuple(f"v{index}" for index in range(8))
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": list(basis),
            "structure_constants": [
                {
                    "i": first,
                    "j": second,
                    "k": 7,
                    "coefficient": {"num": 1, "den": denominator},
                }
                for first in range(7)
                for second in range(first + 1, 7)
            ],
        }
    )
    vector = LieAlgebraElement.model_validate(
        {
            "basis": list(basis),
            "coordinates": [{"num": 1, "den": denominator} for _ in range(7)]
            + [{"num": 0, "den": 1}],
        }
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        lie_subalgebra_centralizer(algebra, (vector,))
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.centralizer_result_height_bound"
    )


def test_public_example_matches_exact_kernel() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.lie_algebras._tools import TOOLS

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "lie_algebra.subalgebra.centralizer.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert _rows(result) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
