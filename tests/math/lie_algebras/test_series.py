"""Tests for Lie-algebra derived and lower central series."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraRequest,
    LieDerivedSeriesResult,
    LieLowerCentralSeriesResult,
    LieSubspace,
)
from jacobian.math.lie_algebras.operations import (
    lie_derived_series,
    lie_lower_central_series,
)

F = Fraction


def _algebra(
    basis: tuple[str, ...],
    constants: tuple[tuple[int, int, int, int], ...],
) -> FiniteDimensionalLieAlgebra:
    return FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": list(basis),
            "structure_constants": [
                {"i": i, "j": j, "k": k, "coefficient": {"num": value, "den": 1}}
                for i, j, k, value in constants
            ],
        }
    )


def _dims(terms: tuple[LieSubspace, ...]) -> tuple[int, ...]:
    return tuple(term.generators.row_count for term in terms)


def _rows(term: LieSubspace) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in term.generators.entries
    )


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
ABELIAN_2 = _algebra(("a", "b"), ())
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))
GL2 = _algebra(
    ("a", "b", "c", "d"),
    (
        (0, 1, 1, 1),
        (0, 2, 2, -1),
        (1, 2, 0, 1),
        (1, 2, 3, -1),
        (1, 3, 1, 1),
        (2, 3, 2, -1),
    ),
)
JACOBI_VIOLATOR = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, 1), (1, 2, 0, 1)))


class TestDerivedSeries:
    @pytest.mark.parametrize(
        ("algebra", "dimensions", "solvable"),
        (
            ("sl2", (3,), False),
            ("heisenberg", (3, 1, 0), True),
            ("abelian", (2, 0), True),
            ("affine", (2, 1, 0), True),
            ("gl2", (4, 3), False),
        ),
    )
    def test_dimensions_and_decision(
        self, algebra: str, dimensions: tuple[int, ...], solvable: bool
    ) -> None:
        algebras = {
            "sl2": SL2,
            "heisenberg": HEISENBERG,
            "abelian": ABELIAN_2,
            "affine": AFFINE,
            "gl2": GL2,
        }
        result = lie_derived_series(algebras[algebra])
        assert _dims(result.terms) == dimensions
        assert result.solvable is solvable

    def test_heisenberg_derived_rows(self) -> None:
        result = lie_derived_series(HEISENBERG)
        assert _rows(result.terms[1]) == ((F(0), F(0), F(1)),)
        assert _rows(result.terms[2]) == ()

    def test_gl2_derived_term_is_trace_zero(self) -> None:
        result = lie_derived_series(GL2)
        assert _rows(result.terms[1]) == (
            (F(1), F(0), F(0), F(-1)),
            (F(0), F(1), F(0), F(0)),
            (F(0), F(0), F(1), F(0)),
        )

    def test_sl2_is_perfect(self) -> None:
        """sl2 equals its own commutator: the series fixes at the head."""
        result = lie_derived_series(SL2)
        assert len(result.terms) == 1
        assert result.solvable is False


class TestLowerCentralSeries:
    @pytest.mark.parametrize(
        ("algebra", "dimensions", "nilpotent"),
        (
            ("sl2", (3,), False),
            ("heisenberg", (3, 1, 0), True),
            ("abelian", (2, 0), True),
            ("affine", (2, 1), False),
            ("gl2", (4, 3), False),
        ),
    )
    def test_dimensions_and_decision(
        self, algebra: str, dimensions: tuple[int, ...], nilpotent: bool
    ) -> None:
        algebras = {
            "sl2": SL2,
            "heisenberg": HEISENBERG,
            "abelian": ABELIAN_2,
            "affine": AFFINE,
            "gl2": GL2,
        }
        result = lie_lower_central_series(algebras[algebra])
        assert _dims(result.terms) == dimensions
        assert result.nilpotent is nilpotent

    def test_affine_fixes_at_span_of_b(self) -> None:
        """ax+b is solvable but not nilpotent: the series fixes nonzero."""
        result = lie_lower_central_series(AFFINE)
        assert _rows(result.terms[1]) == ((F(0), F(1)),)
        assert result.nilpotent is False

    def test_series_terms_bind_source_algebra(self) -> None:
        result = lie_lower_central_series(GL2)
        assert all(term.basis == GL2.basis for term in result.terms)
        assert result.algebra == GL2


class TestSeriesRejections:
    @pytest.mark.parametrize(
        "operation", [lie_derived_series, lie_lower_central_series]
    )
    def test_jacobi_violator_rejected(
        self, operation: Callable[[FiniteDimensionalLieAlgebra], Any]
    ) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            operation(JACOBI_VIOLATOR)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_forged_non_descent_rejected(self) -> None:
        import json

        result = lie_derived_series(HEISENBERG)
        forged = json.loads(result.model_dump_json())
        forged["terms"] = [forged["terms"][0], forged["terms"][0]]
        with pytest.raises(ValidationError) as exc_info:
            LieDerivedSeriesResult.model_validate_json(json.dumps(forged))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "lie_algebra.series_descent"
        )

    def test_forged_decision_mismatch_rejected(self) -> None:
        import json

        result = lie_lower_central_series(AFFINE)
        forged = json.loads(result.model_dump_json())
        forged["nilpotent"] = True
        with pytest.raises(ValidationError) as exc_info:
            LieLowerCentralSeriesResult.model_validate_json(json.dumps(forged))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "lie_algebra.series_decision"
        )


class TestSeriesComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = LieAlgebraRequest(algebra=HEISENBERG)
        assert lie_derived_series(request.algebra) == lie_derived_series(HEISENBERG)
        assert lie_lower_central_series(request.algebra) == lie_lower_central_series(
            HEISENBERG
        )

    def test_serialized_results_round_trip(self) -> None:
        derived = lie_derived_series(GL2)
        assert (
            LieDerivedSeriesResult.model_validate_json(derived.model_dump_json())
            == derived
        )
        central = lie_lower_central_series(AFFINE)
        assert (
            LieLowerCentralSeriesResult.model_validate_json(central.model_dump_json())
            == central
        )

    def test_catalog_declares_both_operations_with_valid_examples(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        for operation_id, native in (
            ("lie_algebra.derived_series.compute", lie_derived_series),
            ("lie_algebra.lower_central_series.compute", lie_lower_central_series),
        ):
            tools = {
                tool.operation_id: tool
                for tool in TOOLS
                if tool.operation_id == operation_id
            }
            assert set(tools) == {operation_id}
            tool = tools[operation_id]
            assert tool.examples
            payload = tool.request_type.model_validate_json(
                encode_strict_json(tool.examples[0].input), strict=True
            )
            assert tool.run(payload) == native(payload.algebra)
