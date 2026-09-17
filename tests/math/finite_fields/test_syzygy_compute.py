"""Producer-side finite-field syzygy leaves (#964).

Covers the characteristic-p gradient, quotient normal forms with replayable
ledgers, and bounded-degree syzygy-slice bases, all against the frozen Graf
example plus independent brute-force replays written separately in the tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields import (
    Axis,
    FiniteFieldElement,
    FiniteFieldPresentation,
    check_jacobian_syzygy,
    finite_field,
    jacobian_compute,
    quotient_reduce,
    syzygy_generators,
    verify_jacobian,
    verify_quotient_reduction,
    verify_syzygy_generators,
)
from jacobian.math.finite_fields._algebraic_sets import (
    AlgebraicMonomial,
    AlgebraicPolynomial,
)
from jacobian.math.finite_fields._syzygy_compute_models import (
    FiniteFieldJacobianRequest,
    FiniteFieldJacobianResult,
    QuotientReduceRequest,
    QuotientReduceResult,
    SyzygyGeneratorsRequest,
    SyzygyGeneratorsResult,
)
from jacobian.math.finite_fields._tools import TOOLS


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _poly(
    presentation: FiniteFieldPresentation,
    axis: Axis,
    terms: dict[tuple[int, ...], int],
) -> AlgebraicPolynomial:
    canonical = tuple(
        AlgebraicMonomial._from_kernel(
            coefficient=FiniteFieldElement(
                presentation=presentation, coordinates=(coefficient,)
            ),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(terms.items(), reverse=True)
    )
    if not canonical:
        canonical = (
            AlgebraicMonomial._from_kernel(
                coefficient=FiniteFieldElement(
                    presentation=presentation, coordinates=(0,)
                ),
                exponents=(0,) * len(axis.labels),
            ),
        )
    return AlgebraicPolynomial._from_kernel(
        presentation=presentation, variable_axis=axis, terms=canonical
    )


def _term_map(polynomial: AlgebraicPolynomial) -> dict[tuple[int, ...], int]:
    return {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }


def _graf() -> tuple[FiniteFieldPresentation, Axis, AlgebraicPolynomial]:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x", "y", "z"))
    f = _poly(presentation, axis, {(0, 0, 2): 1, (3, 0, 0): 1, (0, 5, 0): 1})
    return presentation, axis, f


def _naive_partials(
    terms: dict[tuple[int, ...], int], prime: int, nvars: int
) -> list[dict[tuple[int, ...], int]]:
    """Independent formal differentiation written separately for the tests."""

    partials: list[dict[tuple[int, ...], int]] = [{} for _ in range(nvars)]
    for exponents, coefficient in terms.items():
        for index in range(nvars):
            if exponents[index] % prime == 0:
                continue
            reduced = list(exponents)
            reduced[index] -= 1
            key = tuple(reduced)
            partials[index][key] = (
                partials[index].get(key, 0) + coefficient * exponents[index]
            ) % prime
    return [
        {key: value for key, value in partial.items() if value} for partial in partials
    ]


class TestJacobianCompute:
    def test_graf_gradient_matches_char_two_jacobian(self) -> None:
        _, _, f = _graf()

        result = jacobian_compute(f)

        assert [_term_map(partial) for partial in result.partials] == [
            {(2, 0, 0): 1},
            {(0, 4, 0): 1},
            {},
        ]
        assert result.characteristic == 2
        assert verify_jacobian(result)

    def test_gradient_agrees_with_independent_differentiation(self) -> None:
        presentation = finite_field(3, (0, 1))
        axis = Axis(name="vars", labels=("x", "y"))
        # 2x^3 + xy^2 over F3 differentiates to (y^2, xy).
        f = _poly(presentation, axis, {(3, 0): 2, (1, 2): 1})

        result = jacobian_compute(f)
        expected = _naive_partials(_term_map(f), 3, 2)

        assert [_term_map(partial) for partial in result.partials] == [
            {(0, 2): 1},
            {(1, 1): 2},
        ]
        assert [_term_map(partial) for partial in result.partials] == expected

    def test_p_power_monomials_vanish(self) -> None:
        presentation = finite_field(5, (0, 1))
        axis = Axis(name="v", labels=("t",))
        f = _poly(presentation, axis, {(5,): 3, (1,): 2})

        (partial,) = jacobian_compute(f).partials

        assert _term_map(partial) == {(0,): 2}

    def test_native_and_catalog_paths_agree(self) -> None:
        _, _, f = _graf()
        tool = _tool("finite_field.polynomial.jacobian.compute")

        assert tool.run(FiniteFieldJacobianRequest(polynomial=f)) == jacobian_compute(f)

    def test_round_trip_and_forgery(self) -> None:
        _, _, f = _graf()
        result = jacobian_compute(f)
        restored = FiniteFieldJacobianResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_jacobian(restored)
        forged = json.loads(restored.model_dump_json())
        forged["characteristic"] = 3
        with pytest.raises(ValidationError):
            FiniteFieldJacobianResult.model_validate_json(json.dumps(forged))
        forged_partials = json.loads(restored.model_dump_json())
        forged_partials["partials"][0], forged_partials["partials"][1] = (
            forged_partials["partials"][1],
            forged_partials["partials"][0],
        )
        swapped = FiniteFieldJacobianResult.model_validate_json(
            json.dumps(forged_partials)
        )
        assert not verify_jacobian(swapped)

    def test_extension_field_is_rejected(self) -> None:
        presentation = finite_field(2, (1, 1, 1))
        axis = Axis(name="vars", labels=("x",))
        f = AlgebraicPolynomial._from_kernel(
            presentation=presentation,
            variable_axis=axis,
            terms=(
                AlgebraicMonomial._from_kernel(
                    coefficient=FiniteFieldElement(
                        presentation=presentation, coordinates=(1, 0)
                    ),
                    exponents=(2,),
                ),
            ),
        )

        with pytest.raises(OperationDomainValidationError) as exc_info:
            jacobian_compute(f)
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.jacobian_syzygy_prime_field_only"
        )


class TestQuotientReduce:
    def test_graf_generator_reduces_to_zero_in_one_step(self) -> None:
        _, _, f = _graf()

        result = quotient_reduce(f, (f,), "z")

        assert _term_map(result.remainder) == {}
        assert len(result.steps) == 1
        assert verify_quotient_reduction(result)

    def test_z_power_four_reduces_by_frobenius(self) -> None:
        presentation, axis, f = _graf()
        fourth = _poly(presentation, axis, {(0, 0, 4): 1})

        result = quotient_reduce(fourth, (f,), "z")

        # z^4 -> z^2(x^3+y^5) -> x^6+x^3y^5+y^10+x^3y^5 -> x^6+y^10
        # in characteristic 2, in three recorded steps.
        assert _term_map(result.remainder) == {(6, 0, 0): 1, (0, 10, 0): 1}
        assert len(result.steps) == 3
        assert verify_quotient_reduction(result)

    def test_equal_quotient_elements_share_remainders(self) -> None:
        presentation, axis, f = _graf()
        # z^2 and x^3+y^5 differ by the generator, so they agree in R.
        square = _poly(presentation, axis, {(0, 0, 2): 1})
        tail = _poly(presentation, axis, {(3, 0, 0): 1, (0, 5, 0): 1})

        reduced_square = quotient_reduce(square, (f,), "z")
        reduced_tail = quotient_reduce(tail, (f,), "z")

        assert _term_map(reduced_square.remainder) == _term_map(reduced_tail.remainder)
        assert _term_map(reduced_tail.remainder) == {(3, 0, 0): 1, (0, 5, 0): 1}

    def test_empty_ideal_is_the_identity(self) -> None:
        _, _, f = _graf()

        result = quotient_reduce(f)

        assert _term_map(result.remainder) == _term_map(f)
        assert result.steps == ()
        assert verify_quotient_reduction(result)

    def test_native_and_catalog_paths_agree(self) -> None:
        _, _, f = _graf()
        tool = _tool("finite_field.quotient.reduce.compute")

        assert tool.run(
            QuotientReduceRequest(
                polynomial=f, ideal_generators=(f,), reduction_variable="z"
            )
        ) == quotient_reduce(f, (f,), "z")

    def test_round_trip_and_forgery(self) -> None:
        _, _, f = _graf()
        result = quotient_reduce(f, (f,), "z")
        restored = QuotientReduceResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_quotient_reduction(restored)
        forged = json.loads(restored.model_dump_json())
        forged["remainder"]["terms"][0]["coefficient"]["coordinates"] = ["1"]
        forged_claim = QuotientReduceResult.model_validate_json(json.dumps(forged))
        assert not verify_quotient_reduction(forged_claim)
        forged_step = json.loads(restored.model_dump_json())
        forged_step["steps"][0]["coefficient"] = 0
        with pytest.raises(ValidationError):
            QuotientReduceResult.model_validate_json(json.dumps(forged_step))

    def test_non_monic_generator_is_rejected(self) -> None:
        presentation, axis, f = _graf()
        # yz + x is not monic in z: the top z-power carries y.
        bad = _poly(presentation, axis, {(0, 1, 1): 1, (1, 0, 0): 1})

        with pytest.raises(OperationDomainValidationError) as exc_info:
            quotient_reduce(f, (bad,), "z")
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.jacobian_syzygy_monic_generator"
        )

    def test_missing_reduction_variable_is_rejected(self) -> None:
        _, _, f = _graf()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            quotient_reduce(f, (f,), None)
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.jacobian_syzygy_reduction_variable"
        )


def _naive_mul(
    left: dict[tuple[int, ...], int], right: dict[tuple[int, ...], int]
) -> dict[tuple[int, ...], int]:
    """Naive characteristic-two polynomial multiplication for the tests."""

    out: dict[tuple[int, ...], int] = {}
    for exponents, coefficient in left.items():
        for other, other_coefficient in right.items():
            key = tuple(a + b for a, b in zip(exponents, other, strict=True))
            out[key] = (out.get(key, 0) + coefficient * other_coefficient) % 2
    return {key: value for key, value in out.items() if value}


def _naive_reduce_graf(terms: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    """Naive z^2 = x^3 + y^5 rewriting for the tests (terminates here)."""

    working = dict(terms)
    while True:
        target = next((key for key in working if key[2] >= 2), None)
        if target is None:
            return {key: value for key, value in working.items() if value}
        coefficient = working.pop(target)
        base = (target[0], target[1], target[2] - 2)
        for shift in ((3, 0, 0), (0, 5, 0)):
            key = (base[0] + shift[0], base[1] + shift[1], base[2] + shift[2])
            working[key] = (working.get(key, 0) + coefficient) % 2
            if not working[key]:
                del working[key]


def _naive_gf2_rank(matrix: list[list[int]]) -> int:
    """Naive characteristic-two rank by full elimination, for the tests."""

    dense = [row[:] for row in matrix]
    rank = 0
    pivot_row = 0
    for column in range(len(dense[0]) if dense else 0):
        pivot = next(
            (
                candidate
                for candidate in range(pivot_row, len(dense))
                if dense[candidate][column]
            ),
            None,
        )
        if pivot is None:
            continue
        dense[pivot_row], dense[pivot] = dense[pivot], dense[pivot_row]
        for other in range(len(dense)):
            if other != pivot_row and dense[other][column]:
                dense[other] = [
                    (a + b) % 2
                    for a, b in zip(dense[other], dense[pivot_row], strict=True)
                ]
        rank += 1
        pivot_row += 1
    return rank


class TestSyzygyGenerators:
    def test_graf_slice_contains_both_issue_generators(self) -> None:
        _, _, f = _graf()

        result = syzygy_generators(f, (f,), "z", 4)

        rows = [tuple(_term_map(entry) for entry in row) for row in result.rows]
        assert any(a == {} and b == {} and c == {(0, 0, 0): 1} for a, b, c in rows)
        assert any(
            a == {(0, 4, 0): 1} and b == {(2, 0, 0): 1} and c == {} for a, b, c in rows
        )
        assert len(result.rows) == result.unknowns - result.rank
        assert verify_syzygy_generators(result)

    def test_produced_rows_verify_through_the_checker(self) -> None:
        _, _, f = _graf()
        result = syzygy_generators(f, (f,), "z", 2)

        checked = check_jacobian_syzygy(
            polynomial=f,
            rows=result.rows,
            ideal_generators=(f,),
            reduction_variable="z",
        )

        assert checked.status == "VERIFIED"

    def test_degree_zero_slice_is_forced(self) -> None:
        # Constants (c1, c2, c3) with c1 x^2 + c2 y^4 = 0 force c1 = c2 = 0.
        _, _, f = _graf()
        result = syzygy_generators(f, (f,), "z", 0)

        assert result.unknowns == 3
        assert result.rank == 2
        assert len(result.rows) == 1
        ((first, second, third),) = result.rows
        assert _term_map(first) == {}
        assert _term_map(second) == {}
        assert _term_map(third) == {(0, 0, 0): 1}

    def test_plain_ring_slice(self) -> None:
        # xy over F2 has Jacobian (y, x); (x, y) is a degree-one syzygy.
        presentation = finite_field(2, (0, 1))
        axis = Axis(name="vars", labels=("x", "y"))
        f = _poly(presentation, axis, {(1, 1): 1})

        result = syzygy_generators(f, (), None, 1)

        rows = [tuple(_term_map(entry) for entry in row) for row in result.rows]
        assert any(a == {(1, 0): 1} and b == {(0, 1): 1} for a, b in rows)
        assert len(result.rows) == result.unknowns - result.rank
        assert verify_syzygy_generators(result)

    def test_zero_jacobian_slice_spans_the_full_space(self) -> None:
        # x^2 over F2 has vanishing partials, so the slice is the kernel of the
        # zero map: the full coordinate space, not the empty family.
        presentation = finite_field(2, (0, 1))
        axis = Axis(name="vars", labels=("x", "y"))
        f = _poly(presentation, axis, {(2, 0): 1})

        result = syzygy_generators(f, (), None, 0)

        assert result.unknowns == 2
        assert result.rank == 0
        assert len(result.rows) == result.unknowns - result.rank
        rows = {
            tuple(tuple(sorted(_term_map(entry).items())) for entry in row)
            for row in result.rows
        }
        assert rows == {
            ((((0, 0), 1),), ()),
            ((), (((0, 0), 1),)),
        }
        assert verify_syzygy_generators(result)
        restored = SyzygyGeneratorsResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_rows_match_an_independent_nullspace(self) -> None:
        # Rebuild the slice linear system with separately written GF(2)
        # elimination and compare nullities; every produced row must satisfy
        # the naive equations.
        _, _, f = _graf()
        degree = 2
        result = syzygy_generators(f, (f,), "z", degree)

        nvars = 3
        monomials: list[tuple[int, ...]] = []
        for total in range(degree + 1):
            for x in range(total + 1):
                for y in range(total + 1 - x):
                    monomials.append((x, y, total - x - y))
        jacobian: list[dict[tuple[int, ...], int]] = [
            {(2, 0, 0): 1},
            {(0, 4, 0): 1},
            {},
        ]
        equations: dict[tuple[int, ...], int] = {}
        columns: list[dict[int, int]] = []
        for component in range(nvars):
            for monomial in monomials:
                image = _naive_reduce_graf(
                    _naive_mul({monomial: 1}, jacobian[component])
                )
                column: dict[int, int] = {}
                for exponents, coefficient in image.items():
                    row = equations.setdefault(exponents, len(equations))
                    column[row] = coefficient
                columns.append(column)
        unknowns = nvars * len(monomials)
        dense = [[0] * unknowns for _ in range(len(equations))]
        for column_index, column in enumerate(columns):
            for row_index, coefficient in column.items():
                dense[row_index][column_index] = coefficient
        assert unknowns - _naive_gf2_rank(dense) == len(result.rows)
        for syzygy_row in result.rows:
            combined: dict[tuple[int, ...], int] = {}
            for entry_map, partial in zip(
                [_term_map(component) for component in syzygy_row],
                jacobian,
                strict=True,
            ):
                for exponents, coefficient in _naive_mul(entry_map, partial).items():
                    combined[exponents] = (combined.get(exponents, 0) + coefficient) % 2
            assert _naive_reduce_graf(combined) == {}

    def test_native_and_catalog_paths_agree(self) -> None:
        _, _, f = _graf()
        tool = _tool("finite_field.jacobian_syzygy.generators.compute")

        assert tool.run(
            SyzygyGeneratorsRequest(
                polynomial=f,
                ideal_generators=(f,),
                reduction_variable="z",
                max_syzygy_degree=2,
            )
        ) == syzygy_generators(f, (f,), "z", 2)

    def test_round_trip_and_forgery(self) -> None:
        _, _, f = _graf()
        result = syzygy_generators(f, (f,), "z", 2)
        restored = SyzygyGeneratorsResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_syzygy_generators(restored)
        forged = json.loads(restored.model_dump_json())
        forged["rank"] = result.rank + 1
        with pytest.raises(ValidationError):
            SyzygyGeneratorsResult.model_validate_json(json.dumps(forged))
        forged_rows = json.loads(restored.model_dump_json())
        forged_rows["rows"] = forged_rows["rows"][::-1]
        reversed_claim = SyzygyGeneratorsResult.model_validate_json(
            json.dumps(forged_rows)
        )
        assert not verify_syzygy_generators(reversed_claim)

    def test_term_bound_is_a_resource_boundary(self) -> None:
        presentation = finite_field(2, (0, 1))
        axis = Axis(name="vars", labels=("x",))
        too_many = _poly(presentation, axis, {(index,): 1 for index in range(65)})

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            jacobian_compute(too_many)
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.jacobian_syzygy_term_bound"
        )

    def test_degree_bound_is_a_resource_boundary(self) -> None:
        presentation = finite_field(2, (0, 1))
        axis = Axis(name="vars", labels=("x",))
        too_steep = _poly(presentation, axis, {(33,): 1})

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            jacobian_compute(too_steep)
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.jacobian_syzygy_degree_bound"
        )

    def test_slice_degree_bound_is_rejected(self) -> None:
        _, _, f = _graf()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            syzygy_generators(f, (f,), "z", 9)
        assert (
            exc_info.value.errors()[0]["type"]
            == "finite_field.syzygy_slice_degree_bound"
        )
        with pytest.raises(ValidationError):
            SyzygyGeneratorsRequest(
                polynomial=f,
                ideal_generators=(f,),
                reduction_variable="z",
                max_syzygy_degree=9,
            )
