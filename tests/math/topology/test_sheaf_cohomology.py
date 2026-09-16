"""Tests for cellular_sheaf.cohomology.compute (#3755)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafField,
    from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FromCoverMapsRequest,
    SheafCohomologyRequest,
    SheafCohomologyResult,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves._tools import TOOLS
from jacobian.math.topology.cohomology.operations import simplicial_cohomology

OPERATION_ID = "cellular_sheaf.cohomology.compute"

_INTERVAL = canonical_complex(("a", "b"), (("a", "b"),))
_CIRCLE = canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
_TRIANGLE = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
_POINT = canonical_complex(("a",), (("a",),))


def _tool():
    return next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)


def _cells(complex_: FiniteSimplicialComplex) -> list[tuple[str, ...]]:
    return [face for group in complex_.faces_by_dimension for face in group.faces]


def _covers(
    complex_: FiniteSimplicialComplex,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    known = set(_cells(complex_))
    covers = []
    for coface in known:
        for position in range(len(coface)):
            face = coface[:position] + coface[position + 1 :]
            if face in known:
                covers.append((face, coface))
    covers.sort()
    return covers


def _constant_sheaf(
    complex_: FiniteSimplicialComplex,
    *,
    field: SheafField = SheafField.RATIONAL,
    prime: int | None = None,
) -> FiniteCellularSheaf:
    request = FromCoverMapsRequest(
        complex=complex_,
        coefficient_field=field,
        prime=prime,
        stalks=tuple(
            SheafStalk(simplex=cell, basis=("x",)) for cell in _cells(complex_)
        ),
        cover_maps=tuple(
            CoverRestrictionMatrix(source=source, target=target, entries=(("1",),))
            for source, target in _covers(complex_)
        ),
    )
    result = from_cover_maps(
        request.complex,
        request.coefficient_field,
        request.prime,
        request.stalks,
        request.cover_maps,
    )
    assert result.sheaf is not None
    return result.sheaf


def _native(sheaf: FiniteCellularSheaf) -> SheafCohomologyResult:
    from jacobian.math.topology.cellular_sheaves.operations import sheaf_cohomology

    return sheaf_cohomology(sheaf)


def _bettis(result: SheafCohomologyResult) -> list[int]:
    return [group.betti_number for group in result.groups]


def _parse_matrix(
    matrix: tuple[tuple[str, ...], ...], prime: int | None
) -> list[list[Fraction | int]]:
    if prime is None:
        return [[Fraction(entry) for entry in row] for row in matrix]
    return [[int(entry) % prime for entry in row] for row in matrix]


def _mat_mul(
    left: list[list[Fraction | int]],
    right: list[list[Fraction | int]],
    prime: int | None,
) -> list[list[Fraction | int]]:
    if not left or not right or not right[0]:
        return [[] for _ in range(len(left))]
    columns = list(zip(*right, strict=True))
    product = []
    for row in left:
        rendered = []
        for column in columns:
            total: Fraction | int = sum(a * b for a, b in zip(row, column, strict=True))
            if prime is not None:
                total = int(total) % prime
            rendered.append(total)
        product.append(rendered)
    return product


def _mat_vec(
    matrix: list[list[Fraction | int]],
    vector: list[Fraction | int],
    prime: int | None,
) -> list[Fraction | int]:
    result = []
    for row in matrix:
        total: Fraction | int = sum(a * b for a, b in zip(row, vector, strict=True))
        if prime is not None:
            total = int(total) % prime
        result.append(total)
    return result


class TestKnownAnswer:
    def test_interval_constant_sheaf_is_acyclic_above_zero(self) -> None:
        result = _native(_constant_sheaf(_INTERVAL))
        assert result.cochain_dimensions == (2, 1)
        assert _bettis(result) == [1, 0]

    def test_circle_constant_sheaf_has_first_cohomology(self) -> None:
        result = _native(_constant_sheaf(_CIRCLE))
        assert result.cochain_dimensions == (3, 3)
        assert _bettis(result) == [1, 1]

    def test_filled_triangle_constant_sheaf_is_acyclic_above_zero(self) -> None:
        result = _native(_constant_sheaf(_TRIANGLE))
        assert _bettis(result) == [1, 0, 0]

    def test_point_constant_sheaf(self) -> None:
        result = _native(_constant_sheaf(_POINT))
        assert result.cochain_dimensions == (1,)
        assert _bettis(result) == [1]
        assert result.coboundary_matrices == ()

    def test_skyscraper_sheaf_on_a_vertex(self) -> None:
        basis_for = {("a",): ("x",), ("b",): (), ("a", "b"): ()}
        request = FromCoverMapsRequest(
            complex=_INTERVAL,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=basis_for[cell])
                for cell in _cells(_INTERVAL)
            ),
            cover_maps=(
                CoverRestrictionMatrix(source=("a",), target=("a", "b"), entries=()),
                CoverRestrictionMatrix(source=("b",), target=("a", "b"), entries=()),
            ),
        )
        result = from_cover_maps(
            request.complex,
            request.coefficient_field,
            request.prime,
            request.stalks,
            request.cover_maps,
        )
        assert result.sheaf is not None
        cohomology = _native(result.sheaf)
        assert cohomology.cochain_dimensions == (1, 0)
        assert _bettis(cohomology) == [1, 0]

    def test_zero_stalks_give_zero_cohomology(self) -> None:
        request = FromCoverMapsRequest(
            complex=_INTERVAL,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=()) for cell in _cells(_INTERVAL)
            ),
            cover_maps=tuple(
                CoverRestrictionMatrix(source=source, target=target, entries=())
                for source, target in _covers(_INTERVAL)
            ),
        )
        result = from_cover_maps(
            request.complex,
            request.coefficient_field,
            request.prime,
            request.stalks,
            request.cover_maps,
        )
        assert result.sheaf is not None
        cohomology = _native(result.sheaf)
        assert _bettis(cohomology) == [0, 0]


class TestIndependentOracle:
    @pytest.mark.parametrize(
        ("complex_", "expected"),
        [(_INTERVAL, [1, 0]), (_CIRCLE, [1, 1]), (_TRIANGLE, [1, 0, 0])],
    )
    def test_prime_field_constant_sheaf_matches_simplicial_cohomology(
        self, complex_: FiniteSimplicialComplex, expected: list[int]
    ) -> None:
        prime = 5
        result = _native(
            _constant_sheaf(complex_, field=SheafField.PRIME_FIELD, prime=prime)
        )
        oracle = simplicial_cohomology(complex_, prime)
        assert _bettis(result) == expected
        assert [group.betti_number for group in oracle.groups] == expected


class TestDefiningInvariant:
    def test_coboundaries_square_to_zero(self) -> None:
        for complex_ in (_INTERVAL, _CIRCLE, _TRIANGLE):
            result = _native(_constant_sheaf(complex_))
            prime = result.sheaf.prime
            replayed = 0
            for degree in range(len(result.coboundary_matrices) - 1):
                upper = _parse_matrix(result.coboundary_matrices[degree + 1], prime)
                lower = _parse_matrix(result.coboundary_matrices[degree], prime)
                product = _mat_mul(upper, lower, prime)
                assert all(value == 0 for row in product for value in row)
                replayed += 1
            assert replayed == len(result.differential_squared_zero)

    def test_representative_cocycles_lie_in_the_kernel(self) -> None:
        result = _native(_constant_sheaf(_CIRCLE))
        prime = result.sheaf.prime
        for degree, group in enumerate(result.groups):
            assert len(group.cocycle_representatives) == group.betti_number
            if degree < len(result.coboundary_matrices):
                outgoing = _parse_matrix(result.coboundary_matrices[degree], prime)
                for cocycle in group.cocycle_representatives:
                    residual = _mat_vec(
                        outgoing, list(_parse_matrix((cocycle,), prime)[0]), prime
                    )
                    assert all(value == 0 for value in residual)

    def test_euler_characteristic_is_consistent(self) -> None:
        for complex_ in (_INTERVAL, _CIRCLE, _TRIANGLE, _POINT):
            result = _native(_constant_sheaf(complex_))
            stalk_sum = sum(
                len(stalk.basis) if degree % 2 == 0 else -len(stalk.basis)
                for degree, faces in enumerate(
                    [group.faces for group in complex_.faces_by_dimension]
                )
                for stalk in [
                    next(
                        stalk for stalk in result.sheaf.stalks if stalk.simplex == face
                    )
                    for face in faces
                ]
            )
            assert result.euler_characteristic_stalk == stalk_sum
            assert result.euler_characteristic_cohomology == sum(
                group.betti_number if group.degree % 2 == 0 else -group.betti_number
                for group in result.groups
            )


class TestAdversarial:
    def test_composite_modulus_is_a_domain_rejection(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        sheaf = _constant_sheaf(_INTERVAL, field=SheafField.PRIME_FIELD, prime=5)
        forged = sheaf.model_copy(update={"prime": 4})
        with pytest.raises(OperationDomainValidationError):
            _native(forged)

    def test_incomplete_restriction_diagram_is_rejected(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        sheaf = _constant_sheaf(_TRIANGLE)
        forged = sheaf.model_copy(
            update={"derived_restrictions": sheaf.derived_restrictions[1:]}
        )
        with pytest.raises(OperationDomainValidationError):
            _native(forged)

    def test_inexact_restriction_entry_is_rejected(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        sheaf = _constant_sheaf(_INTERVAL)
        restrictions = tuple(
            restriction.model_copy(update={"entries": (("1.5",),)})
            if index == 0
            else restriction
            for index, restriction in enumerate(sheaf.cover_restrictions)
        )
        forged = sheaf.model_copy(update={"cover_restrictions": restrictions})
        with pytest.raises(OperationDomainValidationError):
            _native(forged)

    def test_stalk_rank_above_the_envelope_is_a_resource_rejection(self) -> None:
        from jacobian.catalog.models import OperationResourceAdmissionError

        sheaf = _constant_sheaf(_INTERVAL)
        stalks = (
            SheafStalk(simplex=("a",), basis=tuple(f"e{index}" for index in range(9))),
            *sheaf.stalks[1:],
        )
        forged = FiniteCellularSheaf.model_construct(
            complex=sheaf.complex,
            coefficient_field=sheaf.coefficient_field,
            prime=sheaf.prime,
            stalks=stalks,
            cover_restrictions=sheaf.cover_restrictions,
            derived_restrictions=sheaf.derived_restrictions,
            diamonds=sheaf.diamonds,
            comparable_pairs=sheaf.comparable_pairs,
        )
        with pytest.raises(OperationResourceAdmissionError):
            _native(forged)


class TestNativeCatalogParity:
    def test_tool_runs_the_same_kernel(self) -> None:
        tool = _tool()
        request = SheafCohomologyRequest(sheaf=_constant_sheaf(_CIRCLE))
        assert tool.run(request) == _native(request.sheaf)

    def test_published_examples_execute(self) -> None:
        tool = _tool()
        for example in tool.examples:
            request = SheafCohomologyRequest.model_validate(example.input)
            result = tool.run(request)
            assert isinstance(result, SheafCohomologyResult)
            assert _bettis(result) == [1, 0]


class TestSerialization:
    def test_cohomology_result_round_trips(self) -> None:
        result = _native(_constant_sheaf(_CIRCLE))
        restored = SheafCohomologyResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_forged_coboundary_shape_fails_validation(self) -> None:
        result = _native(_constant_sheaf(_INTERVAL))
        payload = result.model_dump(mode="json")
        payload["coboundary_matrices"][0] = [["1", "1", "1"]]
        with pytest.raises(ValidationError):
            SheafCohomologyResult.model_validate(payload)

    def test_forged_euler_claim_fails_validation(self) -> None:
        result = _native(_constant_sheaf(_INTERVAL))
        payload = result.model_dump(mode="json")
        payload["euler_characteristic_cohomology"] += 1
        with pytest.raises(ValidationError):
            SheafCohomologyResult.model_validate(payload)

    def test_forged_representative_count_fails_validation(self) -> None:
        result = _native(_constant_sheaf(_CIRCLE))
        payload = result.model_dump(mode="json")
        payload["groups"][1]["cocycle_representatives"] = []
        with pytest.raises(ValidationError):
            SheafCohomologyResult.model_validate(payload)


class TestConsumerComposition:
    def test_cover_map_output_feeds_cohomology_unchanged(self) -> None:
        request = FromCoverMapsRequest(
            complex=_CIRCLE,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=("x",)) for cell in _cells(_CIRCLE)
            ),
            cover_maps=tuple(
                CoverRestrictionMatrix(source=source, target=target, entries=(("1",),))
                for source, target in _covers(_CIRCLE)
            ),
        )
        result = from_cover_maps(
            request.complex,
            request.coefficient_field,
            request.prime,
            request.stalks,
            request.cover_maps,
        )
        assert result.sheaf is not None
        cohomology = _native(result.sheaf)
        assert _bettis(cohomology) == [1, 1]
        wire = SheafCohomologyRequest(sheaf=result.sheaf)
        assert _tool().run(wire) == cohomology

    def test_strict_json_request_round_trip(self) -> None:
        request = SheafCohomologyRequest(sheaf=_constant_sheaf(_INTERVAL))
        restored = SheafCohomologyRequest.model_validate_json(request.model_dump_json())
        assert restored == request
