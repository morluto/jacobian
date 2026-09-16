"""Tests for homological.spectral_sequence.page.compute (#3755)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_SPECTRAL_PAGE,
    AssociatedGradedResult,
    FilteredSubspace,
    FiltrationLevel,
    SpectralPageRequest,
    SpectralPageResult,
    SpectralPageStatus,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    associated_graded,
    spectral_page,
)
from jacobian.math.topology.chain_complexes._tools import TOOLS
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

OPERATION_ID = "homological.spectral_sequence.page.compute"


def _tool():
    return next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)


def _level(*degree_vectors: tuple[tuple[str, ...], ...]) -> FiltrationLevel:
    return FiltrationLevel(
        subspaces=tuple(
            FilteredSubspace(vectors=tuple(vectors)) for vectors in degree_vectors
        )
    )


def _two_step_complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        prime=None,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=((("1",),),),
    )


def _two_step_filtration() -> tuple[FiltrationLevel, ...]:
    return (
        _level((("1",),), ()),
        _level((("1",),), (("1",),)),
    )


def _three_step_complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        prime=None,
        degree_min=0,
        degree_max=2,
        basis_sizes=(1, 1, 1),
        differential_matrices=((("1",),), (("0",),)),
    )


def _three_step_filtration() -> tuple[FiltrationLevel, ...]:
    return (
        _level((("1",),), (), ()),
        _level((("1",),), (("1",),), ()),
        _level((("1",),), (("1",),), (("1",),)),
    )


def _native(
    complex_value: ChainComplexValue,
    filtration: tuple[FiltrationLevel, ...],
    page: int,
) -> SpectralPageResult:
    return spectral_page(complex_value, filtration, page)


def _differentials_by_source(
    result: SpectralPageResult,
) -> dict[tuple[int, int], tuple[tuple[str, ...], ...]]:
    return {
        (record.source_level, record.source_degree): record.entries
        for record in result.differentials
    }


def _mat_mul_fractions(
    left: tuple[tuple[str, ...], ...], right: tuple[tuple[str, ...], ...]
) -> list[list[Fraction]]:
    left_parsed = [[Fraction(entry) for entry in row] for row in left]
    right_parsed = [[Fraction(entry) for entry in row] for row in right]
    if not left_parsed or not right_parsed or not right_parsed[0]:
        return [[] for _ in range(len(left_parsed))]
    columns = list(zip(*right_parsed, strict=True))
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in columns]
        for row in left_parsed
    ]


class TestKnownAnswer:
    def test_two_step_page_zero_is_the_associated_graded(self) -> None:
        complex_value = _two_step_complex()
        filtration = _two_step_filtration()
        graded = associated_graded(complex_value, filtration)
        page = _native(complex_value, filtration, 0)
        assert page.page_dimensions == graded.graded_dimensions
        assert page.page_representatives == graded.quotient_representatives
        assert page.page_status is SpectralPageStatus.ACTIVE

    def test_two_step_page_one_carries_the_connecting_differential(self) -> None:
        page = _native(_two_step_complex(), _two_step_filtration(), 1)
        assert page.page_dimensions == ((1, 0), (0, 1))
        assert _differentials_by_source(page) == {(1, 1): (("1",),)}
        assert page.next_page_dimensions == ((0, 0), (0, 0))
        assert page.page_status is SpectralPageStatus.ACTIVE

    def test_two_step_page_two_collapses_and_stabilizes(self) -> None:
        page = _native(_two_step_complex(), _two_step_filtration(), 2)
        assert page.page_dimensions == ((0, 0), (0, 0))
        assert page.next_page_dimensions == ((0, 0), (0, 0))
        assert page.page_status is SpectralPageStatus.STABILIZED

    def test_three_step_page_two_stabilizes_on_the_surviving_class(self) -> None:
        first = _native(_three_step_complex(), _three_step_filtration(), 1)
        assert first.page_dimensions == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        assert _differentials_by_source(first)[(1, 1)] == (("1",),)
        assert _differentials_by_source(first)[(2, 2)] == (("0",),)
        assert first.next_page_dimensions == ((0, 0, 0), (0, 0, 0), (0, 0, 1))
        second = _native(_three_step_complex(), _three_step_filtration(), 2)
        assert second.page_dimensions == ((0, 0, 0), (0, 0, 0), (0, 0, 1))
        assert second.page_status is SpectralPageStatus.STABILIZED

    def test_first_page_is_the_homology_of_page_zero(self) -> None:
        complex_value = _three_step_complex()
        filtration = _three_step_filtration()
        zero = _native(complex_value, filtration, 0)
        one = _native(complex_value, filtration, 1)
        assert zero.next_page_dimensions == one.page_dimensions


class TestDefiningInvariant:
    def test_page_differentials_square_to_zero(self) -> None:
        for page_number in (0, 1, 2):
            result = _native(
                _three_step_complex(), _three_step_filtration(), page_number
            )
            by_source = {
                (record.source_level, record.source_degree): record
                for record in result.differentials
            }
            replayed = 0
            for record in result.differentials:
                follower = by_source.get((record.target_level, record.target_degree))
                if follower is None:
                    continue
                product = _mat_mul_fractions(follower.entries, record.entries)
                assert all(value == 0 for row in product for value in row), (
                    f"d^2 != 0 on page {page_number}"
                )
                replayed += 1
            assert replayed == len(result.differential_squared_zero)

    def test_page_dimensions_shrink_toward_the_limit(self) -> None:
        filtration = _two_step_filtration()
        complex_value = _two_step_complex()
        totals = [
            sum(
                sum(row)
                for row in _native(complex_value, filtration, r).page_dimensions
            )
            for r in (0, 1, 2)
        ]
        assert totals == [2, 2, 0]

    def test_single_level_zero_page_with_zero_differential_stabilizes(self) -> None:
        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )
        filtration = (_level((("1",),), (("1",),)),)
        page = _native(complex_value, filtration, 0)
        assert page.page_dimensions == ((1, 1),)
        assert page.page_status is SpectralPageStatus.STABILIZED


class TestBoundaryDegenerate:
    def test_zero_differential_two_step_page_one_stabilizes(self) -> None:
        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )
        page = _native(complex_value, _two_step_filtration(), 1)
        assert page.page_dimensions == ((1, 0), (0, 1))
        assert _differentials_by_source(page) == {(1, 1): (("0",),)}
        assert page.page_status is SpectralPageStatus.STABILIZED
        assert page.next_page_dimensions == page.page_dimensions

    def test_ceiling_page_with_long_filtration_is_truncated(self) -> None:
        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("0",),),),
        )
        filtration = (
            *(_level((), ()) for _ in range(5)),
            _level((("1",),), (("1",),)),
        )
        page = _native(complex_value, filtration, MAX_SPECTRAL_PAGE)
        assert page.page_status is SpectralPageStatus.TRUNCATED
        assert page.max_page == MAX_SPECTRAL_PAGE
        assert page.level_count == 6

    def test_prime_field_two_step_page(self) -> None:
        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.PRIME_FIELD,
            prime=5,
            degree_min=0,
            degree_max=1,
            basis_sizes=(1, 1),
            differential_matrices=((("1",),),),
        )
        page = _native(complex_value, _two_step_filtration(), 1)
        assert page.page_dimensions == ((1, 0), (0, 1))
        assert _differentials_by_source(page) == {(1, 1): (("1",),)}


class TestAdversarial:
    def test_over_page_request_is_refused(self) -> None:
        with pytest.raises(OperationResourceAdmissionError):
            _native(_two_step_complex(), _two_step_filtration(), MAX_SPECTRAL_PAGE + 1)

    def test_over_page_wire_request_fails_schema_validation(self) -> None:
        with pytest.raises(ValidationError):
            SpectralPageRequest(
                complex=_two_step_complex(),
                filtration=_two_step_filtration(),
                page=MAX_SPECTRAL_PAGE + 1,
            )

    def test_negative_page_is_a_domain_rejection(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        with pytest.raises(OperationDomainValidationError):
            _native(_two_step_complex(), _two_step_filtration(), -1)

    def test_non_preserving_differential_is_rejected(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        complex_value = _two_step_complex()
        # The differential sends the bottom C_1 generator (absent) nowhere,
        # but the top-level representative below breaks preservation: F_0 C_1
        # is spanned by the C_1 generator while F_0 C_0 is zero.
        filtration = (
            _level((), (("1",),)),
            _level((("1",),), (("1",),)),
        )
        with pytest.raises(OperationDomainValidationError):
            _native(complex_value, filtration, 1)

    def test_non_complex_source_is_rejected(self) -> None:
        from jacobian.catalog.models import OperationDomainValidationError

        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=2,
            basis_sizes=(1, 1, 1),
            differential_matrices=((("1",),), (("1",),)),
        )
        filtration = (
            _level((), (), ()),
            _level((("1",),), (("1",),), (("1",),)),
        )
        with pytest.raises(OperationDomainValidationError):
            _native(complex_value, filtration, 1)

    def test_over_rank_ambient_is_refused(self) -> None:
        size = 33
        zeros = tuple(("0",) * size for _ in range(size))
        complex_value = ChainComplexValue(
            coefficient_ring=CoefficientRing.RATIONAL,
            prime=None,
            degree_min=0,
            degree_max=1,
            basis_sizes=(size, size),
            differential_matrices=(zeros,),
        )
        identity = tuple(
            tuple("1" if i == j else "0" for j in range(size)) for i in range(size)
        )
        filtration = (
            _level(tuple(() for _ in range(size)), tuple(() for _ in range(size))),
            _level(identity, identity),
        )
        with pytest.raises(OperationResourceAdmissionError):
            _native(complex_value, filtration, 1)


class TestNativeCatalogParity:
    def test_tool_runs_the_same_kernel(self) -> None:
        tool = _tool()
        request = SpectralPageRequest(
            complex=_two_step_complex(),
            filtration=_two_step_filtration(),
            page=1,
        )
        assert tool.run(request) == _native(
            request.complex, request.filtration, request.page
        )

    def test_published_examples_execute(self) -> None:
        tool = _tool()
        for example in tool.examples:
            request = SpectralPageRequest.model_validate(example.input)
            result = tool.run(request)
            assert isinstance(result, SpectralPageResult)


class TestSerialization:
    def test_page_result_round_trips(self) -> None:
        result = _native(_two_step_complex(), _two_step_filtration(), 1)
        restored = SpectralPageResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_stabilized_result_round_trips(self) -> None:
        result = _native(_two_step_complex(), _two_step_filtration(), 2)
        restored = SpectralPageResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_forged_differential_shape_fails_validation(self) -> None:
        result = _native(_two_step_complex(), _two_step_filtration(), 1)
        payload = result.model_dump(mode="json")
        payload["differentials"][0]["rows"] = 7
        with pytest.raises(ValidationError):
            SpectralPageResult.model_validate(payload)

    def test_forged_status_window_fails_validation(self) -> None:
        result = _native(_two_step_complex(), _two_step_filtration(), 1)
        payload = result.model_dump(mode="json")
        payload["max_page"] = 99
        with pytest.raises(ValidationError):
            SpectralPageResult.model_validate(payload)


class TestConsumerComposition:
    def test_associated_graded_output_feeds_the_page_unchanged(self) -> None:
        graded = associated_graded(_three_step_complex(), _three_step_filtration())
        assert isinstance(graded, AssociatedGradedResult)
        first = spectral_page(graded.complex, graded.filtration, 1)
        direct = _native(_three_step_complex(), _three_step_filtration(), 1)
        assert first == direct

    def test_strict_json_request_round_trip(self) -> None:
        request = SpectralPageRequest(
            complex=_two_step_complex(),
            filtration=_two_step_filtration(),
            page=1,
        )
        restored = SpectralPageRequest.model_validate_json(request.model_dump_json())
        assert restored == request
        assert _tool().run(restored).page_status is SpectralPageStatus.ACTIVE
