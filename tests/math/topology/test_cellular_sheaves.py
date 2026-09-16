"""Tests for cellular_sheaf.from_cover_maps.compute (#1897)."""

from __future__ import annotations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    FromCoverMapsResult,
    SheafField,
    SheafObstructionCode,
    SheafOutcome,
    SheafStalk,
    from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FromCoverMapsRequest,
)

OPERATION_ID = "cellular_sheaf.from_cover_maps.compute"

_INTERVAL = canonical_complex(("a", "b"), (("a", "b"),))
_TRIANGLE = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))


def _cells(complex_: FiniteSimplicialComplex) -> list[tuple[str, ...]]:
    return [face for group in complex_.faces_by_dimension for face in group.faces]


def _native(request: FromCoverMapsRequest) -> FromCoverMapsResult:
    return from_cover_maps(
        request.complex,
        request.coefficient_field,
        request.prime,
        request.stalks,
        request.cover_maps,
    )


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


def _rank_one_request(
    complex_: FiniteSimplicialComplex,
    scalar: str = "1",
    *,
    field: SheafField = SheafField.RATIONAL,
    prime: int | None = None,
) -> FromCoverMapsRequest:
    return FromCoverMapsRequest(
        complex=complex_,
        coefficient_field=field,
        prime=prime,
        stalks=tuple(
            SheafStalk(simplex=cell, basis=("x",)) for cell in _cells(complex_)
        ),
        cover_maps=tuple(
            CoverRestrictionMatrix(source=source, target=target, entries=((scalar,),))
            for source, target in _covers(complex_)
        ),
    )


class TestKnownAnswer:
    def test_constant_rank_one_triangle_sheaf_has_identity_restrictions(
        self,
    ) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        assert sheaf.diamonds == 3
        assert sheaf.comparable_pairs == 12
        for restriction in (
            *sheaf.cover_restrictions,
            *sheaf.derived_restrictions,
        ):
            assert restriction.entries == (("1",),)

    def test_derived_maps_compose_declared_scalars_over_qq(self) -> None:
        request = _rank_one_request(_TRIANGLE, scalar="1")
        maps = {(map_.source, map_.target): map_ for map_ in request.cover_maps}
        scaled = tuple(
            CoverRestrictionMatrix(
                source=source,
                target=target,
                entries=(("3",),) if len(target) == 3 else (("2",),),
            )
            for source, target in maps
        )
        result = _native(request.model_copy(update={"cover_maps": scaled}))
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        derived = {
            (restriction.source, restriction.target): restriction.entries
            for restriction in sheaf.derived_restrictions
        }
        for vertex in (("a",), ("b",), ("c",)):
            assert derived[(vertex, ("a", "b", "c"))] == (("6",),)

    def test_prime_field_reduction_is_exact(self) -> None:
        request = _rank_one_request(_TRIANGLE, field=SheafField.PRIME_FIELD, prime=5)
        scaled = tuple(
            map_.model_copy(
                update={"entries": (("3",),) if len(map_.target) == 3 else (("2",),)}
            )
            for map_ in request.cover_maps
        )
        result = _native(request.model_copy(update={"cover_maps": scaled}))
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        derived = {
            (restriction.source, restriction.target): restriction.entries
            for restriction in sheaf.derived_restrictions
        }
        # 2 * 3 = 6 = 1 mod 5
        assert derived[(("a",), ("a", "b", "c"))] == (("1",),)


class TestBoundaryDegenerate:
    def test_single_vertex_constant_sheaf(self) -> None:
        point = canonical_complex(("a",), (("a",),))
        result = _native(_rank_one_request(point))
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        assert sheaf.cover_restrictions == ()
        assert sheaf.derived_restrictions == ()
        assert sheaf.diamonds == 0
        assert sheaf.comparable_pairs == 0
        assert tuple(stalk.simplex for stalk in sheaf.stalks) == (("a",),)

    def test_zero_rank_stalks_are_first_class(self) -> None:
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
        result = _native(request)
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        assert len(sheaf.cover_restrictions) == 2
        for restriction in sheaf.cover_restrictions:
            assert restriction.entries == ()

    def test_interval_constant_sheaf(self) -> None:
        result = _native(_rank_one_request(_INTERVAL))
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        assert len(sheaf.cover_restrictions) == 2
        assert sheaf.derived_restrictions == ()


class TestAdversarial:
    def test_corrupted_triangle_map_breaks_a_diamond(self) -> None:
        request = _rank_one_request(_TRIANGLE)
        corrupted = tuple(
            map_.model_copy(update={"entries": (("2",),)})
            if (map_.source, map_.target) == (("a", "b"), ("a", "b", "c"))
            else map_
            for map_ in request.cover_maps
        )
        result = _native(request.model_copy(update={"cover_maps": corrupted}))
        assert result.outcome is SheafOutcome.NOT_A_SHEAF
        obstruction = result.obstruction
        assert obstruction is not None
        assert obstruction.code is SheafObstructionCode.NONCOMMUTING_DIAMOND
        diamond = obstruction.diamond
        assert diamond is not None
        assert diamond.source == ("a",)
        assert diamond.target == ("a", "b", "c")
        assert diamond.first_path == (("a",), ("a", "b"), ("a", "b", "c"))
        assert diamond.second_path == (("a",), ("a", "c"), ("a", "b", "c"))
        assert diamond.first_matrix == (("2",),)
        assert diamond.second_matrix == (("1",),)
        assert result.sheaf is None

    def test_missing_cover_map_is_the_first_obstruction(self) -> None:
        request = _rank_one_request(_TRIANGLE)
        result = _native(
            request.model_copy(update={"cover_maps": request.cover_maps[:-1]})
        )
        assert result.outcome is SheafOutcome.NOT_A_SHEAF
        obstruction = result.obstruction
        assert obstruction is not None
        assert obstruction.code is SheafObstructionCode.MISSING_COVER_MAP
        dropped_source, dropped_target = (
            request.cover_maps[-1].source,
            (request.cover_maps[-1].target),
        )
        assert (obstruction.source, obstruction.target) == (
            dropped_source,
            dropped_target,
        )

    def test_wrong_axis_cover_map_is_rejected(self) -> None:
        request = FromCoverMapsRequest(
            complex=_INTERVAL,
            stalks=(
                SheafStalk(simplex=("a",), basis=("x", "y")),
                SheafStalk(simplex=("b",), basis=("x",)),
                SheafStalk(simplex=("a", "b"), basis=("z",)),
            ),
            cover_maps=(
                CoverRestrictionMatrix(
                    source=("a",), target=("a", "b"), entries=(("1",),)
                ),
                CoverRestrictionMatrix(
                    source=("b",), target=("a", "b"), entries=(("1",),)
                ),
            ),
        )
        result = _native(request)
        assert result.outcome is SheafOutcome.NOT_A_SHEAF
        obstruction = result.obstruction
        assert obstruction is not None
        assert obstruction.code is SheafObstructionCode.COVER_MAP_WRONG_AXIS
        assert obstruction.source == ("a",)
        assert obstruction.target == ("a", "b")

    def test_nonprime_modulus_is_a_domain_rejection(self) -> None:
        request = _rank_one_request(_INTERVAL, field=SheafField.PRIME_FIELD, prime=4)
        with pytest.raises(OperationDomainValidationError) as excinfo:
            _native(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.prime_not_admitted"
        )

    def test_missing_prime_is_a_domain_rejection(self) -> None:
        request = _rank_one_request(_INTERVAL, field=SheafField.PRIME_FIELD, prime=None)
        with pytest.raises(OperationDomainValidationError) as excinfo:
            _native(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.prime_required"
        )

    def test_incomplete_stalk_domain_is_a_domain_rejection(self) -> None:
        request = _rank_one_request(_INTERVAL)
        with pytest.raises(OperationDomainValidationError) as excinfo:
            _native(request.model_copy(update={"stalks": request.stalks[:2]}))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.stalk_domain_incomplete"
        )

    def test_cover_map_on_a_non_cover_pair_is_a_domain_rejection(self) -> None:
        request = _rank_one_request(_TRIANGLE)
        forged = (
            *request.cover_maps,
            CoverRestrictionMatrix(
                source=("a",), target=("a", "b", "c"), entries=(("1",),)
            ),
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            _native(request.model_copy(update={"cover_maps": forged}))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.cover_map_not_a_cover"
        )

    def test_inexact_scalar_entry_is_a_domain_rejection(self) -> None:
        request = _rank_one_request(_INTERVAL)
        corrupted = (
            request.cover_maps[0].model_copy(update={"entries": (("1.5",),)}),
            request.cover_maps[1],
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            _native(request.model_copy(update={"cover_maps": corrupted}))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.entry_not_exact"
        )


class TestDefiningInvariant:
    def test_every_diamond_replays_from_the_returned_cover_maps(self) -> None:
        request = _rank_one_request(_TRIANGLE, scalar="2")
        result = _native(request)
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF
        sheaf = result.sheaf
        assert sheaf is not None
        cover = {
            (restriction.source, restriction.target): restriction.entries
            for restriction in sheaf.cover_restrictions
        }
        derived = {
            (restriction.source, restriction.target): restriction.entries
            for restriction in sheaf.derived_restrictions
        }

        def compose(
            first: tuple[tuple[str, ...], tuple[str, ...]],
            second: tuple[tuple[str, ...], tuple[str, ...]],
        ) -> tuple[tuple[str, ...], ...]:
            # ``second`` is the later step: rho(second) @ rho(first).
            left, right = cover[second], cover[first]
            rows = len(left)
            inner = len(right)
            columns = len(right[0]) if inner else 0
            return tuple(
                tuple(
                    str(sum(int(left[i][k]) * int(right[k][j]) for k in range(inner)))
                    for j in range(columns)
                )
                for i in range(rows)
            )

        replayed = 0
        for source, target in derived:
            if len(target) - len(source) != 2:
                continue
            middles = sorted(
                cell
                for cell in _cells(_TRIANGLE)
                if len(cell) == len(source) + 1
                and set(source) < set(cell) < set(target)
            )
            composites = [
                compose((source, middle), (middle, target)) for middle in middles
            ]
            assert all(entries == composites[0] for entries in composites)
            assert derived[(source, target)] == composites[0]
            replayed += 1
        assert replayed == 3


class TestNativeCatalogParity:
    def test_catalog_tool_runs_the_same_kernel(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        request = _rank_one_request(_TRIANGLE)
        assert tool.run(request) == _native(request)

    def test_published_examples_execute(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        outcomes = []
        for example in tool.examples:
            request = FromCoverMapsRequest.model_validate(example.input)
            result = tool.run(request)
            outcomes.append(result.outcome)
        assert outcomes == [
            SheafOutcome.CELLULAR_SHEAF,
            SheafOutcome.NOT_A_SHEAF,
        ]


class TestSerialization:
    def test_sheaf_result_round_trips(self) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        restored = FromCoverMapsResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_negative_result_round_trips(self) -> None:
        request = _rank_one_request(_TRIANGLE)
        corrupted = tuple(
            map_.model_copy(update={"entries": (("2",),)})
            if (map_.source, map_.target) == (("a", "b"), ("a", "b", "c"))
            else map_
            for map_ in request.cover_maps
        )
        result = _native(request.model_copy(update={"cover_maps": corrupted}))
        restored = FromCoverMapsResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_forged_derived_map_fails_validation(self) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        payload = result.model_dump(mode="json")
        payload["sheaf"]["derived_restrictions"][0]["entries"] = [["1", "2"]]
        with pytest.raises(ValueError):
            FromCoverMapsResult.model_validate(payload)

    def test_forged_cover_path_fails_validation(self) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        payload = result.model_dump(mode="json")
        payload["sheaf"]["derived_restrictions"][0]["cover_path"] = [
            ["a"],
            ["a", "b", "c"],
        ]
        with pytest.raises(ValueError):
            FromCoverMapsResult.model_validate(payload)

    def test_forged_outcome_payload_mix_is_rejected(self) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        payload = result.model_dump(mode="json")
        payload["obstruction"] = {
            "code": "MISSING_COVER_MAP",
            "message": "forged",
            "source": ["a"],
            "target": ["a", "b"],
        }
        with pytest.raises(ValueError):
            FromCoverMapsResult.model_validate(payload)

    def test_forged_commutativity_claim_is_rejected(self) -> None:
        result = _native(_rank_one_request(_TRIANGLE))
        payload = result.model_dump(mode="json")
        payload["sheaf"]["stalks"][0]["basis"] = ["forged"]
        with pytest.raises(ValueError):
            FiniteCellularSheaf.model_validate(payload["sheaf"])


class TestEnvelope:
    def _six_simplex_plus_isolated_point(self) -> FiniteSimplicialComplex:
        return canonical_complex(
            ("a", "b", "c", "d", "e", "f", "g"),
            (("a", "b", "c", "d", "e", "f"), ("g",)),
        )

    def test_sixty_four_simplices_with_zero_stalks_is_admitted(self) -> None:
        complex_ = self._six_simplex_plus_isolated_point()
        assert complex_.closure_size == 64
        request = FromCoverMapsRequest(
            complex=complex_,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=()) for cell in _cells(complex_)
            ),
            cover_maps=tuple(
                CoverRestrictionMatrix(source=source, target=target, entries=())
                for source, target in _covers(complex_)
            ),
        )
        result = _native(request)
        assert result.outcome is SheafOutcome.CELLULAR_SHEAF

    def test_above_the_simplex_envelope_is_a_resource_rejection(self) -> None:
        complex_ = canonical_complex(
            ("a", "b", "c", "d", "e", "f", "g"),
            (("a", "b", "c", "d", "e", "f", "g"),),
        )
        assert complex_.closure_size == 127
        request = FromCoverMapsRequest(
            complex=complex_,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=()) for cell in _cells(complex_)
            ),
        )
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            _native(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.admission.simplices"
        )

    def test_stalk_rank_above_the_envelope_is_a_resource_rejection(self) -> None:
        request = FromCoverMapsRequest(
            complex=_INTERVAL,
            stalks=(
                SheafStalk(
                    simplex=("a",),
                    basis=tuple(f"e{index}" for index in range(9)),
                ),
                SheafStalk(simplex=("b",), basis=("x",)),
                SheafStalk(simplex=("a", "b"), basis=("x",)),
            ),
        )
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            _native(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.admission.stalk_rank"
        )

    def test_cover_map_count_above_the_envelope_is_a_resource_rejection(
        self,
    ) -> None:
        request = FromCoverMapsRequest(
            complex=_INTERVAL,
            stalks=tuple(
                SheafStalk(simplex=cell, basis=("x",)) for cell in _cells(_INTERVAL)
            ),
            cover_maps=tuple(
                CoverRestrictionMatrix(
                    source=("a",), target=("a", "b"), entries=(("1",),)
                )
                for _ in range(513)
            ),
        )
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            _native(request)
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.admission.cover_maps"
        )

    def test_entry_digits_above_the_envelope_is_a_resource_rejection(self) -> None:
        request = _rank_one_request(_INTERVAL)
        corrupted = (
            request.cover_maps[0].model_copy(update={"entries": (("1" * 65,),)}),
            request.cover_maps[1],
        )
        with pytest.raises(OperationResourceAdmissionError) as excinfo:
            _native(request.model_copy(update={"cover_maps": corrupted}))
        assert excinfo.value.errors()[0]["type"] == (
            "topology.cellular_sheaf.admission.entry_digits"
        )
