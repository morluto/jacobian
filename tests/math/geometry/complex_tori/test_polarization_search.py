"""Bounded polarization search over Neron-Severi lattices (#2448)."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError
from tests.math.geometry.complex_tori._support import (
    quartic_rank_one_torus,
    quartic_rank_zero_torus,
)

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.geometry.complex_tori import (
    polarization_search,
    verify_polarization_search,
)
from jacobian.math.geometry.complex_tori._models import (
    LatticeComplexStructure,
    PolarizationSearchRequest,
    PolarizationSearchResult,
)
from jacobian.math.geometry.complex_tori._tools import TOOLS
from jacobian.math.matrices.values import RationalMatrix


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _rational(entries: tuple[tuple[int, ...], ...]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(
            tuple(CanonicalRational.from_fraction(Fraction(value)) for value in row)
            for row in entries
        )
    )


def _square_torus() -> LatticeComplexStructure:
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2", "e3", "e4"),
        complex_structure=_rational(
            ((0, 0, -1, 0), (0, 0, 0, -1), (1, 0, 0, 0), (0, 1, 0, 0))
        ),
    )


def _elliptic_torus() -> LatticeComplexStructure:
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=_rational(((0, 1), (-1, 0))),
    )


class TestFound:
    def test_elliptic_principal_polarization_is_first(self) -> None:
        result = polarization_search(_elliptic_torus(), 2, 100)

        assert result.status == "FOUND"
        assert result.ns_rank == 1
        assert result.examined == 1
        assert result.form is not None
        assert result.profile is not None
        assert result.profile.outcome.status == "RIEMANN_FORM"
        assert result.profile.form == result.form
        assert verify_polarization_search(result)

    def test_square_torus_finds_a_polarization(self) -> None:
        result = polarization_search(_square_torus(), 1, 100)

        assert result.status == "FOUND"
        assert result.ns_rank == 4
        assert result.profile is not None
        assert result.profile.outcome.status == "RIEMANN_FORM"
        assert result.form is not None
        # The negated standard symplectic form is the principal polarization
        # under the J-transpose-times-E sign convention.
        assert result.form.matrix.entries == (
            (0, 0, -1, 0),
            (0, 0, 0, -1),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
        )
        assert verify_polarization_search(result)

    def test_search_is_deterministic(self) -> None:
        first = polarization_search(_square_torus(), 1, 100)
        second = polarization_search(_square_torus(), 1, 100)

        assert first == second


class TestInfeasible:
    def test_rank_zero_torus_is_unpolarizable(self) -> None:
        result = polarization_search(quartic_rank_zero_torus(), 2, 100)

        assert result.status == "INFEASIBLE"
        assert result.infeasibility_reason == "NS_RANK_ZERO"
        assert result.ns_rank == 0
        assert result.examined == 0
        assert result.total == 1
        assert verify_polarization_search(result)

    def test_indefinite_rank_one_torus_is_unpolarizable(self) -> None:
        # The Bogomolov-Halle-Pazuki-Tanimoto shape: a rank-one lattice whose
        # generator has Hermitian signature (1, 1) rather than a polarization.
        result = polarization_search(quartic_rank_one_torus(), 2, 100)

        assert result.status == "INFEASIBLE"
        assert result.infeasibility_reason == "RANK_ONE_NO_DEFINITE_SIGN"
        assert result.ns_rank == 1
        assert result.examined == 2
        assert verify_polarization_search(result)


class TestUnknown:
    def test_budget_truncation_reports_position(self) -> None:
        result = polarization_search(_square_torus(), 2, 5)

        assert result.status == "UNKNOWN"
        assert result.unknown_reason == "EXAMINATION_BUDGET_EXCEEDED"
        assert result.examined == 5
        assert result.current_coefficients is not None
        assert verify_polarization_search(result)

    def test_rank_one_budget_below_two_stays_unknown(self) -> None:
        # Rank one has two sign candidates; a budget of one must not conclude
        # infeasibility, because the second sign was never examined.
        result = polarization_search(quartic_rank_one_torus(), 2, 1)

        assert result.status == "UNKNOWN"
        assert result.unknown_reason == "EXAMINATION_BUDGET_EXCEEDED"
        assert result.examined == 1
        assert result.current_coefficients == (-1,)
        assert verify_polarization_search(result)


class TestInvalidRequests:
    def test_zero_coefficient_bound_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            polarization_search(_square_torus(), 0, 100)
        assert (
            exc_info.value.errors()[0]["type"]
            == "complex_torus.polarization_search_bound"
        )

    def test_zero_budget_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            polarization_search(_square_torus(), 1, 0)
        assert (
            exc_info.value.errors()[0]["type"]
            == "complex_torus.polarization_search_bound"
        )

    def test_request_validation_rejects_bad_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PolarizationSearchRequest(
                torus=_square_torus(), coefficient_bound=9, examination_budget=100
            )


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        tool = _tool("complex_torus.polarization.find")
        request = PolarizationSearchRequest(
            torus=_elliptic_torus(), coefficient_bound=2, examination_budget=100
        )

        assert tool.run(request) == polarization_search(_elliptic_torus(), 2, 100)

    def test_round_trip_and_verify(self) -> None:
        result = polarization_search(_elliptic_torus(), 2, 100)
        restored = PolarizationSearchResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_polarization_search(restored)

    def test_found_forgery_fails_verify(self) -> None:
        result = polarization_search(_elliptic_torus(), 2, 100)
        assert result.status == "FOUND"
        forged = json.loads(result.model_dump_json())
        forged["examined"] = result.examined + 1
        forged_claim = PolarizationSearchResult.model_validate_json(json.dumps(forged))
        assert not verify_polarization_search(forged_claim)

    def test_infeasible_count_forgery_is_rejected(self) -> None:
        result = polarization_search(quartic_rank_zero_torus(), 2, 100)
        assert result.status == "INFEASIBLE"
        forged = json.loads(result.model_dump_json())
        forged["examined"] = 1
        with pytest.raises(ValidationError):
            PolarizationSearchResult.model_validate_json(json.dumps(forged))

    def test_unknown_forgery_fails_verify(self) -> None:
        result = polarization_search(_square_torus(), 2, 5)
        assert result.status == "UNKNOWN"
        forged = json.loads(result.model_dump_json())
        forged["current_coefficients"] = [0, 0, 0, 1]
        forged_claim = PolarizationSearchResult.model_validate_json(json.dumps(forged))
        assert not verify_polarization_search(forged_claim)
