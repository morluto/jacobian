"""Public Sturm-bound contract checked against independent references."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms.transform_models import (
    SturmBoundRequest,
)
from jacobian.math.number_theory.modular_forms.transform_tools import TOOLS
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

_STURM_TOOL = next(
    tool
    for tool in TOOLS
    if tool.operation_id == "modular_form.space.sturm_bound.compute"
)


def _space(level: int, weight: int, kind: str = "M") -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=weight,
        kind=kind,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("level", "weight", "index", "bound", "sage_precision"),
    [
        # Sage reports precision B+1 because O(q^P) retains indices < P.
        # The usual bound returned by Jacobian is therefore one less.
        (1, 12, 1, 1, 2),
        (11, 2, 12, 2, 3),
        (144, 3, 288, 72, 73),
    ],
)
def test_public_sturm_operation_matches_sage_precision_convention(
    level: int,
    weight: int,
    index: int,
    bound: int,
    sage_precision: int,
) -> None:
    """Check known Sage values and the exact Gamma0 index independently.

    Sage's documented examples give ``sturm_bound() == 73`` for
    ``CuspForms(Gamma0(144), 3)`` and its generic API specifies that this is
    one plus the usual Sturm bound. Its Gamma0(11), weight-2 example gives
    precision 3. For level one, the classical weight-12 bound is 1, hence
    precision 2. The subgroup indices are evaluated here from their explicit
    prime factorizations, independently of Jacobian's index helper.
    """

    expected_index = level
    for prime in {2, 3, 5, 7, 11}:
        if level % prime == 0:
            expected_index = expected_index // prime * (prime + 1)
    assert expected_index == index
    assert sage_precision == bound + 1

    request = SturmBoundRequest(space=_space(level, weight).model_dump())
    result = _STURM_TOOL.run(request)

    assert result.space == request.space
    assert result.index == index
    assert result.bound == bound


def test_public_sturm_operation_preserves_supported_character_parent() -> None:
    space = ModularFormSpace(
        level=4,
        weight=3,
        kind="M",
        character={
            "group": {
                "modulus": 4,
                "unit_residues": [1, 3],
                "character_count": 2,
                "invariant_factors": [2],
                "generators": [3],
                "generator_orders": [2],
                "unit_coordinates": [[0], [1]],
                "exponent": 2,
            },
            "coordinates": [1],
        },
    )
    request = SturmBoundRequest(space=space)
    result = _STURM_TOOL.run(request)

    assert result.space == space
    assert result.index == 6
    assert result.bound == 1


def test_sturm_operation_rejects_unimplemented_coefficient_parent() -> None:
    space = ModularFormSpace(
        level=1,
        weight=12,
        kind="M",
        coefficient_domain={"order": 3},
    )

    with pytest.raises(OperationDomainValidationError):
        _STURM_TOOL.run(SturmBoundRequest(space=space))


def test_sturm_operation_admits_arithmetic_before_computing_bound() -> None:
    too_large_level = _space(10_001, 12)
    with pytest.raises(OperationResourceAdmissionError):
        _STURM_TOOL.run(SturmBoundRequest(space=too_large_level))
