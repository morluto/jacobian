from __future__ import annotations

import json
from types import ModuleType
from typing import get_type_hints

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential import operations as differential
from jacobian.math.geometry.exact import operations as exact_geometry
from jacobian.math.matrices.quadratic_spectral import operations as quadratic_spectral
from jacobian.math.number_theory.sequences.recurrence_solving import (
    operations as recurrence,
)
from jacobian.math.polynomials.real_algebra import operations as real_algebra


@pytest.mark.parametrize(
    ("module", "recompute"),
    [
        (differential, "lie_derivative"),
        (real_algebra, "common_interlacing_profile"),
        (quadratic_spectral, "inertia"),
        (exact_geometry, "distance_profile"),
        (recurrence, "closed_form"),
    ],
)
@pytest.mark.parametrize("resource_refusal", [False, True])
def test_verifiers_propagate_operational_failures(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    recompute: str,
    resource_refusal: bool,
) -> None:
    verifier = getattr(module, f"verify_{recompute}")
    result_type = get_type_hints(verifier)["claim"]
    if module is exact_geometry and recompute == "distance_profile":
        # Native rational-only and catalog union specializations intentionally
        # have distinct Pydantic generic identities.
        tool = next(
            tool
            for tool in BUILTIN_TOOLS
            if tool.operation_id == "geometry.points.distance_profile.compute"
        )
    else:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.result_type is result_type)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    claim = tool.result_type.model_validate_json(result.model_dump_json())
    assert verifier(claim)

    failure = (
        OperationResourceAdmissionError(
            location=("source",),
            code="test.resource_bound",
            message="resource bound exceeded",
        )
        if resource_refusal
        else RuntimeError("backend failed")
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(module, recompute, fail)
    with pytest.raises(type(failure)) as caught:
        verifier(claim)
    assert caught.value is failure
