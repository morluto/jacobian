"""Exact geometry verifiers retain operational failures across supplied claims."""

import json
from importlib import import_module
from typing import Any, get_type_hints

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError

_CASES = (
    ("exact", "verify_distance_profile", "distance_profile", None),
    ("exact", "verify_distance_graph", "distance_graph", None),
    ("exact", "verify_euclidean_orbit_profile", "euclidean_orbit_profile", None),
    (
        "exact",
        "verify_pinned_line_distance_profile",
        "pinned_line_distance_profile",
        None,
    ),
    (
        "exact.triangle_area_profile",
        "verify_triangle_area_profile",
        "compute_triangle_area_profile",
        "MAX_CANONICAL_RATIONAL_DIGITS",
    ),
    (
        "exact.spanned_line_profile",
        "verify_spanned_line_profile",
        "compute_spanned_line_profile",
        "MAX_CANONICAL_RATIONAL_DIGITS",
    ),
    (
        "exact.pinned_distance",
        "verify_pinned_distance_support_profile",
        "compute_pinned_distance_support_profile",
        "MAX_DISTANCE_INTERMEDIATE_DIGITS",
    ),
    (
        "framework",
        "verify_planar_rigidity_profile",
        "rank_result",
        "MAX_FRAMEWORK_COORDINATE_WORK",
    ),
    ("differential", "verify_lie_derivative", "lie_derivative", None),
)


def _claim(owner: str, verifier: str) -> tuple[Any, Any]:
    module = import_module("jacobian.math.geometry." + owner + ".operations")
    tools = import_module("jacobian.math.geometry." + owner + "._tools").TOOLS
    result_type = get_type_hints(getattr(module, verifier))["claim"]
    tool = next(tool for tool in tools if tool.result_type is result_type)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    return module, tool.run(request)


@pytest.mark.parametrize("owner,verifier,backend,budget", _CASES)
def test_geometry_verifier_preserves_backend_value_error(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    verifier: str,
    backend: str,
    budget: str | None,
) -> None:
    module, claim = _claim(owner, verifier)
    verify = getattr(module, verifier)
    assert verify(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("backend failure")

    monkeypatch.setattr(module, backend, fail)
    with pytest.raises(ValueError, match="backend failure"):
        verify(claim)


@pytest.mark.parametrize(
    "owner,verifier,backend,budget", [case for case in _CASES if case[3]]
)
def test_geometry_verifier_preserves_real_resource_refusal(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    verifier: str,
    backend: str,
    budget: str,
) -> None:
    module, claim = _claim(owner, verifier)
    if owner == "exact.spanned_line_profile":
        # Two points and coordinate-axis lines use a proved reduction. Keep
        # this refusal test on the general noncollinear line-key path.
        configuration = type(claim.configuration).model_validate(
            {
                "points": [
                    {
                        "label": label,
                        "coordinates": [{"num": x, "den": 1}, {"num": y, "den": 1}],
                    }
                    for label, x, y in (("a", 0, 0), ("b", 1, 0), ("c", 0, 1))
                ]
            }
        )
        claim = module.compute_spanned_line_profile(configuration)
    monkeypatch.setattr(module, budget, 0)
    with pytest.raises(OperationResourceAdmissionError):
        getattr(module, verifier)(claim)
