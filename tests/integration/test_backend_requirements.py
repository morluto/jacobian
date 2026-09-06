"""Optional system runtimes do not disable ordinary library computations."""

from __future__ import annotations

import copy
import shutil

import pytest

from jacobian.backends import BackendUnavailableError, check_backend
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationMatchRequest
from jacobian.dispatch import invoke_operation

SINGULAR_OPERATIONS = (
    "polynomial.ideal.minimal_primes.compute",
    "polynomial.ideal.radical.compute",
    "polynomial.ideal.quotient.compute",
    "polynomial.ideal.saturation.compute",
    "polynomial.map.generic_degree.compute",
)


@pytest.fixture
def without_native_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    original = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: (
            None if command in ("Singular", "qepcad") else original(command)
        ),
    )


def test_default_python_operations_work_without_system_backends(
    without_native_backends: None,
) -> None:
    import sympy

    from jacobian.math import matrices

    assert matrices.determinant(sympy.Matrix([[1, 2], [3, 4]])) == -2
    result = invoke_operation(
        "integer.compute.extended_gcd", {"left": "84", "right": "30"}, Catalog.open()
    )
    assert result.output["gcd"] == "6"
    assert check_backend("singular").status == "MISSING"
    assert check_backend("qepcad").status == "MISSING"


@pytest.mark.parametrize(
    "operation_id",
    (
        *SINGULAR_OPERATIONS,
        "real_algebraic.plane_semialgebraic.component_profile.compute",
    ),
)
def test_native_calls_raise_actionable_error(
    operation_id: str,
    without_native_backends: None,
) -> None:
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    payload = copy.deepcopy(operation.examples[0].input)
    if operation_id not in SINGULAR_OPERATIONS:
        empty = operation.run(operation.request_type.model_validate(payload))
        assert empty.model_dump(mode="json")["outcome"]["components"] == []
        payload["semialgebraic_set"]["sign_conditions"] = [
            {"signs": ["NEGATIVE", "NEGATIVE"]}
        ]
    request = operation.request_type.model_validate(payload)
    with pytest.raises(BackendUnavailableError) as caught:
        operation.run(request)
    backend = "singular" if operation_id in SINGULAR_OPERATIONS else "qepcad"
    assert caught.value.backend == backend
    assert caught.value.required_version == (
        "4.4.x" if backend == "singular" else "B 1.74"
    )
    assert f"apt-get install {backend}" in caught.value.installation
    assert "server operator" in caught.value.installation


def test_discovery_keeps_declarations_when_backends_are_missing(
    without_native_backends: None,
) -> None:
    catalog = Catalog.open()
    declared = {
        item.operation_id: item.runtime_requirements
        for item in catalog.snapshot().operations
        if item.runtime_requirements
    }
    assert declared == {
        **dict.fromkeys(SINGULAR_OPERATIONS, ("singular",)),
        "real_algebraic.plane_semialgebraic.component_profile.compute": ("qepcad",),
    }
    matched = catalog.match(OperationMatchRequest(need="ideal radical", limit=20))
    radical = next(
        item for item in matched.matches if item.operation_id == SINGULAR_OPERATIONS[1]
    )
    assert radical.runtime_requirements == ("singular",)


def test_unknown_requirement_fails_catalog_declaration() -> None:
    operation = Catalog.open().operation(SINGULAR_OPERATIONS[1])
    assert operation is not None
    with pytest.raises(ValueError, match="unknown optional runtime"):
        MathTool(
            operation_id=operation.operation_id,
            title=operation.title,
            description=operation.description,
            request_type=operation.request_type,
            result_type=operation.result_type,
            run=operation.run,
            runtime_requirements=("unknown",),  # type: ignore[arg-type]
        )
