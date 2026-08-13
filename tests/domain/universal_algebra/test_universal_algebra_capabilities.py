from __future__ import annotations

from typing import Any

import pytest
from tests.support.core_capability_harnesses import UniversalAlgebraTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.universal_algebra import (
    UniversalAlgebraCountermodelSearchRequest,
)


def _variable(name: str) -> dict[str, object]:
    return {"kind": "VARIABLE", "variable": name, "left": None, "right": None}


def _product(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    return {"kind": "PRODUCT", "variable": None, "left": left, "right": right}


def _left_projection_problem() -> dict[str, object]:
    x = _variable("x")
    y = _variable("y")
    z = _variable("z")
    return {
        "problem_schema_version": "1",
        "structure": {
            "structure_schema_version": "1",
            "operation": "binary",
            "order": 2,
            "table": [[0, 0], [1, 1]],
        },
        "laws": [
            {
                "law_id": "associative",
                "variables": ["x", "y", "z"],
                "left": _product(_product(x, y), z),
                "right": _product(x, _product(y, z)),
            },
            {
                "law_id": "commutative",
                "variables": ["x", "y"],
                "left": _product(x, y),
                "right": _product(y, x),
            },
        ],
    }


def test_countermodel_descriptor_publishes_a_model_valid_invocation_example(
    universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = universal_algebra_services.services
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    descriptor = descriptors["universal_algebra.search.countermodel"]

    assert len(descriptor.invocation_examples) == 1
    example = descriptor.invocation_examples[0]
    validated = UniversalAlgebraCountermodelSearchRequest.model_validate(example.input)
    assert validated.order == 2
    assert validated.target_law.law_id == "associative"


def test_evaluate_laws_descriptor_example_encodes_idempotence(
    universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = universal_algebra_services.services
    descriptor = next(
        descriptor
        for descriptor in runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "universal_algebra.evaluate_laws"
    )
    example = descriptor.invocation_examples[0]
    law = example.input["problem"]["laws"][0]

    assert law["law_id"] == "idempotence"
    assert law["left"] == {
        "kind": "PRODUCT",
        "left": {"kind": "VARIABLE", "variable": "x"},
        "right": {"kind": "VARIABLE", "variable": "x"},
    }
    assert law["right"] == {"kind": "VARIABLE", "variable": "x"}

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=example.input,
        )
    )

    assert result.output["records"] == [
        {
            "law_id": "idempotence",
            "holds": True,
            "coverage": "EXHAUSTIVE",
            "checked_valuations": 1,
            "counterexample": None,
        }
    ]


def test_evaluate_laws_returns_exact_truth_and_counterexample(
    universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = universal_algebra_services.services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={"problem": _left_projection_problem()},
        )
    )

    records = {record["law_id"]: record for record in result.output["records"]}
    assert records["associative"] == {
        "law_id": "associative",
        "holds": True,
        "coverage": "EXHAUSTIVE",
        "checked_valuations": 8,
        "counterexample": None,
    }
    assert records["commutative"] == {
        "law_id": "commutative",
        "holds": False,
        "coverage": "COUNTEREXAMPLE_FOUND",
        "checked_valuations": 2,
        "counterexample": {
            "assignment": [
                {"variable": "x", "value": 0},
                {"variable": "y", "value": 1},
            ],
            "left_value": 0,
            "right_value": 1,
        },
    }
    assert result.output["certificate_uri"] in result.artifact_uris
    assert "verification_handoff" not in result.output
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.law_evaluation.verify",
            input={"certificate_uri": result.output["certificate_uri"]},
        )
    )
    assert verified.verification_record_uri is not None
    assert "conclusion" not in result.output


def test_complete_request_validation_precedes_artifact_writes(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    problem = _left_projection_problem()
    problem["structure"]["table"] = [[0, 0]]
    artifact_put_calls = 0
    original_put = runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(runtime.core.artifacts, "put", recording_put)

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={"problem": problem},
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_FINITE_MAGMA_LAW_REQUEST"
    assert artifact_put_calls == 0


def test_countermodel_search_reports_fixed_order_no_witness_without_conclusion(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    laws = _left_projection_problem()["laws"]

    search = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.search.countermodel",
            input={
                "order": 1,
                "source_laws": [laws[0]],
                "target_law": laws[1],
            },
        )
    )

    assert search.execution.status.value == "COMPLETED"
    assert search.output["status"] == "NO_WITNESS_FOUND"
    assert search.output["structure"] is None
    assert "conclusion" not in search.output


def test_countermodel_request_validation_precedes_artifact_writes(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    laws = _left_projection_problem()["laws"]
    duplicate_target = dict(laws[1])
    duplicate_target["law_id"] = laws[0]["law_id"]
    artifact_put_calls = 0
    original_put = runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(runtime.core.artifacts, "put", recording_put)

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.search.countermodel",
            input={
                "order": 2,
                "source_laws": [laws[0]],
                "target_law": duplicate_target,
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_FINITE_MAGMA_COUNTERMODEL_REQUEST"
    assert artifact_put_calls == 0


def test_finite_magma_table_enumeration_is_exact_and_canonical(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 2},
        )
    )

    assert result.output["enumerated_count"] == 16
    assert result.output["total_count"] == 16
    assert result.output["ordering"] == "LEXICOGRAPHIC_ROW_MAJOR"
    table_payloads = [
        runtime.core.store.get(uri).payload for uri in result.output["table_uris"]
    ]
    assert table_payloads[0]["table"] == [[0, 0], [0, 0]]
    assert table_payloads[-1]["table"] == [[1, 1], [1, 1]]
    assert len({str(payload["table"]) for payload in table_payloads}) == 16
    enumeration = runtime.core.store.get(result.output["enumeration_uri"])
    assert enumeration.payload["table_uris"] == result.output["table_uris"]
    assert set(enumeration.manifest.parents) == set(result.output["table_uris"])


def test_finite_magma_table_enumeration_handles_order_one(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 1},
        )
    )

    assert result.output["enumerated_count"] == 1
    table = runtime.core.store.get(result.output["table_uris"][0])
    assert table.payload["table"] == [[0]]


def test_finite_magma_table_enumeration_rejects_unsupported_order_before_writes(
    unauthorized_universal_algebra_services: UniversalAlgebraTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = unauthorized_universal_algebra_services.services
    artifact_put_calls = 0
    original_put = runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(runtime.core.artifacts, "put", recording_put)
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 3},
        )
    )

    assert result.execution.status.value == "ERROR"
    assert (
        result.diagnostics[0].code == "INVALID_FINITE_MAGMA_TABLE_ENUMERATION_REQUEST"
    )
    assert artifact_put_calls == 0
