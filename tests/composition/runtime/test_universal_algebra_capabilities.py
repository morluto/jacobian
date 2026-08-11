from __future__ import annotations

from typing import Any

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion
from jacobian.contracts.universal_algebra import (
    MagmaTerm,
    UniversalAlgebraCountermodelSearchRequest,
)
from jacobian.universal_algebra_capabilities import (
    _evaluate_term,
    _z3_evaluate_term,
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


def test_magma_evaluators_reject_malformed_constructed_terms() -> None:
    missing_variable = MagmaTerm.model_construct(kind="VARIABLE")
    missing_children = MagmaTerm.model_construct(kind="PRODUCT")

    with pytest.raises(ValueError, match="only a variable name"):
        _evaluate_term(missing_variable, (), {})
    with pytest.raises(ValueError, match="exactly two child terms"):
        _evaluate_term(missing_children, (), {})
    with pytest.raises(ValueError, match="only a variable name"):
        _z3_evaluate_term(missing_variable, (), {}, 0, None)
    with pytest.raises(ValueError, match="exactly two child terms"):
        _z3_evaluate_term(missing_children, (), {}, 0, None)


def test_countermodel_descriptor_publishes_a_model_valid_invocation_example(
    authorized_complete_runtime,
) -> None:
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
    }
    descriptor = descriptors["universal_algebra.search.countermodel"]

    assert len(descriptor.invocation_examples) == 1
    example = descriptor.invocation_examples[0]
    assert example.mode is CapabilityMode.EXPLORE
    validated = UniversalAlgebraCountermodelSearchRequest.model_validate(example.input)
    assert validated.order == 2
    assert validated.target_law.law_id == "associative"


def test_evaluate_laws_descriptor_example_encodes_idempotence(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
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

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            mode=example.mode,
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
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={"problem": _left_projection_problem()},
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
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
    assert result.output["checker_id"] == (
        authorized_complete_runtime.portfolio.universal_algebra.evaluation_checker_id
    )
    assert result.output["verification_handoff"] == {
        "capability_id": "certificate.verify",
        "mode": "VERIFY",
        "payload": {
            "certificate_uri": result.output["certificate_uri"],
            "checker_id": result.output["checker_id"],
            "timeout_seconds": 150,
        },
    }
    assert "conclusion" not in result.output

    handoff = result.output["verification_handoff"]
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=handoff["capability_id"],
            mode=CapabilityMode(handoff["mode"]),
            input=handoff["payload"],
        )
    )

    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value
    assert verified.output["verification_record_uri"]


def test_complete_request_validation_precedes_artifact_writes(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _left_projection_problem()
    problem["structure"]["table"] = [[0, 0]]
    artifact_put_calls = 0
    original_put = fresh_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(fresh_complete_runtime.core.artifacts, "put", recording_put)

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={"problem": problem},
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_FINITE_MAGMA_LAW_REQUEST"
    assert artifact_put_calls == 0


def test_countermodel_search_composes_with_independent_law_replay(
    authorized_complete_runtime,
) -> None:
    laws = _left_projection_problem()["laws"]

    search = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.search.countermodel",
            input={
                "order": 2,
                "source_laws": [laws[0]],
                "target_law": laws[1],
            },
        )
    )

    assert search.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert search.output["status"] == "WITNESS_FOUND"
    assert search.output["verification"] == "UNVERIFIED"
    assert search.output["target_record"]["holds"] is False
    assert all(record["holds"] for record in search.output["source_records"])

    evaluation = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="universal_algebra.evaluate_laws",
            input={
                "problem": {
                    "problem_schema_version": "1",
                    "structure": search.output["structure"],
                    "laws": laws,
                }
            },
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            mode=CapabilityMode.VERIFY,
            input={
                "certificate_uri": evaluation.output["certificate_uri"],
                "checker_id": evaluation.output["checker_id"],
            },
        )
    )

    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value


def test_countermodel_search_reports_fixed_order_no_witness_without_conclusion(
    fresh_complete_runtime,
) -> None:
    laws = _left_projection_problem()["laws"]

    search = fresh_complete_runtime.core.capabilities.invoke(
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
    assert search.scope.parameters["order"] == 1
    assert "conclusion" not in search.output


def test_countermodel_request_validation_precedes_artifact_writes(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    laws = _left_projection_problem()["laws"]
    duplicate_target = dict(laws[1])
    duplicate_target["law_id"] = laws[0]["law_id"]
    artifact_put_calls = 0
    original_put = fresh_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(fresh_complete_runtime.core.artifacts, "put", recording_put)

    result = fresh_complete_runtime.core.capabilities.invoke(
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
    fresh_complete_runtime,
) -> None:

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 2},
        )
    )

    assert result.output["enumerated_count"] == 16
    assert result.output["total_count"] == 16
    assert result.output["ordering"] == "LEXICOGRAPHIC_ROW_MAJOR"
    assert result.output["completeness"] == "COMPLETE"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    table_payloads = [
        fresh_complete_runtime.core.store.get(uri).payload
        for uri in result.output["table_uris"]
    ]
    assert table_payloads[0]["table"] == [[0, 0], [0, 0]]
    assert table_payloads[-1]["table"] == [[1, 1], [1, 1]]
    assert len({str(payload["table"]) for payload in table_payloads}) == 16
    enumeration = fresh_complete_runtime.core.store.get(
        result.output["enumeration_uri"]
    )
    assert enumeration.payload["table_uris"] == result.output["table_uris"]
    assert set(enumeration.manifest.parents) == set(result.output["table_uris"])


def test_finite_magma_table_enumeration_handles_order_one(
    fresh_complete_runtime,
) -> None:

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 1},
        )
    )

    assert result.output["enumerated_count"] == 1
    table = fresh_complete_runtime.core.store.get(result.output["table_uris"][0])
    assert table.payload["table"] == [[0]]


def test_finite_magma_table_enumeration_rejects_unsupported_order_before_writes(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_put_calls = 0
    original_put = fresh_complete_runtime.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(fresh_complete_runtime.core.artifacts, "put", recording_put)
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="finite_magma.table.enumerate",
            input={"order": 3},
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert artifact_put_calls == 0
