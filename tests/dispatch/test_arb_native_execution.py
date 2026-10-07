"""Native Arb owners retain execution controls after real evaluation."""

from __future__ import annotations

import json
import time
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian import _execution
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import execute_operation
from jacobian.math.analysis import (
    expression_enclosure,
    operations,
    second_jet_enclosure,
)
from jacobian.math.analysis._expression_enclosure import (
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
)
from jacobian.math.analysis._second_jet import (
    IntervalExpressionSecondJetEnclosureRequest,
    IntervalExpressionSecondJetEnclosureResult,
)
from jacobian.math.analysis._tools import (
    EXPRESSION_ENCLOSURE_OPERATIONS,
    SECOND_JET_ENCLOSURE_OPERATIONS,
)


@pytest.fixture(scope="module")
def arb_catalog() -> Catalog:
    return Catalog((*EXPRESSION_ENCLOSURE_OPERATIONS, *SECOND_JET_ENCLOSURE_OPERATIONS))


@pytest.mark.parametrize("case", ["point", "point-domain", "second-jet"])
@pytest.mark.parametrize("control", ["timeout", "cancelled", "none"])
@pytest.mark.parametrize("surface", ["native", "dispatch"])
def test_arb_native_and_dispatch_observe_controls_after_actual_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    arb_catalog: Catalog,
    case: str,
    control: str,
    surface: str,
) -> None:
    is_jet = case == "second-jet"
    variable: dict[str, Any] = {"op": "var"}
    if is_jet:
        variable["variable"] = "x"
    expression: dict[str, Any] = {
        "op": "log" if case == "point-domain" else "pow",
        "children": [variable],
    }
    if case != "point-domain":
        expression["exponent"] = 2
    payload: dict[str, Any] = {"expression": expression, "precision_bits": 128}
    if is_jet:
        payload["box"] = {
            "variables": ["x"],
            "intervals": [
                {"lower": {"num": "1", "den": "1"}, "upper": {"num": "1", "den": "1"}}
            ],
        }
    else:
        payload["argument"] = {
            "num": "-1" if case == "point-domain" else "2",
            "den": "1",
        }

    started = time.monotonic()
    deadline = started + 30
    clock = [started]
    cancellation = Event()
    evaluated: list[object] = []
    name = "_evaluate_second_jet" if is_jet else "_evaluate_expression"
    evaluate = getattr(operations, name)

    def complete_evaluation(*args: Any, **kwargs: Any) -> Any:
        result = evaluate(*args, **kwargs)
        if args[0].op == expression["op"]:
            evaluated.append(result)
            # Keep the actual Arb answer or domain failure. Only the request
            # control changes after the root's real evaluation has completed.
            if control == "timeout":
                clock[0] = deadline + 1
            elif control == "cancelled":
                cancellation.set()
        return result

    monkeypatch.setattr(operations, name, complete_evaluation)
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def run() -> dict[str, Any]:
        if surface == "dispatch":
            operation_id = (
                "interval.expression.second_jet_enclosure.compute"
                if is_jet
                else "interval.compute.enclosure"
            )
            return execute_operation(
                operation_id,
                payload,
                arb_catalog,
                projector=lambda _operation_id, result, _runtime_ms: result.model_dump(
                    mode="json"
                ),
                cancellation_signal=cancellation,
            )
        if is_jet:
            jet_request = (
                IntervalExpressionSecondJetEnclosureRequest.model_validate_json(
                    json.dumps(payload)
                )
            )
            return second_jet_enclosure(
                jet_request.expression, jet_request.box, jet_request.precision_bits
            ).model_dump(mode="json")
        request = IntervalExpressionEnclosureRequest.model_validate_json(
            json.dumps(payload)
        )
        return expression_enclosure(
            request.expression, request.argument, request.precision_bits
        ).model_dump(mode="json")

    with request_execution(
        started, outer_deadline=deadline, cancellation_signal=cancellation
    ):
        if control == "none":
            output = run()
            assert output["status"] == (
                "DOMAIN_ERROR" if case == "point-domain" else "ENCLOSED"
            )
            if case == "point":
                point = IntervalExpressionEnclosureResult.model_validate_json(
                    json.dumps(output)
                )
                assert point.lower is not None and point.upper is not None
                assert point.lower.as_fraction() == point.upper.as_fraction() == 4
            elif is_jet:
                jet = IntervalExpressionSecondJetEnclosureResult.model_validate_json(
                    json.dumps(output)
                )
                assert jet.value is not None
                assert (
                    jet.value.lower.as_fraction() == jet.value.upper.as_fraction() == 1
                )
                assert jet.gradient[0].enclosure.lower.as_fraction() == 2
                assert jet.gradient[0].enclosure.upper.as_fraction() == 2
                assert jet.hessian[0].enclosure.lower.as_fraction() == 2
                assert jet.hessian[0].enclosure.upper.as_fraction() == 2
        else:
            expected = (
                OperationExecutionTimeoutError
                if control == "timeout"
                else OperationExecutionCancelledError
            )
            with pytest.raises(expected):
                run()

    assert len(evaluated) == 1
    assert _execution.current_request_execution() is None
