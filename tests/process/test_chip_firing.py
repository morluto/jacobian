"""Kill-isolated chip-firing rejection, cancellation and deadline regressions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("operation", ["stabilize", "q_reduced"])
@pytest.mark.parametrize("surface", ["native", "dispatch"])
def test_unreachable_sink_rejects_in_isolation(operation: str, surface: str) -> None:
    # A regression may restore a genuinely nonterminating loop. The test
    # parent enforces the wall bound even if request checkpoints disappear.
    script = """
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.chip_firing import stabilize, q_reduced
from jacobian.math.graphs.values import SimpleUndirectedGraph
import sys
operation, surface = sys.argv[1:]
g = SimpleUndirectedGraph(vertices=("a","b","c"), edges=(("b","c"),))
try:
    if surface == "native":
        if operation == "stabilize":
            stabilize(g, "a", (0,1,1))
        else:
            q_reduced(g, (0,1,1), "a")
    else:
        payload = {"graph": g.model_dump(mode="json"), "sink": "a"}
        if operation == "stabilize":
            payload = {"configuration": {**payload, "configuration": [0,1,1]}}
        else:
            payload["divisor"] = [0,1,1]
        invoke_operation("graph.chip_firing." + operation + ".compute", payload, Catalog.open())
except OperationDomainValidationError as exc:
    assert exc.errors()[0]["type"] == "chip_firing.requires_connected_graph"
else:
    raise AssertionError("disconnected graph was accepted")
edge = SimpleUndirectedGraph(vertices=("a","b"), edges=(("a","b"),))
assert stabilize(edge, "a", (0,3)).stable == (3,0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, operation, surface],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("operation", ["stabilize", "q_reduced"])
@pytest.mark.parametrize("interrupt", ["cancel", "deadline"])
def test_interrupt_inside_real_iteration(operation: str, interrupt: str) -> None:
    script = """
import sys, time
from jacobian._execution import (request_execution, request_cancellation,
    bind_request_deadline, OperationExecutionCancelledError, OperationExecutionTimeoutError)
from jacobian.math.graphs.chip_firing import stabilize, q_reduced
from jacobian.math.graphs.values import SimpleUndirectedGraph
operation, interrupt = sys.argv[1:]
labels = tuple(f"v{i:02d}" for i in range(50))
g = SimpleUndirectedGraph(vertices=labels, edges=tuple(zip(labels, labels[1:])))
class Signal:
    checks = 0
    def is_set(self):
        self.checks += 1
        if self.checks == 12:
            if interrupt == "deadline":
                bind_request_deadline(time.monotonic() - 1)
            else:
                return True
        return False
signal = Signal()
expected = OperationExecutionCancelledError if interrupt == "cancel" else OperationExecutionTimeoutError
with request_execution(started_at=time.monotonic()), request_cancellation(signal):
    try:
        if operation == "stabilize":
            stabilize(g, labels[0], (0,) * 49 + (1_000_000,))
        else:
            q_reduced(g, (-100,) * 50, labels[0])
    except expected as exc:
        assert "during chip-firing stabilization" in str(exc), str(exc)
    else:
        raise AssertionError("iteration ignored interruption")
assert signal.checks == 12
assert stabilize(g, labels[0], (0,) * 50).total_firings == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, operation, interrupt],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_smith_preserves_inherited_deadline_and_empty_axis() -> None:
    from time import monotonic

    from jacobian._execution import (
        OperationExecutionTimeoutError,
        bind_request_deadline,
        current_request_execution,
        request_execution,
    )
    from jacobian.math.graphs.chip_firing._snf_process import smith_coordinates

    assert smith_coordinates([], []) == ((), ())
    with request_execution(started_at=monotonic()):
        deadline = monotonic() - 1
        bind_request_deadline(deadline)
        with pytest.raises(OperationExecutionTimeoutError):
            smith_coordinates([[2, -1], [-1, 2]], [2, -1])
        execution = current_request_execution()
        assert execution is not None and execution.deadline == deadline


def test_smith_cancels_actual_worker_and_recovers() -> None:
    from threading import Event, Timer

    from jacobian._execution import (
        OperationExecutionCancelledError,
        request_cancellation,
    )
    from jacobian.math.graphs.chip_firing._snf_process import smith_coordinates

    event = Event()
    timer = Timer(0.05, event.set)
    with request_cancellation(event):
        timer.start()
        try:
            with pytest.raises(OperationExecutionCancelledError):
                smith_coordinates([[2, -1], [-1, 2]], [2, -1])
        finally:
            timer.cancel()
            timer.join()
    assert smith_coordinates([[2, -1], [-1, 2]], [2, -1]) == ((1, 3), (0,))


@pytest.mark.parametrize("factors", [["0", "3"], ["-1", "3"], ["2", "3"]])
def test_smith_decoder_requires_positive_divisibility_chain(factors: list[str]) -> None:
    import hashlib

    from jacobian.canonical import encode_strict_json
    from jacobian.math.graphs.chip_firing._snf_process import _decode_projection

    request = b"admitted-source"
    response = encode_strict_json(
        {
            "request_digest": hashlib.sha256(request).hexdigest(),
            "diagonal": factors,
            "coordinates": ["0"],
        }
    )
    with pytest.raises(RuntimeError, match="malformed output") as caught:
        _decode_projection(response, request, 2, True)
    assert "positive" in str(caught.value.__cause__)


def test_smith_bound_covers_adversarial_two_by_two_transforms() -> None:
    # The near-unimodular matrix has huge entries but determinant one. A
    # bound based only on determinant would miss its initial Bezout work.
    script = """
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_decomp
from jacobian.math.graphs.chip_firing._smith_bounds import _two_by_two_bound
for power in (1, 30, 1000):
    n = 10**power
    a = Matrix([[n,n-1],[n+1,n]])
    envelope = _two_by_two_bound((n+1).bit_length(), 1)
    s,u,v = smith_normal_decomp(a, domain=ZZ)
    assert u*a*v == s
    assert abs(u.det()) == abs(v.det()) == 1
    assert max(abs(int(x)).bit_length() for m in (s,u,v) for x in m) <= envelope.bits
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_coordinate_worker_observes_cancellation_during_modular_hnf() -> None:
    script = """
import sys, time
from threading import Event
from jacobian._execution import request_cancellation, OperationExecutionCancelledError
from jacobian.math.graphs.chip_firing._snf_worker import _coordinate_snf
event = Event()
entered = False
def profile(frame, kind, argument):
    global entered
    if kind == "call" and frame.f_code.co_name == "_hermite_normal_form_modulo_D":
        entered = True
        event.set()
with request_cancellation(event):
    sys.setprofile(profile)
    try:
        _coordinate_snf([[2,-1],[-1,2]], [2,-1], time.monotonic()+10)
    except OperationExecutionCancelledError as exc:
        assert entered
        assert "after determinant-modular Hermite form" in str(exc)
    else:
        raise AssertionError("Hermite preprocessing ignored cancellation")
    finally:
        sys.setprofile(None)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("bound_source", [False, True])
def test_worker_admission_diagnostic_is_bound_to_its_source(bound_source: bool) -> None:
    import hashlib

    from jacobian.canonical import encode_strict_json
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.graphs.chip_firing._snf_process import _decode_projection

    request = b"admitted-coordinate-source"
    message = "residual work exceeds the Smith transformation bound"
    response = encode_strict_json(
        {
            "request_digest": hashlib.sha256(
                request if bound_source else b"other"
            ).hexdigest(),
            "resource_error": message,
        }
    )
    if bound_source:
        with pytest.raises(OperationResourceAdmissionError) as caught:
            _decode_projection(response, request, 2, True)
        assert caught.value.errors() == (
            {
                "loc": ("graph",),
                "type": "chip_firing.smith_transform_bound",
                "msg": message,
            },
        )
    else:
        with pytest.raises(RuntimeError, match="malformed output"):
            _decode_projection(response, request, 2, True)
