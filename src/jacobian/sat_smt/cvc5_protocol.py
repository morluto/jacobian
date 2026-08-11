"""Typed, closed protocol shared by the isolated cvc5 worker and its adapter."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt, TypeAdapter, ValidationError

from jacobian.contracts.results import ContractModel

CVC5_WORKER_PROTOCOL = "jacobian.cvc5-worker/v1"


class Cvc5SatisfiableWorkerResult(ContractModel):
    protocol: Literal["jacobian.cvc5-worker/v1"]
    solver_status: Literal["SATISFIABLE"]
    proof_written: Literal[False]
    alethe_hole_count: None


class Cvc5UnsatisfiableWorkerResult(ContractModel):
    protocol: Literal["jacobian.cvc5-worker/v1"]
    solver_status: Literal["UNSATISFIABLE"]
    proof_written: Literal[True]
    alethe_hole_count: StrictInt = Field(ge=0, le=1_000_000)


class Cvc5UnknownWorkerResult(ContractModel):
    protocol: Literal["jacobian.cvc5-worker/v1"]
    solver_status: Literal["UNKNOWN"]
    proof_written: Literal[False]
    alethe_hole_count: None


type Cvc5WorkerResult = Annotated[
    Cvc5SatisfiableWorkerResult
    | Cvc5UnsatisfiableWorkerResult
    | Cvc5UnknownWorkerResult,
    Field(discriminator="solver_status"),
]

_RESULT_ADAPTER: TypeAdapter[Cvc5WorkerResult] = TypeAdapter(Cvc5WorkerResult)


def parse_cvc5_worker_result(value: object) -> Cvc5WorkerResult:
    """Parse one complete worker result, rejecting every unrepresentable state."""

    try:
        return _RESULT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid cvc5 worker result") from exc
