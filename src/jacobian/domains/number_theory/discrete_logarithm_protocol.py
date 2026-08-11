"""Closed request and result envelopes for the discrete-logarithm worker."""

from __future__ import annotations

from typing import Literal

from jacobian.contracts.number_theory import (
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
)
from jacobian.contracts.results import ContractModel

PROTOCOL = "jacobian.number-theory.discrete-logarithm.sympy.v1"


class DiscreteLogarithmWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.discrete-logarithm.sympy.v1"]
    request: DiscreteLogarithmRequest


class DiscreteLogarithmWorkerResult(ContractModel):
    protocol: Literal["jacobian.number-theory.discrete-logarithm.sympy.v1"]
    result: DiscreteLogarithmResult
