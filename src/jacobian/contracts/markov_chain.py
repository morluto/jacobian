"""Typed wire contracts for Markov chain operations."""
from __future__ import annotations
from typing import Literal
from pydantic import Field
from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class TransitionMatrixRequest(ContractModel):
    """A finite stochastic transition matrix with rational entries."""
    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=32
    )


class StationaryDistributionResult(ContractModel):
    distribution: tuple[CanonicalRational, ...]
    method: Literal["SYMPY_EIGENVECTOR"] = "SYMPY_EIGENVECTOR"


class ErgodicDecisionResult(ContractModel):
    is_ergodic: bool
    is_irreducible: bool
    is_aperdiodic: bool
