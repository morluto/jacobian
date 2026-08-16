"""Typed wire contracts for multivariate polynomial operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian.contracts.base import ContractModel

MAX_VARS = 16
MAX_TERMS = 256
MAX_DEGREE = 1000


class MultivariatePolynomialInput(ContractModel):
    variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    expression: str = Field(min_length=1, max_length=MAX_TERMS * 20)


class MultivariateGCDRequest(ContractModel):
    left: MultivariatePolynomialInput
    right: MultivariatePolynomialInput


class MultivariateGCDResult(ContractModel):
    gcd: str
    method: Literal["SYMPY_POLY_GCD"] = "SYMPY_POLY_GCD"


class MultivariateResultantRequest(ContractModel):
    left: MultivariatePolynomialInput
    right: MultivariatePolynomialInput
    eliminate_variable: str = Field(min_length=1, max_length=20)


class MultivariateResultantResult(ContractModel):
    resultant: str
    method: Literal["SYMPY_RESULTANT"] = "SYMPY_RESULTANT"
