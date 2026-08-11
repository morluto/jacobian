"""Closed protocol for isolated typed polynomial normalization."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jacobian.contracts.polynomial_expressions import PolynomialExpressionArtifact
from jacobian.contracts.polynomials import RationalPolynomial, SparseRationalPolynomial
from jacobian.contracts.results import ContractModel
from jacobian.provider_runtime import SYMPY_VERSION


class SympyPolynomialWorkerRequest(ContractModel):
    protocol: Literal["jacobian.sympy-polynomial-normalization/v1"]
    expression: PolynomialExpressionArtifact


class SympyPolynomialWorkerResponse(ContractModel):
    protocol: Literal["jacobian.sympy-polynomial-normalization/v1"]
    status: Literal["NORMALIZATION_PRODUCED"]
    backend_version: Literal["1.14.0"]
    normalized: SparseRationalPolynomial


def make_sympy_polynomial_worker_request(
    expression: PolynomialExpressionArtifact,
) -> SympyPolynomialWorkerRequest:
    return SympyPolynomialWorkerRequest(
        protocol="jacobian.sympy-polynomial-normalization/v1",
        expression=expression,
    )


def parse_sympy_polynomial_worker_request(
    value: object,
) -> SympyPolynomialWorkerRequest:
    return SympyPolynomialWorkerRequest.model_validate(value)


def parse_sympy_polynomial_worker_response(
    value: object,
    *,
    variables: tuple[str, ...],
) -> SparseRationalPolynomial:
    try:
        response = SympyPolynomialWorkerResponse.model_validate(value)
    except ValidationError as exc:
        raise ValueError("invalid SymPy polynomial worker response") from exc
    if response.backend_version != SYMPY_VERSION:
        raise ValueError("SymPy polynomial worker version does not match")
    return RationalPolynomial(
        variables=variables,
        polynomial=response.normalized,
    ).polynomial
