"""Typed wire contracts for cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class SteenrodSquareRequest(StrictModel):
    """Compute Steenrod squares Sq^k(x) for a cocycle over GF(2).

    The input is a simplicial cocycle represented as a sparse cochain
    over GF(2): a list of (dimension, simplex_vertices, value) tuples.
    Only Sq^0 (identity) and the top operation Sq^{deg} = cup product are
    supported; intermediate squares 0<k<deg require cup-i structure and are
    rejected as unsupported.
    """

    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(
        min_length=0, max_length=1024
    )
    simplex_coefficients: tuple[int, ...] = Field(
        min_length=0, max_length=1024
    )
    square_degree: int = Field(ge=0, le=16)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.simplex_values) != len(self.simplex_coefficients):
            raise ValueError("simplex_values and simplex_coefficients must have the same length")
        # Validate simplex dimensions: each simplex must have exactly cochain_degree+1 distinct vertices.
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise ValueError(
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices"
                )
            if len(set(simplex)) != len(simplex):
                raise ValueError("simplex vertices must be distinct")
            if tuple(sorted(simplex)) != simplex:
                raise ValueError("simplex vertices must be sorted canonical")
        # Intermediate squares 0<k<deg are not implemented; reject to avoid false zero.
        if 0 < self.square_degree < self.cochain_degree:
            raise ValueError(
                "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
            )
        return self


class SteenrodSquareResult(StrictModel):
    """The result of a Steenrod square operation."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool
    square_degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.result_simplex_values) != len(self.result_simplex_coefficients):
            raise ValueError("result simplex arrays must have the same length")
        if self.is_zero and self.result_simplex_values:
            raise ValueError("a zero result must not carry simplex values")
        return self


class BocksteinRequest(StrictModel):
    """Compute the Bockstein homomorphism beta: H^n(Z/p) -> H^{n+1}(Z/p).

    The Bockstein for the short exact sequence 0 -> Z/p -> Z/p^2 -> Z/p -> 0
    requires the ambient simplicial complex to compute the coboundary of a
    lift. This operation currently only supports the trivial case where the
    input cocycle is zero modulo p (hence Bockstein is zero); non-zero
    cocycles are rejected as unsupported until the complex is provided.
    """

    prime: int = Field(ge=2, le=10_000)
    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(
        min_length=0, max_length=1024
    )
    simplex_coefficients: tuple[int, ...] = Field(
        min_length=0, max_length=1024
    )

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.simplex_values) != len(self.simplex_coefficients):
            raise ValueError("simplex_values and simplex_coefficients must have the same length")
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime integer")
        expected_dim = self.cochain_degree + 1
        for simplex in self.simplex_values:
            if len(simplex) != expected_dim:
                raise ValueError(
                    f"each simplex must have exactly cochain_degree+1={expected_dim} vertices"
                )
            if len(set(simplex)) != len(simplex):
                raise ValueError("simplex vertices must be distinct")
            if tuple(sorted(simplex)) != simplex:
                raise ValueError("simplex vertices must be sorted canonical")
        # Only zero cocycles are supported without the ambient complex.
        # Non-zero coefficients would require computing (1/p) d(lift), which needs
        # the simplicial coboundary and hence the full complex. Reject as unsupported
        # to avoid returning a false exact zero.
        if self.simplex_coefficients and any(c % self.prime != 0 for c in self.simplex_coefficients):
            raise ValueError(
                "non-zero Bockstein requires the ambient simplicial complex; unsupported in this bounded operation"
            )
        return self


class BocksteinResult(StrictModel):
    """The result of the Bockstein homomorphism."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool
    prime: int = Field(ge=2, le=10_000)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.result_simplex_values) != len(self.result_simplex_coefficients):
            raise ValueError("result simplex arrays must have the same length")
        return self


__all__ = [
    "BocksteinRequest",
    "BocksteinResult",
    "SteenrodSquareRequest",
    "SteenrodSquareResult",
]
