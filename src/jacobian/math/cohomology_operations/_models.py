"""Typed wire contracts for cohomology operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


def _validate_simplex_entries(
    entries: tuple[tuple[int, ...], ...],
    label: str,
) -> None:
    for simplex in entries:
        if not simplex:
            raise ValueError(f"{label} must have at least one vertex")
        if len(set(simplex)) != len(simplex):
            raise ValueError(f"{label} vertices must be distinct")
        if tuple(sorted(simplex)) != simplex:
            raise ValueError(f"{label} vertices must be sorted canonical")


class SteenrodSquareRequest(StrictModel):
    """Compute Steenrod squares Sq^k(x) for a cocycle over GF(2).

    The input is a simplicial cocycle represented as a sparse cochain
    over GF(2): a list of (dimension, simplex_vertices, value) tuples.
    Only Sq^0 (identity) and the top operation Sq^{deg} = cup product are
    supported; intermediate squares 0<k<deg require cup-i structure and are
    rejected as unsupported. A top square is a cup product whose targets are
    (2*deg)-simplices of the ambient complex, so top squares additionally
    require ``ambient_simplices`` to establish which target faces exist.
    """

    cochain_degree: int = Field(ge=0, le=16)
    simplex_values: tuple[tuple[int, ...], ...] = Field(
        min_length=0, max_length=1024
    )
    simplex_coefficients: tuple[int, ...] = Field(
        min_length=0, max_length=1024
    )
    square_degree: int = Field(ge=0, le=16)
    ambient_simplices: tuple[tuple[int, ...], ...] = Field(
        default=(),
        max_length=4096,
        description=(
            "Simplices of the ambient complex, each a canonically sorted "
            "vertex tuple; required whenever square_degree equals "
            "cochain_degree and the cochain degree is positive."
        ),
    )

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
            _validate_simplex_entries((simplex,), "simplex")
        # Intermediate squares 0<k<deg are not implemented; reject to avoid false zero.
        if 0 < self.square_degree < self.cochain_degree:
            raise ValueError(
                "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
            )
        if self.square_degree == self.cochain_degree and self.cochain_degree >= 1:
            # A top square is the cup product x cup x carried by
            # (2*deg)-simplices of the ambient complex. Without the complex,
            # the vertex union cannot distinguish an absent face from a
            # present one, so no exact claim is possible.
            if not self.ambient_simplices:
                raise ValueError(
                    "the top Steenrod square requires the ambient simplicial "
                    "complex; supply ambient_simplices"
                )
            _validate_simplex_entries(self.ambient_simplices, "ambient simplex")
            known = set(self.ambient_simplices)
            for simplex in self.simplex_values:
                if simplex not in known:
                    raise ValueError(
                        "cochain support must lie inside the ambient complex"
                    )
        return self


class SteenrodSquareResult(SteenrodSquareRequest):
    """The result of a Steenrod square operation, bound to its source."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @model_validator(mode="after")
    def bind_to_source_cochain(self) -> Self:
        from jacobian.math.cohomology_operations._operations import (
            steenrod_square_fields,
        )

        expected = steenrod_square_fields(
            self.cochain_degree,
            self.simplex_values,
            self.simplex_coefficients,
            self.square_degree,
            self.ambient_simplices,
        )
        actual = (
            self.result_degree,
            self.result_simplex_values,
            self.result_simplex_coefficients,
            self.is_zero,
        )
        if actual != expected:
            raise ValueError(
                "result must equal the exact Steenrod-square replay of the "
                "retained source cochain"
            )
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
            _validate_simplex_entries((simplex,), "simplex")
        # Only zero cocycles are supported without the ambient complex.
        # Non-zero coefficients would require computing (1/p) d(lift), which needs
        # the simplicial coboundary and hence the full complex. Reject as unsupported
        # to avoid returning a false exact zero.
        if self.simplex_coefficients and any(c % self.prime != 0 for c in self.simplex_coefficients):
            raise ValueError(
                "non-zero Bockstein requires the ambient simplicial complex; unsupported in this bounded operation"
            )
        return self


class BocksteinResult(BocksteinRequest):
    """The result of the Bockstein homomorphism, bound to its source."""

    result_degree: int = Field(ge=0)
    result_simplex_values: tuple[tuple[int, ...], ...] = Field(default=())
    result_simplex_coefficients: tuple[int, ...] = Field(default=())
    is_zero: bool

    @model_validator(mode="after")
    def bind_to_source_cochain(self) -> Self:
        from jacobian.math.cohomology_operations._operations import (
            bockstein_fields,
        )

        expected = bockstein_fields(
            self.prime,
            self.cochain_degree,
            self.simplex_coefficients,
        )
        actual = (
            self.result_degree,
            self.result_simplex_values,
            self.result_simplex_coefficients,
            self.is_zero,
        )
        if actual != expected:
            raise ValueError(
                "result must equal the exact Bockstein replay of the "
                "retained source cochain"
            )
        return self


__all__ = [
    "BocksteinRequest",
    "BocksteinResult",
    "SteenrodSquareRequest",
    "SteenrodSquareResult",
]
