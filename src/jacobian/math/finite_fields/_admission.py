"""Admitted mathematical checks for caller-authored finite-field values."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import (
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
)


def require_field(presentation: FiniteFieldPresentation) -> Any:
    """Recognize the bounded field presentation and retain its FLINT modulus.

    Structural parsing already bounds p^degree by 65536 and degree by 16.
    Neither result decoding nor nested value construction invokes this check.
    """
    from flint import fmpz, fmpz_mod_poly_ctx

    if not fmpz(presentation.characteristic).is_prime():
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.characteristic_prime_integer",
            message="characteristic must be prime",
        )
    modulus = fmpz_mod_poly_ctx(presentation.characteristic)(
        list(presentation.modulus_coefficients)
    )
    if not modulus.is_irreducible():
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.modulus_irreducible_over_prime_field",
            message="modulus must be irreducible over the prime field",
        )
    return modulus


def require_independent_basis(subspace: FiniteDimensionalSubspace) -> None:
    """Check the declared prime-field basis after the field is admitted.

    The structural carrier bounds the flattened matrix by 256^2 cells and
    the number of basis vectors by 256; elimination is bounded by 256^3.
    """
    from flint import nmod_mat

    flattened = [
        [
            coordinate
            for row in matrix.entries
            for element in row
            for coordinate in element.coordinates
        ]
        for matrix in subspace.basis
    ]
    if nmod_mat(flattened, subspace.presentation.characteristic).rank() != len(
        subspace.basis
    ):
        raise OperationDomainValidationError(
            location=("subspace", "basis"),
            code="finite_field.subspace_basis_matrices_linearly_independent",
            message="subspace basis matrices must be linearly independent over the prime field",
        )
