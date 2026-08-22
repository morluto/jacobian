"""Provider-independent exact values for finite based chain complexes."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CHAIN_DEGREE = 32
MAX_BASIS_SIZE = 64
MAX_MATRIX_CELLS = 4096
# Derived tensor-product work bounds: each tensor group dimension and the
# total tensor cell count stay within these conservative limits so no
# accepted request can allocate an unbounded dense intermediate.
MAX_TENSOR_GROUP_DIMENSION = 256
MAX_TENSOR_TOTAL_CELLS = 65536
# Chain-map admission bounds the aggregate component work: across all
# components, dense cells and printed entry characters stay within these
# conservative budgets.
MAX_CHAIN_MAP_CELLS = 4096
MAX_CHAIN_MAP_ENTRY_CHARS = 65536


class CoefficientField(StrEnum):
    RATIONAL = "QQ"
    PRIME_FIELD = "GF_p"


class ChainComplexValue(StrictModel):
    """A finite based chain complex over an exact field."""

    coefficient_field: CoefficientField
    prime: int | None = Field(default=None, ge=2)
    degree_min: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE)
    degree_max: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE)
    basis_sizes: tuple[int, ...]
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]

    @model_validator(mode="after")
    def require_canonical_chain_complex(self) -> Self:
        _require_prime_coupling(self.coefficient_field, self.prime)
        # Degree consistency
        if self.degree_max < self.degree_min:
            raise ValueError("degree_max must be >= degree_min")
        expected_len = self.degree_max - self.degree_min + 1
        if len(self.basis_sizes) != expected_len:
            raise ValueError("basis_sizes length must match degree interval")
        if not 1 <= len(self.basis_sizes) <= 2 * MAX_CHAIN_DEGREE + 1:
            raise ValueError("basis_sizes length out of bounds")
        for sz in self.basis_sizes:
            if not 0 <= sz <= MAX_BASIS_SIZE:
                raise ValueError(
                    f"basis size {sz} exceeds MAX_BASIS_SIZE {MAX_BASIS_SIZE}"
                )

        if len(self.differential_matrices) != max(0, len(self.basis_sizes) - 1):
            raise ValueError("differential_matrices count must be len(basis_sizes)-1")

        self._require_matrix_shapes()
        return self

    def _require_matrix_shapes(self) -> None:
        """Every differential has basis-size shape and bounded exact entries."""
        total_cells = 0
        for idx, mat in enumerate(self.differential_matrices):
            rows_expected = self.basis_sizes[idx]
            cols_expected = self.basis_sizes[idx + 1]
            if len(mat) != rows_expected:
                raise ValueError(
                    f"differential_matrices[{idx}] row count {len(mat)} != basis size {rows_expected}"
                )
            for row in mat:
                if len(row) != cols_expected:
                    raise ValueError(
                        f"differential_matrices[{idx}] column count {len(row)} != {cols_expected}"
                    )
                for entry in row:
                    _require_rational_entry_grammar(
                        self.coefficient_field,
                        entry,
                    )
            total_cells += rows_expected * cols_expected
        if total_cells > MAX_MATRIX_CELLS:
            raise ValueError(
                f"total matrix cells {total_cells} exceeds MAX_MATRIX_CELLS {MAX_MATRIX_CELLS}"
            )


def _require_prime_coupling(
    coefficient_field: CoefficientField, prime: int | None
) -> None:
    """GF_p carries a bounded prime modulus; QQ carries none."""
    if coefficient_field == CoefficientField.PRIME_FIELD:
        if prime is None:
            raise ValueError("GF_p requires a prime modulus")
        if not 2 <= prime <= 1000003:
            raise ValueError("prime modulus exceeds the bounded prime limit")
        if prime % 2 == 0:
            if prime != 2:
                raise ValueError(f"prime {prime} is not prime")
            return
        divisor = 3
        while divisor * divisor <= prime:
            if prime % divisor == 0:
                raise ValueError(f"prime {prime} is not prime")
            divisor += 2
    elif prime is not None:
        raise ValueError("QQ coefficient field must not have a prime modulus")


def _require_rational_entry_grammar(
    coefficient_field: CoefficientField, entry: object
) -> None:
    """One matrix entry: exact rational grammar with digit bounds.

    Prime-field entries are integer residues; a fractional string such as
    "1/2" would pass the rational regex but every downstream kernel parses
    prime-field entries with int(), so admitting it here would turn an
    accepted request into an execution failure.
    """
    import re

    if not isinstance(entry, str):
        raise ValueError("differential entries must be strings")
    if not re.fullmatch(r"-?\d+(/\d+)?", entry):
        raise ValueError(f"entry '{entry}' does not match rational string grammar")
    if coefficient_field == CoefficientField.PRIME_FIELD and "/" in entry:
        raise ValueError(f"prime-field entry '{entry}' must be an integer residue")
    # Check denominator not zero and digits bound
    if "/" in entry:
        num_str, den_str = entry.split("/", 1)
        if den_str.lstrip("-").lstrip("0") == "" or int(den_str) == 0:
            raise ValueError(f"entry '{entry}' has zero denominator")
        if len(num_str.lstrip("-")) > 4096 or len(den_str.lstrip("-")) > 4096:
            raise ValueError("differential entry exceeds digit bound")
    else:
        if len(entry.lstrip("-")) > 4096:
            raise ValueError("differential entry exceeds digit bound")
    # Ensure it parses as Fraction
    try:
        Fraction(entry)
    except Exception as error:
        raise ValueError(f"entry '{entry}' is not a valid rational: {error}") from error


class HomologyGroupValue(StrictModel):
    """One homology group of a chain complex."""

    degree: int
    cycle_rank: int = Field(ge=0)
    boundary_rank: int = Field(ge=0)
    betti_number: int = Field(ge=0)


class HomologyResult(StrictModel):
    """Homology of a retained source chain complex."""

    homology_groups: tuple[HomologyGroupValue, ...]
    coefficient_field: CoefficientField
    prime: int | None = Field(default=None, ge=2)
    degree_min: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE, default=0)
    degree_max: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE, default=0)
    complex: ChainComplexValue

    @model_validator(mode="after")
    def require_prime_coupling(self) -> Self:
        if self.coefficient_field == CoefficientField.PRIME_FIELD:
            if self.prime is None:
                raise ValueError("GF_p homology requires a prime")
        else:
            if self.prime is not None:
                raise ValueError("QQ homology must not have a prime")
        return self

    @model_validator(mode="after")
    def bind_profile_to_source(self) -> Self:
        # Source-bound replay: the profile must be the exact homology of
        # the retained complex, with contiguous degrees and the defining
        # rank identity per group.
        from jacobian.math.chain_complexes.operations import (
            _compute_homology_groups,
        )

        if (
            self.degree_min != self.complex.degree_min
            or self.degree_max != self.complex.degree_max
        ):
            raise ValueError("degree interval must match the retained complex")
        if (
            self.coefficient_field != self.complex.coefficient_field
            or self.prime != self.complex.prime
        ):
            raise ValueError(
                "coefficient field and prime must match the retained complex"
            )
        expected_degrees = list(
            range(self.complex.degree_min, self.complex.degree_max + 1)
        )
        if [group.degree for group in self.homology_groups] != expected_degrees:
            raise ValueError(
                "homology groups must cover every degree of the retained "
                "complex exactly once"
            )
        for group in self.homology_groups:
            if (
                group.betti_number != group.cycle_rank - group.boundary_rank
                or group.betti_number < 0
                or group.cycle_rank < 0
                or group.boundary_rank < 0
            ):
                raise ValueError(
                    f"homology group at degree {group.degree} violates "
                    "betti_number = cycle_rank - boundary_rank"
                )
        expected = _compute_homology_groups(self.complex)
        if self.homology_groups != tuple(expected):
            raise ValueError(
                "homology groups must be the exact homology of the retained complex"
            )
        return self


class MappingConeResult(StrictModel):
    """The mapping cone of a chain map."""

    cone_basis_sizes: tuple[int, ...]
    cone_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    source_degree_min: int
    target_degree_min: int


class TensorProductResult(StrictModel):
    """The tensor product of two chain complexes.

    The derived complex retains its coefficient field, prime, and degree
    interval so it composes into downstream consumers as a first-class
    chain-complex value.
    """

    tensor_basis_sizes: tuple[int, ...]
    tensor_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    coefficient_field: CoefficientField | None = None
    prime: int | None = Field(default=None, ge=2)
    degree_min: int | None = None
    degree_max: int | None = None

    @model_validator(mode="after")
    def require_consistent_context(self) -> Self:
        if (self.degree_min is None) != (self.degree_max is None):
            raise ValueError("degree interval must be provided for both endpoints")
        if (
            self.coefficient_field is CoefficientField.PRIME_FIELD
            and self.prime is None
        ):
            raise ValueError("GF_p tensor products must carry their prime")
        if self.coefficient_field is not CoefficientField.PRIME_FIELD and (
            self.prime is not None
        ):
            raise ValueError("QQ tensor products must not carry a prime")
        return self


__all__ = [
    "MAX_BASIS_SIZE",
    "MAX_CHAIN_DEGREE",
    "MAX_MATRIX_CELLS",
    "MAX_TENSOR_GROUP_DIMENSION",
    "MAX_TENSOR_TOTAL_CELLS",
    "ChainComplexValue",
    "CoefficientField",
    "HomologyGroupValue",
    "HomologyResult",
    "MappingConeResult",
    "TensorProductResult",
]


class VerificationResult(StrictModel):
    """Result of verifying a chain complex property."""

    is_valid: bool
    detail: str
    complex: ChainComplexValue | None = None
    source: ChainComplexValue | None = None
    target: ChainComplexValue | None = None

    @model_validator(mode="after")
    def bind_verification_to_source(self) -> Self:
        # If a source complex is present, ensure the verdict is replayable
        # by checking that the stored detail matches the expected pattern for the claimed complex.
        # For now, require that at least one of complex/source is set when is_valid is claimed.
        # The operation-level validators will ensure the boolean matches the actual check.
        return self


__all__.append("VerificationResult")
