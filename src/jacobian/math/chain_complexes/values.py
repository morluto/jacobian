"""Provider-independent exact values for finite based chain complexes."""

from __future__ import annotations

from enum import StrEnum
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
# Serialization envelope for expanded tensor differentials.
MAX_TENSOR_SERIALIZED_CHARS = 4_000_000
# Aggregate printed characters across every differential cell. Coupling
# total coefficient size to the matrix work bounds rational elimination
# bit complexity at admission instead of only bounding input shape.
MAX_MATRIX_ENTRY_CHARS = 65536


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
        total_entry_chars = 0
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
                        prime=self.prime,
                    )
                    total_entry_chars += len(entry)
            total_cells += rows_expected * cols_expected
        if total_cells > MAX_MATRIX_CELLS:
            raise ValueError(
                f"total matrix cells {total_cells} exceeds MAX_MATRIX_CELLS {MAX_MATRIX_CELLS}"
            )
        if total_entry_chars > MAX_MATRIX_ENTRY_CHARS:
            raise ValueError(
                f"total differential characters {total_entry_chars} exceeds "
                f"MAX_MATRIX_ENTRY_CHARS {MAX_MATRIX_ENTRY_CHARS}; coefficient "
                "size is coupled to matrix work so exact elimination stays "
                "inside the admitted request boundary"
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


def _require_canonical_integer_spelling(entry: str, part: str) -> int:
    """Parse one canonical integer: no leading zeros, no negative zero."""
    if len(part) > 1 and (part[0] == "0" or part.startswith("-0")):
        raise ValueError(f"entry '{entry}' is not canonically spelled")
    return int(part)


def _require_canonical_fraction_entry(entry: str) -> None:
    """Fraction spellings are reduced with denominator >= 2 and digit bounds."""
    from math import gcd

    num_str, den_str = entry.split("/", 1)
    numerator = _require_canonical_integer_spelling(entry, num_str)
    denominator = _require_canonical_integer_spelling(entry, den_str)
    if den_str.lstrip("-").lstrip("0") == "" or denominator == 0:
        raise ValueError(f"entry '{entry}' has zero denominator")
    if numerator == 0:
        raise ValueError(
            f"entry '{entry}' is not canonically spelled; spell zero as '0'"
        )
    if len(num_str.lstrip("-")) > 4096 or len(den_str.lstrip("-")) > 4096:
        raise ValueError("differential entry exceeds digit bound")
    # One rational has one reduced spelling.
    if denominator <= 1 or gcd(abs(numerator), denominator) != 1:
        raise ValueError(
            f"entry '{entry}' is not a reduced fraction; use its "
            "canonical reduced spelling"
        )


def _require_rational_entry_grammar(
    coefficient_field: CoefficientField,
    entry: object,
    *,
    prime: int | None = None,
) -> None:
    """One canonical matrix entry: reduced rational grammar with digit bounds.

    Spellings must be canonical so one based complex has exactly one
    serialized identity: integers carry no leading zeros and no negative
    zero, fraction strings are fully reduced with denominator >= 2, and
    prime-field entries are residues in ``[0, p)``. A fractional string
    such as "1/2" would pass the rational regex but every downstream
    kernel parses prime-field entries with int(), so admitting it here
    would turn an accepted request into an execution failure.
    """
    import re

    if not isinstance(entry, str):
        raise ValueError("differential entries must be strings")
    if not re.fullmatch(r"-?\d+(/\d+)?", entry):
        raise ValueError(f"entry '{entry}' does not match rational string grammar")

    if "/" in entry:
        if coefficient_field == CoefficientField.PRIME_FIELD:
            raise ValueError(
                f"prime-field entry '{entry}' must be an integer residue"
            )
        _require_canonical_fraction_entry(entry)
        return
    value = _require_canonical_integer_spelling(entry, entry)
    if len(entry.lstrip("-")) > 4096:
        raise ValueError("differential entry exceeds digit bound")
    if coefficient_field == CoefficientField.PRIME_FIELD and (
        value < 0 or (prime is not None and value >= prime)
    ):
        raise ValueError(
            f"prime-field entry '{entry}' must be a canonical "
            f"integer residue in [0, {prime})"
        )


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
    coefficient_field: CoefficientField
    prime: int | None = Field(default=None, ge=2)
    degree_min: int
    degree_max: int
    value: ChainComplexValue

    @model_validator(mode="after")
    def require_consistent_context(self) -> Self:
        # One canonical tensor-product value carries the mathematical
        # result; every duplicated projection must agree with it so a
        # revalidated result cannot hold two contradictory products.
        if (
            self.tensor_basis_sizes != self.value.basis_sizes
            or self.tensor_differential_matrices
            != self.value.differential_matrices
        ):
            raise ValueError(
                "tensor projections must equal the retained canonical "
                "tensor-product complex"
            )
        if self.coefficient_field != self.value.coefficient_field:
            raise ValueError("coefficient field must match the retained value")
        if self.prime != self.value.prime:
            raise ValueError("prime must match the retained value")
        if self.degree_min != self.value.degree_min or self.degree_max != self.value.degree_max:
            raise ValueError("degree interval must match the retained value")
        if (
            self.coefficient_field is CoefficientField.PRIME_FIELD
            and self.prime is None
        ):
            raise ValueError("GF_p tensor products must carry their prime")
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
    """Result of verifying a chain complex property, bound to its input."""

    is_valid: bool
    detail: str
    complex: ChainComplexValue | None = None
    source: ChainComplexValue | None = None
    target: ChainComplexValue | None = None
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...] | None = None

    @model_validator(mode="after")
    def bind_verification_to_source(self) -> Self:
        """Replay the checked relation against the retained inputs so a
        detached or forged verdict cannot validate."""
        from jacobian.math.chain_complexes.operations import (
            _matrix_to_fractions,
            _parsed_differentials,
            _require_chain_map_relation,
            _require_square_zero,
        )
        if self.complex is not None:
            if self.source or self.target or self.map_matrices:
                raise ValueError(
                    "a differential verification result must not carry "
                    "chain-map inputs"
                )
            try:
                _require_square_zero(
                    _parsed_differentials(self.complex),
                    self.complex.prime,
                    label="verified",
                    group_columns=list(self.complex.basis_sizes),
                    degree_min=self.complex.degree_min,
                )
                holds = True
            except ValueError:
                holds = False
        elif (
            self.source is not None
            and self.target is not None
            and self.map_matrices is not None
        ):
            prime = self.source.prime
            try:
                map_mats = [
                    _matrix_to_fractions(
                        m, self.target.basis_sizes[i], self.source.basis_sizes[i], prime
                    )
                    for i, m in enumerate(self.map_matrices)
                ]
                source_diffs = [
                    _matrix_to_fractions(
                        m,
                        self.source.basis_sizes[i],
                        self.source.basis_sizes[i + 1],
                        prime,
                    )
                    for i, m in enumerate(self.source.differential_matrices)
                ]
                target_diffs = [
                    _matrix_to_fractions(
                        m,
                        self.target.basis_sizes[i],
                        self.target.basis_sizes[i + 1],
                        prime,
                    )
                    for i, m in enumerate(self.target.differential_matrices)
                ]
                # Producer semantics: endpoints must be genuine complexes
                # and the map must commute at every differential.
                for endpoint in (self.source, self.target):
                    _require_square_zero(
                        _parsed_differentials(endpoint),
                        endpoint.prime,
                        label="chain-map",
                        group_columns=list(endpoint.basis_sizes),
                        degree_min=endpoint.degree_min,
                    )
                _require_chain_map_relation(
                    source_diffs,
                    target_diffs,
                    map_mats,
                    prime,
                    source_group_columns=list(self.source.basis_sizes),
                )
                holds = True
            except (ValueError, IndexError):
                holds = False
        else:
            raise ValueError(
                "a verification result must retain the complete checked "
                "input (the complex, or both endpoints with their map)"
            )
        if holds != self.is_valid:
            raise ValueError(
                "verification verdict must be the exact replay of the "
                "retained relation"
            )
        return self


__all__.append("VerificationResult")
