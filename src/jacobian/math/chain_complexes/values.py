"""Provider-independent exact values for finite based chain complexes."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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
# Aggregate printed characters across every differential cell. Coupling
# total coefficient size to the matrix work bounds rational elimination
# bit complexity at admission instead of only bounding input shape.
MAX_MATRIX_ENTRY_CHARS = 65536


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"chain_complex.{reason}", message)


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
            raise _validation_error(
                "degree_interval_invalid", "degree_max must be >= degree_min"
            )
        expected_len = self.degree_max - self.degree_min + 1
        if len(self.basis_sizes) != expected_len:
            raise _validation_error(
                "basis_degree_count_mismatch",
                "basis_sizes length must match degree interval",
            )
        if not 1 <= len(self.basis_sizes) <= 2 * MAX_CHAIN_DEGREE + 1:
            raise _validation_error(
                "basis_count_out_of_bounds", "basis_sizes length out of bounds"
            )
        for sz in self.basis_sizes:
            if not 0 <= sz <= MAX_BASIS_SIZE:
                raise _validation_error(
                    "basis_size_out_of_bounds",
                    f"basis size {sz} exceeds MAX_BASIS_SIZE {MAX_BASIS_SIZE}",
                )

        if len(self.differential_matrices) != max(0, len(self.basis_sizes) - 1):
            raise _validation_error(
                "differential_count_mismatch",
                "differential_matrices count must be len(basis_sizes)-1",
            )

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
                raise _validation_error(
                    "differential_row_count_mismatch",
                    f"differential_matrices[{idx}] row count {len(mat)} != basis size {rows_expected}",
                )
            for row in mat:
                if len(row) != cols_expected:
                    raise _validation_error(
                        "differential_column_count_mismatch",
                        f"differential_matrices[{idx}] column count {len(row)} != {cols_expected}",
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
            raise _validation_error(
                "matrix_cell_budget_exceeded",
                f"total matrix cells {total_cells} exceeds MAX_MATRIX_CELLS {MAX_MATRIX_CELLS}",
            )
        if total_entry_chars > MAX_MATRIX_ENTRY_CHARS:
            raise _validation_error(
                "matrix_entry_budget_exceeded",
                f"total differential characters {total_entry_chars} exceeds "
                f"MAX_MATRIX_ENTRY_CHARS {MAX_MATRIX_ENTRY_CHARS}; coefficient "
                "size is coupled to matrix work so exact elimination stays "
                "inside the admitted request boundary",
            )


def _require_prime_coupling(
    coefficient_field: CoefficientField, prime: int | None
) -> None:
    """GF_p carries a bounded prime modulus; QQ carries none."""
    if coefficient_field == CoefficientField.PRIME_FIELD:
        if prime is None:
            raise _validation_error("prime_required", "GF_p requires a prime modulus")
        if not 2 <= prime <= 1000003:
            raise _validation_error(
                "prime_out_of_bounds", "prime modulus exceeds the bounded prime limit"
            )
        if prime % 2 == 0:
            if prime != 2:
                raise _validation_error(
                    "prime_not_prime", f"prime {prime} is not prime"
                )
            return
        divisor = 3
        while divisor * divisor <= prime:
            if prime % divisor == 0:
                raise _validation_error(
                    "prime_not_prime", f"prime {prime} is not prime"
                )
            divisor += 2
    elif prime is not None:
        raise _validation_error(
            "prime_forbidden", "QQ coefficient field must not have a prime modulus"
        )


def _require_canonical_integer_spelling(entry: str, part: str) -> int:
    """Parse one canonical integer: no leading zeros, no negative zero."""
    if len(part) > 1 and (part[0] == "0" or part.startswith("-0")):
        raise _validation_error(
            "entry_not_canonical", f"entry '{entry}' is not canonically spelled"
        )
    return int(part)


def _require_canonical_fraction_entry(entry: str) -> None:
    """Fraction spellings are reduced with denominator >= 2 and digit bounds."""
    from math import gcd

    num_str, den_str = entry.split("/", 1)
    numerator = _require_canonical_integer_spelling(entry, num_str)
    denominator = _require_canonical_integer_spelling(entry, den_str)
    if den_str.lstrip("-").lstrip("0") == "" or denominator == 0:
        raise _validation_error(
            "fraction_zero_denominator", f"entry '{entry}' has zero denominator"
        )
    if numerator == 0:
        raise _validation_error(
            "zero_not_canonical",
            f"entry '{entry}' is not canonically spelled; spell zero as '0'",
        )
    if len(num_str.lstrip("-")) > 4096 or len(den_str.lstrip("-")) > 4096:
        raise _validation_error(
            "entry_digit_bound_exceeded", "differential entry exceeds digit bound"
        )
    # One rational has one reduced spelling.
    if denominator <= 1 or gcd(abs(numerator), denominator) != 1:
        raise _validation_error(
            "fraction_not_reduced",
            f"entry '{entry}' is not a reduced fraction; use its "
            "canonical reduced spelling",
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
        raise _validation_error(
            "entry_not_string", "differential entries must be strings"
        )
    if not re.fullmatch(r"-?\d+(/\d+)?", entry):
        raise _validation_error(
            "entry_grammar_invalid",
            f"entry '{entry}' does not match rational string grammar",
        )

    if "/" in entry:
        if coefficient_field == CoefficientField.PRIME_FIELD:
            raise _validation_error(
                "prime_field_entry_not_integer",
                f"prime-field entry '{entry}' must be an integer residue",
            )
        _require_canonical_fraction_entry(entry)
        return
    value = _require_canonical_integer_spelling(entry, entry)
    if len(entry.lstrip("-")) > 4096:
        raise _validation_error(
            "entry_digit_bound_exceeded", "differential entry exceeds digit bound"
        )
    if coefficient_field == CoefficientField.PRIME_FIELD and (
        value < 0 or (prime is not None and value >= prime)
    ):
        raise _validation_error(
            "prime_field_residue_invalid",
            f"prime-field entry '{entry}' must be a canonical "
            f"integer residue in [0, {prime})",
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
                raise _validation_error(
                    "homology_prime_required", "GF_p homology requires a prime"
                )
        else:
            if self.prime is not None:
                raise _validation_error(
                    "homology_prime_forbidden", "QQ homology must not have a prime"
                )
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
            raise _validation_error(
                "homology_degree_interval_mismatch",
                "degree interval must match the retained complex",
            )
        if (
            self.coefficient_field != self.complex.coefficient_field
            or self.prime != self.complex.prime
        ):
            raise _validation_error(
                "homology_context_mismatch",
                "coefficient field and prime must match the retained complex",
            )
        expected_degrees = list(
            range(self.complex.degree_min, self.complex.degree_max + 1)
        )
        if [group.degree for group in self.homology_groups] != expected_degrees:
            raise _validation_error(
                "homology_degree_coverage_invalid",
                "homology groups must cover every degree of the retained "
                "complex exactly once",
            )
        for group in self.homology_groups:
            if (
                group.betti_number != group.cycle_rank - group.boundary_rank
                or group.betti_number < 0
                or group.cycle_rank < 0
                or group.boundary_rank < 0
            ):
                raise _validation_error(
                    "homology_rank_identity_invalid",
                    f"homology group at degree {group.degree} violates "
                    "betti_number = cycle_rank - boundary_rank",
                )
        expected = _compute_homology_groups(self.complex)
        if self.homology_groups != tuple(expected):
            raise _validation_error(
                "homology_not_bound",
                "homology groups must be the exact homology of the retained complex",
            )
        return self


class MappingConeResult(StrictModel):
    """The mapping cone of a retained chain map.

    The derived complex retains its coefficient field, prime, and degree
    interval as a first-class chain-complex value so it composes into
    homology, tensor, map, and cone operations unchanged.
    """

    cone_basis_sizes: tuple[int, ...]
    cone_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    source_degree_min: int
    target_degree_min: int
    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    value: ChainComplexValue

    @model_validator(mode="after")
    def bind_cone_to_source(self) -> Self:
        """Replay the defining construction so only the exact mapping
        cone of the retained chain map validates."""
        from jacobian.math.chain_complexes._models import (
            _require_chain_map_components,
        )
        from jacobian.math.chain_complexes.operations import (
            _compute_mapping_cone,
        )

        # The retained map must satisfy the request contract before the
        # replay: _matrix_to_fractions zero-pads undersized matrices, so an
        # unpadded replay alone could bind a cone to a malformed non-map.
        _require_chain_map_components(
            self.source,
            self.target,
            self.map_matrices,
            label="mapping cone",
        )
        if (
            self.source_degree_min != self.source.degree_min
            or self.target_degree_min != self.target.degree_min
        ):
            raise _validation_error(
                "cone_degree_provenance_mismatch",
                "cone degree provenance must match the retained endpoints",
            )
        basis_sizes, differential_matrices = _compute_mapping_cone(
            self.source, self.target, self.map_matrices
        )
        if (
            self.cone_basis_sizes != basis_sizes
            or self.cone_differential_matrices != differential_matrices
        ):
            raise _validation_error(
                "cone_not_bound",
                "cone must be the exact mapping cone of the retained chain map",
            )
        if (
            self.value.basis_sizes != basis_sizes
            or self.value.differential_matrices != differential_matrices
        ):
            raise _validation_error(
                "cone_value_not_bound",
                "retained canonical value must equal the exact mapping cone",
            )
        expected_degree_max = self.source.degree_min + len(basis_sizes) - 1
        if (
            self.value.coefficient_field != self.source.coefficient_field
            or self.value.prime != self.source.prime
            or self.value.degree_min != self.source.degree_min
            or self.value.degree_max != expected_degree_max
        ):
            raise _validation_error(
                "cone_context_mismatch",
                "canonical value context must match the retained endpoints",
            )
        return self


class TensorProductResult(StrictModel):
    """The tensor product of two retained chain complexes.

    The derived complex retains its coefficient field, prime, and degree
    interval so it composes into downstream consumers as a first-class
    chain-complex value, and both factors are retained so validation can
    replay the defining construction.
    """

    tensor_basis_sizes: tuple[int, ...]
    tensor_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    coefficient_field: CoefficientField
    prime: int | None = Field(default=None, ge=2)
    degree_min: int
    degree_max: int
    left: ChainComplexValue
    right: ChainComplexValue
    value: ChainComplexValue

    @model_validator(mode="after")
    def require_consistent_context(self) -> Self:
        # One canonical tensor-product value carries the mathematical
        # result; it must be the exact tensor product of the retained
        # factors, so a revalidated result cannot hold a forged or merely
        # shape-compatible product.
        from jacobian.math.chain_complexes._models import (
            _require_admissible_tensor_work,
        )
        from jacobian.math.chain_complexes.operations import (
            _compute_tensor_product,
        )

        if self.left.prime != self.right.prime or (
            self.left.coefficient_field != self.right.coefficient_field
        ):
            raise _validation_error(
                "tensor_context_mismatch",
                "tensor product requires same coefficient field and prime",
            )
        # Bound the replay work before expanding any derived intermediate.
        _require_admissible_tensor_work(self.left, self.right)
        if self.coefficient_field != self.value.coefficient_field:
            raise _validation_error(
                "tensor_result_field_mismatch",
                "coefficient field must match the retained value",
            )
        if self.prime != self.value.prime:
            raise _validation_error(
                "tensor_result_prime_mismatch", "prime must match the retained value"
            )
        expected_sizes, expected_diffs = _compute_tensor_product(self.left, self.right)
        group_count = len(expected_sizes)
        derived_min = self.left.degree_min + self.right.degree_min
        derived_max = derived_min + group_count - 1
        if (self.degree_min, self.degree_max) != (derived_min, derived_max):
            raise _validation_error(
                "tensor_degree_interval_mismatch",
                "degree interval must equal the pairwise-sum interval of "
                "the retained factors",
            )
        expected_value = ChainComplexValue(
            coefficient_field=self.left.coefficient_field,
            prime=self.left.prime,
            degree_min=derived_min,
            degree_max=derived_max,
            basis_sizes=expected_sizes,
            differential_matrices=expected_diffs,
        )
        if self.value != expected_value:
            raise _validation_error(
                "tensor_value_not_bound",
                "tensor projections must equal the retained canonical "
                "tensor-product complex",
            )
        if self.tensor_basis_sizes != self.value.basis_sizes:
            raise _validation_error(
                "tensor_projection_not_bound",
                "tensor projections must equal the retained canonical "
                "tensor-product complex",
            )
        if self.tensor_differential_matrices != self.value.differential_matrices:
            raise _validation_error(
                "tensor_projection_not_bound",
                "tensor projections must equal the retained canonical "
                "tensor-product complex",
            )
        if (
            self.coefficient_field is CoefficientField.PRIME_FIELD
            and self.prime is None
        ):
            raise _validation_error(
                "tensor_prime_required", "GF_p tensor products must carry their prime"
            )
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
    "VerificationResult",
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
        detached or forged verdict cannot validate, and require the detail
        to be the exact authoritative explanation of that replay."""
        from jacobian.math.chain_complexes._models import (
            _require_chain_map_components,
        )
        from jacobian.math.chain_complexes.operations import (
            _chain_map_verdict,
            _differential_verdict,
        )

        if self.complex is not None:
            if self.source or self.target or self.map_matrices:
                raise _validation_error(
                    "verification_inputs_conflict",
                    "a differential verification result must not carry chain-map inputs",
                )
            holds, expected_detail = _differential_verdict(self.complex)
        elif (
            self.source is not None
            and self.target is not None
            and self.map_matrices is not None
        ):
            # Replay must apply the request model's complete component
            # and parent checks: otherwise endpoints with different
            # coefficient fields, primes, or degree intervals validate
            # by being interpreted under the source modulus alone.
            _require_chain_map_components(
                self.source,
                self.target,
                self.map_matrices,
                label="chain-map verification",
            )
            holds, expected_detail = _chain_map_verdict(
                self.source, self.target, self.map_matrices
            )
        else:
            raise _validation_error(
                "verification_inputs_missing",
                "a verification result must retain the complete checked "
                "input (the complex, or both endpoints with their map)",
            )
        if holds != self.is_valid:
            raise _validation_error(
                "verification_verdict_not_bound",
                "verification verdict must be the exact replay of the retained relation",
            )
        if self.detail != expected_detail:
            raise _validation_error(
                "verification_detail_not_bound",
                "verification detail must be the exact explanation of the "
                "replayed relation",
            )
        return self
