"""Typed wire contracts for coalgebra and Hopf algebra operations."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

#: Decimal digit budget for an admitted field characteristic. The digit
#: length, not a magnitude ceiling, is what bounds the admitted work:
#: primality testing and every modular multiply-add operate on operands
#: below the prime, so their cost grows with its digit length while the
#: mathematical domain stays open (any GF(p) whose characteristic carries
#: at most this many digits is admissible when the derived work budgets
#: below also hold).
MAX_PRIME_DIGITS = 64

#: Group-like enumeration scans every element of GF(p)^dimension. Its
#: admission bound is derived work, not a bare candidate count: every
#: candidate pays dimension counit-filter summands plus one unit of
#: candidate-loop overhead, and -- because a valid counit is nonzero --
#: exactly prime**(dimension-1) candidates survive that filter and each
#: pays dimension**3 Delta-construction summands plus 2*dimension**2
#: tensor-square and comparison operations:
#:
#:   group_like_scan_work = prime**dimension * (dimension + 1)
#:       + prime**(dimension-1) * (dimension**3 + 2*dimension**2)
#:
#: Measured throughput exceeds 50M summands/s in the dense regime and 5M
#: units/s in the loop-overhead dominated one-dimensional regime, keeping an
#: admitted call well under two seconds.
GROUP_LIKE_SCAN_WORK_BUDGET = 2_000_000

#: Maximum number of structure-constant entries in an admitted comultiplication
#: tensor. Admission is derived from explicit work and size bounds instead of a
#: coarse dimension ceiling:
#:   - the input tensor and each per-request projection are bounded by exactly
#:     dimension**3 <= MAX_TENSOR_ENTRIES entries;
#:   - coassociativity validation performs at most dimension**5 modular
#:     multiply-adds (16**5 ~ 10**6 at this envelope), each on operands below
#:     the MAX_PRIME_DIGITS-bounded characteristic;
#:   - group-like scans remain separately bounded by GROUP_LIKE_SCAN_WORK_BUDGET
#:     derived work units covering candidates x per-candidate reconstruction
#:     for the exhaustive producer scan.
#: For example the 9-dimensional GF(2) direct-sum coalgebra needs 729 entries
#: and roughly 2*10**5 scan-work units, well inside both budgets.
MAX_TENSOR_ENTRIES = 4096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"coalgebra.{reason}", message)


def group_like_scan_work(prime: int, dimension: int) -> int:
    """Worst-case Python-level work units for one exhaustive group-like scan.

    Every prime**dimension candidate pays dimension counit summands plus one
    loop-overhead unit; exactly prime**(dimension-1) candidates pass the
    nonzero counit filter and each pays dimension**3 Delta-construction
    summands plus 2*dimension**2 tensor-square and comparison operations.
    """
    n = dimension
    candidate_units: int = prime**n * (n + 1)
    survivor_units: int = prime ** (n - 1) * (n**3 + 2 * n * n)
    return candidate_units + survivor_units


def _require_admitted_prime_digits(prime: int) -> None:
    """Reject characteristics beyond the documented digit budget before any
    primality test or modular arithmetic runs."""
    if prime >= 10**MAX_PRIME_DIGITS:
        raise _validation_error(
            "prime_digits_exceeded",
            f"field prime exceeds the {MAX_PRIME_DIGITS}-digit admission bound",
        )


class Coalgebra(StrictModel):
    """A finite-dimensional coalgebra over a prime field GF(p).

    The comultiplication is specified by structure constants:
    Delta(c_i) = sum_{j,k} d_{i}^{jk} * c_j ⊗ c_k
    The counit is epsilon(c_i) = e_i.

    Admission is derived from named work budgets rather than magnitude
    ceilings: the characteristic is bounded by decimal digit length
    (MAX_PRIME_DIGITS), the tensor carries at most MAX_TENSOR_ENTRIES
    structure constants, and group-like enumeration additionally requires
    group_like_scan_work(prime, dimension) <= GROUP_LIKE_SCAN_WORK_BUDGET.
    """

    prime: int = Field(
        ge=2,
        description=(
            "characteristic p of the prime field GF(p), admitted up to "
            "MAX_PRIME_DIGITS decimal digits. Every comultiplication and "
            "counit entry must already be a canonical residue in 0..p-1; "
            "noncanonical representatives are rejected"
        ),
    )
    dimension: int = Field(
        ge=1,
        description=(
            "dimension n of the coalgebra. Fixes the exact shapes: the "
            "comultiplication must be n x n x n and the counit length n; "
            "admission additionally bounds the tensor at n^3 structure "
            "constants"
        ),
    )
    comultiplication: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1,
        description=(
            "structure-constant tensor of exact shape n x n x n where n is "
            "the declared dimension: entry [i][j][k] is the coefficient of "
            "c_j ⊗ c_k in Delta(c_i), a canonical residue in 0..p-1. The "
            "slices must satisfy coassociativity, (Delta x id) o Delta = "
            "(id x Delta) o Delta modulo p"
        ),
    )
    counit: tuple[int, ...] = Field(
        min_length=1,
        description=(
            "counit vector of exact length n where n is the declared "
            "dimension: entry i is epsilon(c_i), a canonical residue in "
            "0..p-1. Must satisfy both counit identities, (epsilon x id) o "
            "Delta = id and (id x epsilon) o Delta = id, modulo p"
        ),
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        # Fail fast on the derived size budget before any O(n^3) scan: it
        # bounds every later validation pass.
        if self.dimension**3 > MAX_TENSOR_ENTRIES:
            raise _validation_error(
                "tensor_budget_exceeded",
                f"coalgebra admission allows at most {MAX_TENSOR_ENTRIES} "
                f"structure constants; dimension {self.dimension} would "
                f"carry {self.dimension**3}",
            )
        if len(self.comultiplication) != self.dimension:
            raise _validation_error(
                "comultiplication_shape", "comultiplication must have dimension entries"
            )
        for row in self.comultiplication:
            if len(row) != self.dimension:
                raise _validation_error(
                    "comultiplication_shape",
                    "comultiplication entry must be dimension x dimension",
                )
            for v in row:
                if len(v) != self.dimension:
                    raise _validation_error(
                        "comultiplication_shape", "comultiplication tensor must be 3D"
                    )
        if len(self.counit) != self.dimension:
            raise _validation_error(
                "counit_shape", "counit must have dimension entries"
            )
        _require_admitted_prime_digits(self.prime)
        from sympy import isprime

        if not isprime(self.prime):
            raise _validation_error("prime_not_prime", "prime must be a prime integer")
        self._require_canonical_residues()
        self._require_coalgebra_axioms()
        return self

    def _require_canonical_residues(self) -> None:
        """Reject noncanonical entries: each must already lie in 0..p-1 to
        avoid implicit field coercion and nonunique serialized identities."""
        for row in self.comultiplication:
            for v in row:
                for value in v:
                    if not 0 <= value < self.prime:
                        raise _validation_error(
                            "noncanonical_structure_constants",
                            "structure constants must be canonical residues "
                            f"in 0..{self.prime - 1}",
                        )
        for value in self.counit:
            if not 0 <= value < self.prime:
                raise _validation_error(
                    "noncanonical_counit",
                    f"counit entries must be canonical residues in 0..{self.prime - 1}",
                )

    def _require_coalgebra_axioms(self) -> None:
        """Validate coassociativity and both counit identities modulo p.

        Group-like conclusions presuppose a coalgebra: (Delta tensor id) o
        Delta = (id tensor Delta) o Delta, (epsilon tensor id) o Delta = id,
        and (id tensor epsilon) o Delta = id. Arbitrary linear maps do not
        satisfy these and must not be admitted.
        """
        p = self.prime
        d = self.comultiplication
        n = self.dimension
        e = self.counit

        # Coassociativity per basis element i:
        # (Delta tensor id) Delta(c_i) = (id tensor Delta) Delta(c_i)
        # Coefficient of c_j tensor c_k tensor c_ell:
        #   sum_t d[i][t][ell] * d[t][j][k] == sum_t d[i][j][t] * d[t][k][ell]  (mod p)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for ell in range(n):
                        left = sum(d[i][t][ell] * d[t][j][k] for t in range(n)) % p
                        right = sum(d[i][j][t] * d[t][k][ell] for t in range(n)) % p
                        if left != right:
                            raise _validation_error(
                                "not_coassociative",
                                "comultiplication must be coassociative",
                            )

        # Counit identities: both (epsilon tensor id)Delta = id and
        # (id tensor epsilon)Delta = id must hold.
        #   sum_t e[t] * d[i][t][j] == delta_{i,j}   ((epsilon tensor id))
        #   sum_t e[t] * d[i][j][t] == delta_{i,j}   ((id tensor epsilon))
        for i in range(n):
            for j in range(n):
                left_counit = sum(e[t] * d[i][t][j] for t in range(n)) % p
                right_counit = sum(e[t] * d[i][j][t] for t in range(n)) % p
                expected = 1 if i == j else 0
                if left_counit != expected or right_counit != expected:
                    raise _validation_error(
                        "counit_axioms",
                        "counit identities (epsilon x id)Delta = id and "
                        "(id x epsilon)Delta = id must hold modulo p",
                    )


class ComultiplicationRequest(StrictModel):
    """Compute the comultiplication Delta applied to a basis element."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.element_index >= self.coalgebra.dimension:
            raise _validation_error(
                "element_index_out_of_range", "element_index must be in 0..dimension-1"
            )
        return self


class ComultiplicationResult(StrictModel):
    """The comultiplication Delta(c_i) as a canonical prime-field matrix."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)
    matrix: PrimeFieldMatrix
    dimension: int = Field(ge=1)

    @model_validator(mode="before")
    @classmethod
    def require_admitted_nested_modulus(cls, data: Any) -> Any:
        """Inspect the raw nested modulus before PrimeFieldMatrix is built.

        A detached result can carry any matrix dict; constructing the shared
        PrimeFieldMatrix type would run its primality test on an unbounded
        modulus before bind_comultiplication_to_source can compare fields.
        The digit budget and the equality with the already-validated
        coalgebra prime are therefore enforced on the raw nested input.
        """
        if not isinstance(data, dict):
            return canonicalize_json_containers(data)
        matrix_input = data.get("matrix")
        if not isinstance(matrix_input, dict):
            return canonicalize_json_containers(data)
        nested_prime = matrix_input.get("prime")
        if type(nested_prime) is not int:
            return canonicalize_json_containers(data)
        _require_admitted_prime_digits(nested_prime)
        coalgebra_input = data.get("coalgebra")
        coalgebra_prime = (
            coalgebra_input.get("prime")
            if isinstance(coalgebra_input, dict)
            else getattr(coalgebra_input, "prime", None)
        )
        if type(coalgebra_prime) is int and nested_prime != coalgebra_prime:
            raise _validation_error(
                "matrix_prime_mismatch",
                "matrix prime must match the retained coalgebra's field",
            )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_structural_source_binding(self) -> Self:
        """Check the retained carrier and matrix shape without recomputing Delta."""
        ca = self.coalgebra
        n = ca.dimension
        p = ca.prime
        if self.element_index >= n:
            raise _validation_error(
                "element_index_out_of_range", "element_index must be in 0..dimension-1"
            )
        if self.dimension != n:
            raise _validation_error(
                "dimension_mismatch", "dimension must match the retained coalgebra"
            )
        if self.dimension != self.matrix.columns or len(self.matrix.entries) != n:
            raise _validation_error(
                "dimension_mismatch", "dimension must match the retained coalgebra"
            )
        # The canonical value carries its own field: a GF(p) coalgebra cannot
        # describe a matrix over a different modulus.
        if self.matrix.prime != p:
            raise _validation_error(
                "matrix_prime_mismatch",
                "matrix prime must match the retained coalgebra's field",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ComultiplicationRequest,
        matrix: PrimeFieldMatrix,
    ) -> Self:
        """Construct a result from the already-admitted kernel output."""
        return cls.model_construct(
            coalgebra=request.coalgebra,
            element_index=request.element_index,
            matrix=matrix,
            dimension=request.coalgebra.dimension,
        )


class CounitRequest(StrictModel):
    """Compute the counit epsilon applied to a basis element."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.element_index >= self.coalgebra.dimension:
            raise _validation_error(
                "element_index_out_of_range", "element_index must be in 0..dimension-1"
            )
        return self


class CounitResult(StrictModel):
    """The counit value epsilon(c_i)."""

    coalgebra: Coalgebra
    element_index: int = Field(ge=0)
    value: int

    @model_validator(mode="after")
    def require_structural_source_binding(self) -> Self:
        """Check the retained carrier without recomputing the counit."""
        ca = self.coalgebra
        if self.element_index >= ca.dimension:
            raise _validation_error(
                "element_index_out_of_range", "element_index must be in 0..dimension-1"
            )
        if not 0 <= self.value < ca.prime:
            raise _validation_error(
                "noncanonical_counit_value",
                "value must be a canonical residue in the coalgebra field",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: CounitRequest, value: int) -> Self:
        """Construct a result from the already-admitted kernel output."""
        return cls.model_construct(
            coalgebra=request.coalgebra,
            element_index=request.element_index,
            value=value,
        )


class GroupLikeElementsRequest(StrictModel):
    """Find all group-like elements in a coalgebra.

    The operation enumerates every element of GF(p)^dimension and
    reconstructs each candidate that survives the counit filter, so
    requests are admitted only when group_like_scan_work(prime,
    dimension) is within the documented scan budget.
    """

    coalgebra: Coalgebra

    @model_validator(mode="after")
    def require_enumerable(self) -> Self:
        work = group_like_scan_work(self.coalgebra.prime, self.coalgebra.dimension)
        if work > GROUP_LIKE_SCAN_WORK_BUDGET:
            raise _validation_error(
                "scan_work_budget_exceeded",
                "group-like enumeration scan work exceeds the documented "
                f"budget: {work} units for prime {self.coalgebra.prime}, "
                f"dimension {self.coalgebra.dimension} exceeds "
                f"{GROUP_LIKE_SCAN_WORK_BUDGET}",
            )
        return self


class GroupLikeElement(StrictModel):
    """One group-like element with its coefficients."""

    coefficients: tuple[int, ...]


class GroupLikeElementsResult(StrictModel):
    """All group-like elements of a coalgebra."""

    coalgebra: Coalgebra
    elements: tuple[GroupLikeElement, ...]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_structural_elements(self) -> Self:
        """Check bounded canonical result structure without replaying a search."""
        if self.count != len(self.elements):
            raise _validation_error("count_mismatch", "count must match element count")
        work = group_like_scan_work(self.coalgebra.prime, self.coalgebra.dimension)
        if work > GROUP_LIKE_SCAN_WORK_BUDGET:
            raise _validation_error(
                "scan_work_budget_exceeded",
                "group-like enumeration scan work exceeds the documented "
                f"budget: {work} units for prime {self.coalgebra.prime}, "
                f"dimension {self.coalgebra.dimension} exceeds "
                f"{GROUP_LIKE_SCAN_WORK_BUDGET}",
            )
        n = self.coalgebra.dimension
        seen = set()
        for element in self.elements:
            if len(element.coefficients) != n:
                raise _validation_error(
                    "element_dimension_mismatch",
                    "element coefficients must match the coalgebra dimension",
                )
            key = tuple(element.coefficients)
            if any(not 0 <= coefficient < self.coalgebra.prime for coefficient in key):
                raise _validation_error(
                    "noncanonical_group_like_coefficients",
                    "group-like coefficients must be canonical field residues",
                )
            if key in seen:
                raise _validation_error(
                    "duplicate_group_like", "group-like elements must be distinct"
                )
            seen.add(key)
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: GroupLikeElementsRequest,
        elements: tuple[GroupLikeElement, ...],
    ) -> Self:
        """Construct an exhaustive result from the already-admitted kernel output."""
        return cls.model_construct(
            coalgebra=request.coalgebra,
            elements=elements,
            count=len(elements),
        )


__all__ = [
    "GROUP_LIKE_SCAN_WORK_BUDGET",
    "MAX_PRIME_DIGITS",
    "Coalgebra",
    "ComultiplicationRequest",
    "ComultiplicationResult",
    "CounitRequest",
    "CounitResult",
    "GroupLikeElement",
    "GroupLikeElementsRequest",
    "GroupLikeElementsResult",
    "group_like_scan_work",
]
