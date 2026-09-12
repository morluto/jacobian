"""Prime-field linear-code canonicalization under coordinate actions."""

from itertools import permutations
from math import factorial
from typing import Literal, Self

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError
from sympy import isprime

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.operations import _backend_group, _full_permutation_form
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    _rref_admitted,
)

MAX_CODE_CANONICALIZATION_ACTION_ORDER = 100_000
MAX_CODE_CANONICALIZATION_RREF_WORK = 20_000_000


class LinearCodeCanonicalizationRequest(StrictModel):
    encoder: PrimeFieldLinearEncoder
    action: Literal["FULL_SYMMETRIC", "SUPPLIED_GROUP"] = "FULL_SYMMETRIC"
    permutation_group: PermutationGroup | None = None

    @model_validator(mode="after")
    def bind_action(self) -> Self:
        if (self.action == "SUPPLIED_GROUP") != (self.permutation_group is not None):
            raise ValueError("SUPPLIED_GROUP requires exactly one permutation_group")
        if self.permutation_group is not None and self.permutation_group.degree != len(
            self.encoder.coordinate_axis
        ):
            raise ValueError("permutation-group degree must match the coordinate axis")
        return self


class LinearCodeCanonicalizationResult(StrictModel):
    source: LinearCodeCanonicalizationRequest
    canonical_encoder: PrimeFieldLinearEncoder
    transporter: tuple[StrictInt, ...]
    transported_axis: tuple[str, ...]
    orbit_size: ExactInteger
    stabilizer_size: ExactInteger

    @model_validator(mode="after")
    def require_structural_canonicalization(self) -> Self:
        """Check result wiring without replaying finite-field elimination."""

        source_encoder = self.source.encoder
        width = len(source_encoder.coordinate_axis)
        if len(self.transporter) != width or sorted(self.transporter) != list(
            range(width)
        ):
            raise PydanticCustomError(
                "code_linear.canonicalization_transporter_not_permutation",
                "transporter must be a permutation of the source coordinates",
            )
        expected_axis = tuple(
            source_encoder.coordinate_axis[index] for index in self.transporter
        )
        if self.transported_axis != expected_axis:
            raise PydanticCustomError(
                "code_linear.canonicalization_transported_axis_mismatch",
                "transported axis must follow the transporter",
            )
        if self.canonical_encoder.field_order != source_encoder.field_order:
            raise PydanticCustomError(
                "code_linear.canonicalization_field_mismatch",
                "canonical encoder must retain the source field",
            )
        if self.canonical_encoder.coordinate_axis != self.transported_axis:
            raise PydanticCustomError(
                "code_linear.canonicalization_axis_mismatch",
                "canonical encoder must retain the transported coordinate axis",
            )
        if len(self.canonical_encoder.generator_matrix) != len(
            source_encoder.generator_matrix
        ):
            raise PydanticCustomError(
                "code_linear.canonicalization_dimension_mismatch",
                "canonical encoder must retain the source dimension",
            )
        if self.canonical_encoder.message_axis != tuple(
            f"m{index}" for index in range(len(self.canonical_encoder.generator_matrix))
        ):
            raise PydanticCustomError(
                "code_linear.canonicalization_message_axis",
                "canonical encoder message labels must be m0, m1, and so on",
            )
        if self.orbit_size < 1 or self.stabilizer_size < 1:
            raise PydanticCustomError(
                "code_linear.canonicalization_orbit_stabilizer_positive",
                "orbit and stabilizer sizes must be positive",
            )
        if (
            self.orbit_size * self.stabilizer_size
            > MAX_CODE_CANONICALIZATION_ACTION_ORDER
        ):
            raise PydanticCustomError(
                "code_linear.canonicalization_orbit_stabilizer_bound",
                "orbit-stabilizer product exceeds the admitted action bound",
            )
        return self


def _rref(
    matrix: tuple[tuple[int, ...], ...], prime: int, width: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Reduce through the canonical prime-field matrix carrier.

    Keeping this operation on the shared carrier is important for composition:
    the same FLINT/SymPy exact backend and empty-axis convention used by the
    other code-linear operations define every canonical representative here.
    """

    return _rref_admitted(PrimeFieldMatrix(prime=prime, entries=matrix, columns=width))


def canonicalize_linear_code(
    request: LinearCodeCanonicalizationRequest,
) -> LinearCodeCanonicalizationResult:
    encoder = request.encoder
    if not isprime(encoder.field_order):
        raise OperationDomainValidationError(
            location=("encoder", "field_order"),
            code="code.canonicalization.prime_field",
            message="linear-code canonicalization requires a prime field order",
        )
    _, pivots = _rref(
        encoder.generator_matrix,
        encoder.field_order,
        len(encoder.coordinate_axis),
    )
    if len(pivots) != len(encoder.generator_matrix):
        raise OperationDomainValidationError(
            location=("encoder", "generator_matrix"),
            code="code.canonicalization.full_row_rank",
            message="linear-code canonicalization requires a full-row-rank encoder",
        )
    width = len(encoder.coordinate_axis)
    if request.action == "FULL_SYMMETRIC":
        order = factorial(width)
        if order > MAX_CODE_CANONICALIZATION_ACTION_ORDER:
            raise OperationResourceAdmissionError(
                location=("encoder", "coordinate_axis"),
                code="code.canonicalization.factorial_bound",
                message="full symmetric coordinate action exceeds its factorial bound",
            )
        backend = None
    else:
        assert request.permutation_group is not None
        backend = _backend_group(request.permutation_group)
        order = int(backend.order())
        if order > MAX_CODE_CANONICALIZATION_ACTION_ORDER:
            raise OperationResourceAdmissionError(
                location=("permutation_group",),
                code="code.canonicalization.group_order_bound",
                message="generated coordinate group exceeds its enumeration bound",
            )
    work = order * max(1, len(encoder.message_axis)) * max(1, width * width)
    if work > MAX_CODE_CANONICALIZATION_RREF_WORK:
        raise OperationResourceAdmissionError(
            location=("encoder",),
            code="code.canonicalization.rref_work_bound",
            message="coordinate-action RREF traversal exceeds its admitted work bound",
        )
    # Do not materialize an action orbit until the complete traversal has been
    # admitted. In particular, S_10 has 3.6M elements but is rejected by the
    # RREF-work bound for every nontrivial encoder before tuple construction.
    if request.action == "FULL_SYMMETRIC":
        elements = tuple(permutations(range(width)))
    else:
        assert backend is not None
        elements = tuple(
            sorted(
                _full_permutation_form(element, width) for element in backend.elements
            )
        )
    candidates: list[
        tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]
    ] = []
    distinct: set[tuple[tuple[int, ...], ...]] = set()
    for element in elements:
        permuted = tuple(
            tuple(row[element[column]] for column in range(width))
            for row in encoder.generator_matrix
        )
        reduced, pivots = _rref(
            permuted,
            encoder.field_order,
            width,
        )
        # Coordinate permutations preserve rank. Keep the assertion local to
        # the producer so a future action adapter cannot silently emit a
        # malformed canonical encoder.
        if len(pivots) != len(encoder.generator_matrix):
            raise OperationDomainValidationError(
                location=("encoder", "generator_matrix"),
                code="code.canonicalization.transported_rank",
                message="a coordinate action must preserve generator rank",
            )
        distinct.add(reduced)
        candidates.append(
            (
                reduced,
                tuple(element),
                tuple(
                    encoder.coordinate_axis[element[column]] for column in range(width)
                ),
            )
        )
    matrix, transporter, transported_axis = min(candidates)
    orbit_size = len(distinct)
    return LinearCodeCanonicalizationResult(
        source=request,
        canonical_encoder=PrimeFieldLinearEncoder(
            field_order=encoder.field_order,
            message_axis=tuple(f"m{index}" for index in range(len(matrix))),
            coordinate_axis=transported_axis,
            generator_matrix=matrix,
        ),
        transporter=transporter,
        transported_axis=transported_axis,
        orbit_size=orbit_size,
        stabilizer_size=order // orbit_size,
    )


__all__ = ["canonicalize_linear_code"]
