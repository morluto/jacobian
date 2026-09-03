# ruff: noqa: B904
"""Exact cyclotomic decomposition of bounded rational cyclic linear maps."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from fractions import Fraction
from math import factorial, gcd
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    MAX_CYCLIC_FIELD_WORK,
    CyclicRationalBlockSymbol,
    CyclicRationalRankKernelProfile,
    CyclotomicNonzeroMinor,
    CyclotomicRankKernelComponent,
    RationalCyclotomicElement,
    RationalCyclotomicField,
    RationalCyclotomicMatrix,
    RationalCyclotomicVectorSpaceBasis,
)
from jacobian.math.matrices.values import RationalVectorSpaceBasis
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

type FieldCoordinates = tuple[Fraction, ...]
type ComponentCoordinates = tuple[tuple[FieldCoordinates, ...], ...]

_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE = 10**MAX_CYCLIC_FIELD_ELEMENT_DIGITS - 1
_CYCLIC_PROFILE_WALL_SECONDS = 3_600.0
_ADMISSION_CHECK_INTERVAL = 256


class CyclicRankKernelAdmissionError(ValueError):
    """A proved owner-local resource rejection before exact elimination."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _ComponentAdmission:
    order: int
    degree: int
    field: RationalCyclotomicField
    matrix_coordinates: ComponentCoordinates
    component_matrix: RationalCyclotomicMatrix
    common_denominator: int
    idempotent_coefficients: tuple[Fraction, ...]
    crt_idempotent: RationalPolynomial


@dataclass(frozen=True, slots=True)
class _ComputedComponent:
    public: CyclotomicRankKernelComponent
    backend_field: Any
    generator: Any
    kernel_vectors: tuple[tuple[Any, ...], ...]
    admission: _ComponentAdmission


def _require_execution_active(phase: str) -> None:
    request_checkpoint(phase)


def _bind_cyclic_profile_deadline() -> None:
    execution = current_request_execution()
    if execution is None:  # pragma: no cover - native entry establishes the context
        raise RuntimeError("cyclic-profile execution context is missing")
    owner_deadline = execution.started_at + _CYCLIC_PROFILE_WALL_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


def _bounded_product(*factors: int, limit: int) -> int | None:
    """Multiply nonnegative integers without constructing a value past ``limit``."""

    result = 1
    for factor in factors:
        if factor < 0:  # pragma: no cover - every caller supplies magnitudes
            raise ValueError("bounded product factors must be nonnegative")
        if factor == 0:
            return 0
        if result > limit // factor:
            return None
        result *= factor
    return result


def _bounded_power(base: int, exponent: int, *, limit: int) -> int | None:
    """Raise a nonnegative integer without materializing an over-limit power."""

    result = 1
    for _ in range(exponent):
        product = _bounded_product(result, base, limit=limit)
        if product is None:
            return None
        result = product
    return result


def _bounded_lcm(
    left: int,
    right: int,
    *,
    maximum_power: int,
    limit: int,
) -> int | None:
    """Extend an LCM only when its priced determinant power remains bounded."""

    reduced = right // gcd(left, right)
    candidate = _bounded_product(left, reduced, limit=limit)
    if candidate is None:
        return None
    if _bounded_power(candidate, maximum_power, limit=limit) is None:
        return None
    return candidate


def _determinant_magnitude_bound(
    *,
    rank: int,
    multiplication_norm: int,
    scaled_height: int,
) -> int | None:
    if rank == 0 or scaled_height == 0:
        return 1
    norm_power = _bounded_power(
        multiplication_norm,
        rank - 1,
        limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
    )
    height_power = _bounded_power(
        scaled_height,
        rank,
        limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
    )
    if norm_power is None or height_power is None:
        return None
    return _bounded_product(
        factorial(rank),
        norm_power,
        height_power,
        limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
    )


def _charge_field_work(current: int, *factors: int) -> int:
    remaining = MAX_CYCLIC_FIELD_WORK - current
    charge = _bounded_product(*factors, limit=max(remaining, 0))
    if charge is None or charge > remaining:
        raise CyclicRankKernelAdmissionError(
            "field_work_bound",
            "exact cyclotomic arithmetic exceeds the 100,000,000-unit "
            "scalar-bit work envelope",
        )
    return current + charge


def _decimal_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    # A strict integer upper approximation to log10(2), followed by one exact
    # comparison, avoids decimal conversion of an already rejected bigint.
    digits = magnitude.bit_length() * 30_103 // 100_000 + 1
    if magnitude < 10 ** (digits - 1):
        return digits - 1
    return digits


def _fraction(value: Any) -> Fraction:
    numerator, denominator = value.as_numer_denom()
    return Fraction(int(numerator), int(denominator))


def _polynomial_coordinates(polynomial: Any, length: int) -> FieldCoordinates:
    return tuple(_fraction(polynomial.nth(power)) for power in range(length))


def _public_polynomial(coefficients: tuple[Fraction, ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=(power,),
                )
                for power in range(len(coefficients) - 1, -1, -1)
                if (coefficient := coefficients[power])
            )
        ),
    )


def _public_element(
    field: RationalCyclotomicField,
    coordinates: FieldCoordinates,
) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coordinates
        ),
    )


def _public_component_matrix(
    field: RationalCyclotomicField,
    coordinates: ComponentCoordinates,
) -> RationalCyclotomicMatrix:
    return RationalCyclotomicMatrix(
        field=field,
        entries=tuple(
            tuple(_public_element(field, value) for value in row) for row in coordinates
        ),
    )


def _multiplication_norm(polynomial: Any, degree: int, variable: Any) -> int:
    """Return an exact coefficient-norm constant for the quotient power basis.

    If power-basis coordinate sup norms are ``A`` and ``B``, multiplication in
    ``QQ[x]/(polynomial)`` has coordinate sup norm at most ``constant*A*B``.
    The constant is the greatest absolute structure-constant row sum.
    """

    import sympy

    output_sums = [0] * degree
    remainders = []
    for power in range(2 * degree - 1):
        if power and power % _ADMISSION_CHECK_INTERVAL == 0:
            _require_execution_active("during cyclotomic remainder admission")
        remainders.append(
            sympy.Poly(variable**power, variable, domain=sympy.QQ).rem(polynomial)
        )
    _require_execution_active("during cyclotomic multiplication-norm admission")
    for total_power, remainder in enumerate(remainders):
        if total_power and total_power % _ADMISSION_CHECK_INTERVAL == 0:
            _require_execution_active(
                "during cyclotomic multiplication-norm admission"
            )
        multiplicity = min(total_power + 1, 2 * degree - 1 - total_power)
        coordinates = _polynomial_coordinates(remainder, degree)
        for output_power, coefficient in enumerate(coordinates):
            if coefficient.denominator != 1:  # pragma: no cover
                raise RuntimeError("monic cyclotomic reduction left ZZ")
            output_sums[output_power] += multiplicity * abs(coefficient.numerator)
    return max(output_sums, default=1)


def _component_coordinates(
    symbol: CyclicRationalBlockSymbol,
    polynomial: Any,
    degree: int,
    variable: Any,
) -> ComponentCoordinates:
    import sympy

    shift_coordinates = []
    for shift in range(symbol.period):
        if shift and shift % _ADMISSION_CHECK_INTERVAL == 0:
            _require_execution_active("during cyclotomic shift-coordinate admission")
        shift_coordinates.append(
            _polynomial_coordinates(
                sympy.Poly(variable**shift, variable, domain=sympy.QQ).rem(polynomial),
                degree,
            )
        )
    rows = [
        [
            [Fraction(0) for _ in range(degree)]
            for _ in range(symbol.source_block_dimension)
        ]
        for _ in range(symbol.target_block_dimension)
    ]
    for entry_index, entry in enumerate(symbol.entries):
        if entry_index and entry_index % _ADMISSION_CHECK_INTERVAL == 0:
            _require_execution_active("during cyclotomic matrix-coordinate admission")
        coefficient = entry.coefficient.as_fraction()
        target = rows[entry.target_coordinate][entry.source_coordinate]
        for power, reduced_coefficient in enumerate(shift_coordinates[entry.shift]):
            if power and power % _ADMISSION_CHECK_INTERVAL == 0:
                _require_execution_active(
                    "during cyclotomic matrix-coordinate admission"
                )
            target[power] += coefficient * reduced_coefficient
    return tuple(
        tuple(tuple(coordinate for coordinate in value) for value in row)
        for row in rows
    )


def _crt_idempotent(
    total_polynomial: Any,
    cyclotomic_polynomial: Any,
    period: int,
) -> tuple[tuple[Fraction, ...], RationalPolynomial]:
    from sympy import invert

    complement = total_polynomial.exquo(cyclotomic_polynomial)
    inverse = invert(complement, cyclotomic_polynomial)
    idempotent = (complement * inverse).rem(total_polynomial)
    coefficients = _polynomial_coordinates(idempotent, period)
    return coefficients, _public_polynomial(coefficients)


def _matrix_coordinate_height(
    coordinates: ComponentCoordinates,
    *,
    maximum_rank: int,
) -> tuple[int, int, int, int]:
    values = tuple(
        coordinate for row in coordinates for entry in row for coordinate in entry
    )
    denominator = 1
    for value in values:
        if (
            abs(value.numerator) > _MAX_CYCLOTOMIC_SCALAR_MAGNITUDE
            or value.denominator > _MAX_CYCLOTOMIC_SCALAR_MAGNITUDE
        ):
            raise CyclicRankKernelAdmissionError(
                "component_coordinate_bound",
                "a specialized cyclotomic matrix coefficient exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit field-element bound",
            )
        extended = _bounded_lcm(
            denominator,
            value.denominator,
            maximum_power=max(maximum_rank, 1),
            limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
        )
        if extended is None:
            raise CyclicRankKernelAdmissionError(
                "elimination_height_bound",
                "the common denominator or its rank power exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit exact-result envelope",
            )
        denominator = extended
    scaled_height = 0
    for value in values:
        scaled = _bounded_product(
            abs(value.numerator),
            denominator // value.denominator,
            limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
        )
        if scaled is None:
            raise CyclicRankKernelAdmissionError(
                "elimination_height_bound",
                "clearing component denominators exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit exact-result envelope",
            )
        scaled_height = max(scaled_height, scaled)
    coordinate_digits = max(
        (
            max(_decimal_digits(value.numerator), _decimal_digits(value.denominator))
            for value in values
        ),
        default=1,
    )
    scalar_bits = max(
        denominator.bit_length(),
        scaled_height.bit_length(),
        *(
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for value in values
        ),
    )
    return denominator, scaled_height, coordinate_digits, scalar_bits


def _idempotent_height(
    coefficients: tuple[Fraction, ...],
) -> tuple[int, int, int]:
    denominator = 1
    for value in coefficients:
        extended = _bounded_lcm(
            denominator,
            value.denominator,
            maximum_power=1,
            limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
        )
        if extended is None:
            raise CyclicRankKernelAdmissionError(
                "reconstruction_height_bound",
                "a CRT idempotent denominator exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit rational basis envelope",
            )
        denominator = extended
    scaled_height = 0
    for value in coefficients:
        scaled = _bounded_product(
            abs(value.numerator),
            denominator // value.denominator,
            limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
        )
        if scaled is None:
            raise CyclicRankKernelAdmissionError(
                "reconstruction_height_bound",
                "clearing CRT idempotent denominators exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit rational basis envelope",
            )
        scaled_height = max(scaled_height, scaled)
    scalar_bits = max(denominator.bit_length(), scaled_height.bit_length())
    return denominator, scaled_height, scalar_bits


def _structural_rank_bound(coordinates: ComponentCoordinates) -> int:
    """Return the maximum matching bound of the exact nonzero support."""

    import networkx as nx

    row_nodes = tuple(("row", row) for row in range(len(coordinates)))
    column_nodes = tuple(("column", column) for column in range(len(coordinates[0])))
    graph: nx.Graph[tuple[str, int]] = nx.Graph()
    graph.add_nodes_from(row_nodes, bipartite=0)
    graph.add_nodes_from(column_nodes, bipartite=1)
    graph.add_edges_from(
        (row_nodes[row], column_nodes[column])
        for row, entries in enumerate(coordinates)
        for column, value in enumerate(entries)
        if any(value)
    )
    matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=row_nodes)
    return len(matching) // 2


def _admit_cyclic_symbol(
    symbol: CyclicRationalBlockSymbol,
) -> tuple[_ComponentAdmission, ...]:
    """Materialize bounded quotient data and prove all later exact outputs fit."""

    import sympy

    _require_execution_active("before cyclic-profile admission")
    variable = sympy.Symbol("x")
    total_polynomial = sympy.Poly(
        variable**symbol.period - 1, variable, domain=sympy.QQ
    )
    components: list[_ComponentAdmission] = []
    field_work = 0
    source_scalar_bits = max(
        (
            max(
                abs(entry.coefficient.as_integer_ratio()[0]).bit_length(),
                entry.coefficient.as_integer_ratio()[1].bit_length(),
            )
            for entry in symbol.entries
        ),
        default=1,
    )

    for order in _divisors(symbol.period):
        _require_execution_active(f"before order-{order} admission")
        polynomial = sympy.Poly(
            sympy.cyclotomic_poly(order, variable), variable, domain=sympy.QQ
        )
        field = RationalCyclotomicField(order=order)
        degree = field.degree
        matrix_coordinates = _component_coordinates(
            symbol, polynomial, degree, variable
        )
        maximum_rank = _structural_rank_bound(matrix_coordinates)
        minimum_rank = int(
            any(any(value) for row in matrix_coordinates for value in row)
        )
        (
            common_denominator,
            scaled_height,
            matrix_digits,
            matrix_scalar_bits,
        ) = _matrix_coordinate_height(
            matrix_coordinates,
            maximum_rank=maximum_rank,
        )
        if matrix_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
            raise CyclicRankKernelAdmissionError(
                "component_coordinate_bound",
                "a specialized cyclotomic matrix coefficient exceeds the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit field-element bound",
            )

        multiplication_norm = _multiplication_norm(polynomial, degree, variable)
        determinant_bound = _determinant_magnitude_bound(
            rank=maximum_rank,
            multiplication_norm=multiplication_norm,
            scaled_height=scaled_height,
        )
        determinant_denominator_bound = _bounded_power(
            common_denominator,
            maximum_rank,
            limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
        )
        if determinant_bound is None or determinant_denominator_bound is None:
            raise CyclicRankKernelAdmissionError(
                "elimination_height_bound",
                "fraction-free component elimination can exceed the "
                f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit exact-result envelope",
            )
        idempotent_coefficients, crt_idempotent = _crt_idempotent(
            total_polynomial, polynomial, symbol.period
        )
        (
            idempotent_denominator,
            idempotent_height,
            idempotent_scalar_bits,
        ) = _idempotent_height(idempotent_coefficients)

        component_matrix = _public_component_matrix(field, matrix_coordinates)
        cyclotomic_structure_scalar_bits = max(
            multiplication_norm.bit_length(),
            idempotent_scalar_bits,
            *(abs(int(value)).bit_length() for value in polynomial.all_coeffs()),
        )
        source_specialization_scalar_bits = max(
            source_scalar_bits,
            matrix_scalar_bits,
            cyclotomic_structure_scalar_bits,
        )
        elimination_scalar_bits = max(
            matrix_scalar_bits,
            multiplication_norm.bit_length(),
            determinant_bound.bit_length(),
            determinant_denominator_bound.bit_length(),
        )
        reconstruction_scalar_bits = max(
            elimination_scalar_bits,
            idempotent_scalar_bits,
        )
        for possible_rank in range(maximum_rank + 1):
            possible_determinant_bound = _determinant_magnitude_bound(
                rank=possible_rank,
                multiplication_norm=multiplication_norm,
                scaled_height=scaled_height,
            )
            possible_denominator_bound = _bounded_power(
                common_denominator,
                possible_rank,
                limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
            )
            if (
                possible_determinant_bound is None or possible_denominator_bound is None
            ):  # pragma: no cover - the maximum-rank guard dominates
                raise RuntimeError("admitted rank height was not monotone")
            nullity = symbol.source_block_dimension - possible_rank
            if nullity:
                reconstruction_numerator_bound = _bounded_product(
                    degree,
                    multiplication_norm,
                    possible_determinant_bound,
                    max(idempotent_height, 1),
                    limit=_MAX_CYCLOTOMIC_SCALAR_MAGNITUDE,
                )
                if reconstruction_numerator_bound is None:
                    raise CyclicRankKernelAdmissionError(
                        "reconstruction_height_bound",
                        "CRT kernel reconstruction can exceed the "
                        f"{MAX_CYCLIC_FIELD_ELEMENT_DIGITS}-digit rational basis envelope",
                    )
                reconstruction_scalar_bits = max(
                    reconstruction_scalar_bits,
                    reconstruction_numerator_bound.bit_length(),
                    idempotent_denominator.bit_length(),
                )
        # Price source-height arithmetic separately from the small fixed
        # cyclotomic structure constants used to construct this component.
        source_specialization_units = degree * max(len(symbol.entries), 1)
        cyclotomic_structure_units = degree**3 + symbol.period * degree**2
        elimination_units = degree**2 * (
            symbol.target_block_dimension
            * symbol.source_block_dimension
            * max(maximum_rank, 1)
            + maximum_rank**3
        )
        maximum_nullity = symbol.source_block_dimension - minimum_rank
        reconstruction_units = (
            degree
            * maximum_nullity
            * symbol.source_block_dimension
            * (degree**2 + degree * symbol.period)
        )
        for units, scalar_bits in (
            (source_specialization_units, source_specialization_scalar_bits),
            (cyclotomic_structure_units, cyclotomic_structure_scalar_bits),
            (elimination_units, elimination_scalar_bits),
            (reconstruction_units, reconstruction_scalar_bits),
        ):
            field_work = _charge_field_work(
                field_work,
                units,
                max(scalar_bits, 1),
            )

        components.append(
            _ComponentAdmission(
                order=order,
                degree=degree,
                field=field,
                matrix_coordinates=matrix_coordinates,
                component_matrix=component_matrix,
                common_denominator=common_denominator,
                idempotent_coefficients=idempotent_coefficients,
                crt_idempotent=crt_idempotent,
            )
        )
        _require_execution_active(f"after order-{order} admission")

    return tuple(components)


def _backend_element(field: Any, generator: Any, coordinates: FieldCoordinates) -> Any:
    from sympy import QQ

    result = field.zero
    power = field.one
    for coefficient in coordinates:
        if coefficient:
            result += field.convert(QQ.convert(coefficient)) * power
        power *= generator
    return result


def _backend_coordinates(value: Any, degree: int) -> FieldCoordinates:
    # SymPy ANP lists are descending and omit leading zero coordinates.  A
    # field multiplication normalizes the degree-one generator as well.
    descending = list(value.to_list())
    if len(descending) > degree:  # pragma: no cover
        raise RuntimeError("cyclotomic backend returned an unreduced field element")
    descending = [0] * (degree - len(descending)) + descending
    return tuple(
        Fraction(int(coefficient.numerator), int(coefficient.denominator))
        for coefficient in reversed(descending)
    )


def _cyclotomic_kernel_child(
    order: int,
    degree: int,
    matrix_coordinates: ComponentCoordinates,
    common_denominator: int,
    conn: multiprocessing.connection.Connection | None = None,
) -> tuple[Any, ...]:
    """Run SymPy rref/det/rref_den and return the result tuple.

    This function receives only picklable primitive data, reconstructs
    the SymPy objects, performs the heavy linear algebra, and returns
    plain coordinate tuples. When ``conn`` is provided (multiprocessing
    mode), the result is sent through the pipe instead of returned.
    """
    from sympy import QQ
    from sympy.polys.matrices import DomainMatrix

    field = QQ.cyclotomic_field(order)
    generator = field.convert(field.ext)
    denominator = common_denominator
    backend_rows = [
        [
            _backend_element(
                field,
                generator,
                tuple(coordinate * denominator for coordinate in value),
            )
            for value in row
        ]
        for row in matrix_coordinates
    ]
    matrix = DomainMatrix(
        backend_rows,
        (len(backend_rows), len(backend_rows[0])),
        field,
    )
    _reduced, pivot_columns = matrix.rref()
    rank = len(pivot_columns)
    source_dimension = len(backend_rows[0])
    kernel_coords: list[list[FieldCoordinates]] = []
    nonzero_minor_data: tuple[Any, ...] | None = None

    if rank == 0:
        for free in range(source_dimension):
            vector = [field.zero for _ in range(source_dimension)]
            vector[free] = field.one
            kernel_coords.append(
                [_backend_coordinates(value, degree) for value in vector]
            )
    else:
        column_basis = matrix.extract(range(len(backend_rows)), pivot_columns)
        _ignored, row_indices = column_basis.transpose().rref()
        pivot_minor = matrix.extract(row_indices, pivot_columns)
        original_determinant_coordinates: FieldCoordinates
        if degree == 1:
            from jacobian.math.matrices._flint import rational_determinant

            original_determinant_coordinates = (
                rational_determinant(
                    tuple(
                        tuple(
                            matrix_coordinates[row][column][0]
                            for column in pivot_columns
                        )
                        for row in row_indices
                    )
                ),
            )
            scaled_determinant = _backend_element(
                field,
                generator,
                tuple(
                    coordinate * denominator**rank
                    for coordinate in original_determinant_coordinates
                ),
            )
        else:
            scaled_determinant = pivot_minor.det()
            original_determinant_coordinates = _backend_coordinates(
                scaled_determinant
                * field.convert(QQ.convert(Fraction(1, denominator**rank))),
                degree,
            )
        if not scaled_determinant:  # pragma: no cover
            raise RuntimeError("selected cyclotomic rank minor vanished")
        nonzero_minor_data = (
            tuple(row_indices),
            tuple(pivot_columns),
            original_determinant_coordinates,
        )
        pivot_set = set(pivot_columns)
        free_columns = tuple(
            column for column in range(source_dimension) if column not in pivot_set
        )
        solution_rows: list[list[Any]] = [[] for _ in range(rank)]
        solution_denominator = scaled_determinant
        if free_columns:
            right_columns = matrix.extract(row_indices, free_columns)
            augmented = pivot_minor.hstack(right_columns)
            reduced, solution_denominator, augmented_pivots = augmented.rref_den(
                method="FF"
            )
            if augmented_pivots != tuple(range(rank)):  # pragma: no cover
                raise RuntimeError("fraction-free cyclotomic solve lost pivot rank")
            solution_rows = reduced.extract(
                range(rank), range(rank, rank + len(free_columns))
            ).to_list()
        for free_index, free in enumerate(free_columns):
            vector = [field.zero for _ in range(source_dimension)]
            vector[free] = solution_denominator
            for index, pivot in enumerate(pivot_columns):
                vector[pivot] = -solution_rows[index][free_index]
            kernel_coords.append(
                [_backend_coordinates(value, degree) for value in vector]
            )

    result = (rank, source_dimension, nonzero_minor_data, kernel_coords)
    if conn is not None:
        conn.send(result)
    return result


def _compute_component(admission: _ComponentAdmission) -> _ComputedComponent:
    from sympy import QQ

    order = admission.order
    _require_execution_active(f"before order-{order} kernel")

    # Run the SymPy kernel (rref, det, rref_den) in a bounded worker so the
    # parent can kill it on deadline or cancellation.
    from jacobian.math.matrices.cyclic_linear._kernel_process import (
        run_cyclotomic_kernel,
    )

    execution = current_request_execution()
    deadline = execution.deadline if execution is not None else None

    request_checkpoint(f"before order-{order} kernel")

    try:
        rank, source_dimension, nonzero_minor_data, kernel_coords = (
            run_cyclotomic_kernel(
                order,
                admission.degree,
                admission.matrix_coordinates,
                admission.common_denominator,
                deadline,
            )
        )
    except OperationExecutionTimeoutError:
        raise OperationExecutionTimeoutError(
            f"request deadline expired during order-{order} kernel"
        )

    _require_execution_active(f"after order-{order} kernel")

    field = QQ.cyclotomic_field(order)
    generator = field.convert(field.ext)

    nonzero_minor: CyclotomicNonzeroMinor | None = None
    if nonzero_minor_data is not None:
        row_indices, pivot_columns, determinant_coords = nonzero_minor_data
        nonzero_minor = CyclotomicNonzeroMinor(
            row_indices=tuple(row_indices),
            column_indices=tuple(pivot_columns),
            determinant=_public_element(
                admission.field,
                determinant_coords,
            ),
        )

    # Reconstruct backend kernel vectors from serialized coordinates.
    kernel_vectors: list[tuple[Any, ...]] = []
    for vector_coords in kernel_coords:
        vector = [
            _backend_element(field, generator, tuple(coords))
            for coords in vector_coords
        ]
        kernel_vectors.append(tuple(vector))

    public_kernel = RationalCyclotomicVectorSpaceBasis(
        field=admission.field,
        ambient_dimension=source_dimension,
        vectors=tuple(
            tuple(
                _public_element(
                    admission.field,
                    coords,
                )
                for coords in vector_coords
            )
            for vector_coords in kernel_coords
        ),
    )
    component = CyclotomicRankKernelComponent(
        order=order,
        field=admission.field,
        component_matrix=admission.component_matrix,
        rank=rank,
        nullity=source_dimension - rank,
        kernel_basis=public_kernel,
        nonzero_minor=nonzero_minor,
        crt_idempotent=admission.crt_idempotent,
    )
    _require_execution_active(f"after order-{order} result construction")
    return _ComputedComponent(
        public=component,
        backend_field=field,
        generator=generator,
        kernel_vectors=tuple(kernel_vectors),
        admission=admission,
    )


def _cyclic_product(
    left: FieldCoordinates,
    right: tuple[Fraction, ...],
    period: int,
) -> tuple[Fraction, ...]:
    result = [Fraction(0) for _ in range(period)]
    for left_power, left_coefficient in enumerate(left):
        if not left_coefficient:
            continue
        for right_power, right_coefficient in enumerate(right):
            if right_coefficient:
                result[(left_power + right_power) % period] += (
                    left_coefficient * right_coefficient
                )
    return tuple(result)


def _reconstruct_global_kernel(
    symbol: CyclicRationalBlockSymbol,
    components: tuple[_ComputedComponent, ...],
) -> RationalVectorSpaceBasis:
    global_vectors: list[tuple[CanonicalRational, ...]] = []
    period = symbol.period
    block_dimension = symbol.source_block_dimension
    for component in components:
        power = component.backend_field.one
        powers: list[Any] = []
        for _ in range(component.admission.degree):
            powers.append(power)
            power *= component.generator
        for vector in component.kernel_vectors:
            for basis_power in powers:
                coordinate_polynomials = tuple(
                    _backend_coordinates(
                        value * basis_power, component.admission.degree
                    )
                    for value in vector
                )
                cyclic_coordinates = tuple(
                    _cyclic_product(
                        coordinates,
                        component.admission.idempotent_coefficients,
                        period,
                    )
                    for coordinates in coordinate_polynomials
                )
                flattened = tuple(
                    CanonicalRational.from_fraction(
                        cyclic_coordinates[block_coordinate][shift]
                    )
                    for shift in range(period)
                    for block_coordinate in range(block_dimension)
                )
                global_vectors.append(flattened)
                _require_execution_active(
                    "during order-"
                    f"{component.admission.order} CRT kernel reconstruction"
                )
        _require_execution_active(
            f"after order-{component.admission.order} CRT reconstruction"
        )
    return RationalVectorSpaceBasis(
        ambient_dimension=period * block_dimension,
        vectors=tuple(global_vectors),
    )


def cyclic_rational_rank_kernel_profile(
    symbol: CyclicRationalBlockSymbol,
) -> CyclicRationalRankKernelProfile:
    """Return every rational cyclotomic component rank and kernel exactly.

    The maintained SymPy ``AlgebraicField`` and ``DomainMatrix`` kernels work
    in each ``QQ[x]/(Phi_d)`` component.  Fraction-free minors give a nonzero
    rank witness and deterministic kernel basis; exact CRT idempotents lift
    the component bases back to rational cyclic coordinates.
    """

    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return _cyclic_rational_rank_kernel_profile_in_request(symbol)
    return _cyclic_rational_rank_kernel_profile_in_request(symbol)


def _cyclic_rational_rank_kernel_profile_in_request(
    symbol: CyclicRationalBlockSymbol,
) -> CyclicRationalRankKernelProfile:
    _bind_cyclic_profile_deadline()
    admission = _admit_cyclic_symbol(symbol)
    _require_execution_active("after cyclic-profile admission")
    components = tuple(_compute_component(item) for item in admission)
    global_kernel = _reconstruct_global_kernel(symbol, components)
    global_rank = sum(
        component.admission.degree * component.public.rank for component in components
    )
    global_nullity = sum(
        component.admission.degree * component.public.nullity
        for component in components
    )
    _require_execution_active("after global cyclic-kernel reconstruction")
    return CyclicRationalRankKernelProfile._from_kernel(
        symbol=symbol,
        components=tuple(component.public for component in components),
        global_rank=global_rank,
        global_nullity=global_nullity,
        global_kernel_basis=global_kernel,
    )


__all__ = [
    "CyclicRankKernelAdmissionError",
    "cyclic_rational_rank_kernel_profile",
]
