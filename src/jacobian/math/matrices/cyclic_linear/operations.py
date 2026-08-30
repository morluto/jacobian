"""Exact cyclotomic decomposition of bounded rational cyclic linear maps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from math import factorial, gcd
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import CanonicalLimits
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
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and time.monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


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
    remainders = tuple(
        sympy.Poly(variable**power, variable, domain=sympy.QQ).rem(polynomial)
        for power in range(2 * degree - 1)
    )
    for left_power in range(degree):
        for right_power in range(degree):
            coordinates = _polynomial_coordinates(
                remainders[left_power + right_power], degree
            )
            for output_power, coefficient in enumerate(coordinates):
                if coefficient.denominator != 1:  # pragma: no cover
                    raise RuntimeError("monic cyclotomic reduction left ZZ")
                output_sums[output_power] += abs(coefficient.numerator)
    return max(output_sums, default=1)


def _component_coordinates(
    symbol: CyclicRationalBlockSymbol,
    polynomial: Any,
    degree: int,
    variable: Any,
) -> ComponentCoordinates:
    import sympy

    shift_coordinates = tuple(
        _polynomial_coordinates(
            sympy.Poly(variable**shift, variable, domain=sympy.QQ).rem(polynomial),
            degree,
        )
        for shift in range(symbol.period)
    )
    rows = [
        [
            [Fraction(0) for _ in range(degree)]
            for _ in range(symbol.source_block_dimension)
        ]
        for _ in range(symbol.target_block_dimension)
    ]
    for entry in symbol.entries:
        coefficient = entry.coefficient.as_fraction()
        target = rows[entry.target_coordinate][entry.source_coordinate]
        for power, reduced_coefficient in enumerate(shift_coordinates[entry.shift]):
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
    predicted_result_bytes = len(symbol.model_dump_json().encode("utf-8")) + 8_192
    predicted_global_basis_bytes = 0
    field_work = 0
    global_dimension = symbol.period * symbol.source_block_dimension
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
        field_bytes = len(field.model_dump_json().encode("utf-8"))
        fixed_component_bytes = (
            len(component_matrix.model_dump_json().encode("utf-8"))
            + len(crt_idempotent.model_dump_json().encode("utf-8"))
            + 4_096
        )
        maximum_component_bytes = 0
        maximum_global_component_bytes = 0
        maximum_scalar_bits = max(
            source_scalar_bits,
            matrix_scalar_bits,
            multiplication_norm.bit_length(),
            idempotent_scalar_bits,
            determinant_bound.bit_length(),
            determinant_denominator_bound.bit_length(),
            *(abs(int(value)).bit_length() for value in polynomial.all_coeffs()),
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
            possible_determinant_digits = max(
                _decimal_digits(possible_determinant_bound),
                _decimal_digits(possible_denominator_bound),
            )
            nullity = symbol.source_block_dimension - possible_rank
            reconstruction_digits = 1
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
                reconstruction_digits = max(
                    _decimal_digits(max(reconstruction_numerator_bound, 1)),
                    _decimal_digits(idempotent_denominator),
                )
                maximum_scalar_bits = max(
                    maximum_scalar_bits,
                    reconstruction_numerator_bound.bit_length(),
                    idempotent_denominator.bit_length(),
                )
            scalar_bytes = (
                field_bytes + degree * (2 * possible_determinant_digits + 80) + 256
            )
            maximum_component_bytes = max(
                maximum_component_bytes,
                fixed_component_bytes
                + (symbol.source_block_dimension * nullity + int(possible_rank > 0))
                * scalar_bytes,
            )
            maximum_global_component_bytes = max(
                maximum_global_component_bytes,
                degree * nullity * global_dimension * (2 * reconstruction_digits + 64),
            )
        predicted_result_bytes += maximum_component_bytes
        predicted_global_basis_bytes += maximum_global_component_bytes
        component_arithmetic_units = (
            degree * max(len(symbol.entries), 1)
            + degree**3
            + symbol.period * degree**2
            + symbol.source_block_dimension**2 * degree**2 * symbol.period
            + degree**2
            * (
                symbol.target_block_dimension
                * symbol.source_block_dimension
                * max(maximum_rank, 1)
                + maximum_rank**3
            )
            + degree
            * symbol.source_block_dimension**2
            * (degree**2 + degree * symbol.period)
        )
        field_work = _charge_field_work(
            field_work,
            component_arithmetic_units,
            max(maximum_scalar_bits, 1),
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

    predicted_result_bytes += predicted_global_basis_bytes
    maximum_result_bytes = CanonicalLimits().max_output_bytes
    if predicted_result_bytes > maximum_result_bytes:
        raise CyclicRankKernelAdmissionError(
            "result_byte_bound",
            "the retained complete cyclic profile exceeds the "
            f"{maximum_result_bytes:,}-byte canonical result envelope",
        )
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


def _compute_component(admission: _ComponentAdmission) -> _ComputedComponent:
    from sympy import QQ
    from sympy.polys.matrices import DomainMatrix

    order = admission.order
    _require_execution_active(f"before order-{order} field construction")
    field = QQ.cyclotomic_field(order)
    generator = field.convert(field.ext)
    denominator = admission.common_denominator
    backend_rows = [
        [
            _backend_element(
                field,
                generator,
                tuple(coordinate * denominator for coordinate in value),
            )
            for value in row
        ]
        for row in admission.matrix_coordinates
    ]
    matrix = DomainMatrix(
        backend_rows,
        (len(backend_rows), len(backend_rows[0])),
        field,
    )
    _require_execution_active(f"before order-{order} exact row reduction")
    _reduced, pivot_columns = matrix.rref()
    _require_execution_active(f"after order-{order} exact row reduction")
    rank = len(pivot_columns)
    source_dimension = len(backend_rows[0])
    kernel_vectors: list[tuple[Any, ...]] = []
    nonzero_minor: CyclotomicNonzeroMinor | None = None

    if rank == 0:
        kernel_vectors.extend(
            tuple(
                field.one if index == free else field.zero
                for index in range(source_dimension)
            )
            for free in range(source_dimension)
        )
    else:
        column_basis = matrix.extract(range(len(backend_rows)), pivot_columns)
        _ignored, row_indices = column_basis.transpose().rref()
        _require_execution_active(f"after order-{order} pivot-row selection")
        pivot_minor = matrix.extract(row_indices, pivot_columns)
        original_determinant_coordinates: FieldCoordinates
        if admission.degree == 1:
            from jacobian.math.matrices._flint import rational_determinant

            original_determinant_coordinates = (
                rational_determinant(
                    tuple(
                        tuple(
                            admission.matrix_coordinates[row][column][0]
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
                admission.degree,
            )
        _require_execution_active(f"after order-{order} exact minor determinant")
        if not scaled_determinant:  # pragma: no cover
            raise RuntimeError("selected cyclotomic rank minor vanished")
        nonzero_minor = CyclotomicNonzeroMinor(
            row_indices=tuple(row_indices),
            column_indices=tuple(pivot_columns),
            determinant=_public_element(
                admission.field,
                original_determinant_coordinates,
            ),
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
            # Fraction-free RREF expresses the solve by rank-minor numerators
            # over one rank-minor denominator.  Admission's determinant bound
            # therefore covers every coordinate retained below without an
            # inverse or adjugate expansion.
            reduced, solution_denominator, augmented_pivots = augmented.rref_den(
                method="FF"
            )
            if augmented_pivots != tuple(range(rank)):  # pragma: no cover
                raise RuntimeError("fraction-free cyclotomic solve lost pivot rank")
            solution_rows = reduced.extract(
                range(rank), range(rank, rank + len(free_columns))
            ).to_list()
            _require_execution_active(f"after order-{order} fraction-free kernel solve")
        for free_index, free in enumerate(free_columns):
            vector = [field.zero for _ in range(source_dimension)]
            vector[free] = solution_denominator
            for index, pivot in enumerate(pivot_columns):
                vector[pivot] = -solution_rows[index][free_index]
            kernel_vectors.append(tuple(vector))

    public_kernel = RationalCyclotomicVectorSpaceBasis(
        field=admission.field,
        ambient_dimension=source_dimension,
        vectors=tuple(
            tuple(
                _public_element(
                    admission.field,
                    _backend_coordinates(value, admission.degree),
                )
                for value in vector
            )
            for vector in kernel_vectors
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
