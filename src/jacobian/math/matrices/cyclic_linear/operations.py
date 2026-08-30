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
from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_WORK,
    CyclicRationalBlockSymbol,
    CyclicRationalRankKernelProfile,
    CyclotomicNonzeroMinor,
    CyclotomicRankKernelComponent,
)
from jacobian.math.matrices.values import (
    RationalVectorSpaceBasis,
    SimpleNumberFieldMatrix,
    SimpleNumberFieldVectorSpaceBasis,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

type FieldCoordinates = tuple[Fraction, ...]
type ComponentCoordinates = tuple[tuple[FieldCoordinates, ...], ...]


class CyclicRankKernelAdmissionError(ValueError):
    """A proved owner-local resource rejection before exact elimination."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _ComponentAdmission:
    order: int
    degree: int
    field: SimpleNumberFieldPresentation
    matrix_coordinates: ComponentCoordinates
    component_matrix: SimpleNumberFieldMatrix
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


def _lcm(left: int, right: int) -> int:
    return left // gcd(left, right) * right


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
    field: SimpleNumberFieldPresentation,
    coordinates: FieldCoordinates,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coordinates
        ),
    )


def _public_component_matrix(
    field: SimpleNumberFieldPresentation,
    coordinates: ComponentCoordinates,
) -> SimpleNumberFieldMatrix:
    return SimpleNumberFieldMatrix(
        presentation=field,
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
) -> tuple[int, int, int]:
    values = tuple(
        coordinate for row in coordinates for entry in row for coordinate in entry
    )
    denominator = 1
    for value in values:
        denominator = _lcm(denominator, value.denominator)
    scaled_height = max(
        (abs(value.numerator) * (denominator // value.denominator) for value in values),
        default=0,
    )
    coordinate_digits = max(
        (
            max(_decimal_digits(value.numerator), _decimal_digits(value.denominator))
            for value in values
        ),
        default=1,
    )
    return denominator, scaled_height, coordinate_digits


def _idempotent_height(coefficients: tuple[Fraction, ...]) -> tuple[int, int]:
    denominator = 1
    for value in coefficients:
        denominator = _lcm(denominator, value.denominator)
    scaled_height = max(
        (
            abs(value.numerator) * (denominator // value.denominator)
            for value in coefficients
        ),
        default=0,
    )
    return denominator, scaled_height


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

    for order in _divisors(symbol.period):
        _require_execution_active(f"before order-{order} admission")
        polynomial = sympy.Poly(
            sympy.cyclotomic_poly(order, variable), variable, domain=sympy.QQ
        )
        coefficients_descending = tuple(
            format_canonical_integer(int(coefficient))
            for coefficient in polynomial.all_coeffs()
        )
        field = SimpleNumberFieldPresentation(
            coefficients_descending=coefficients_descending
        )
        degree = field.degree
        matrix_coordinates = _component_coordinates(
            symbol, polynomial, degree, variable
        )
        common_denominator, scaled_height, matrix_digits = _matrix_coordinate_height(
            matrix_coordinates
        )
        if matrix_digits > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS:
            raise CyclicRankKernelAdmissionError(
                "component_coordinate_bound",
                "a specialized cyclotomic matrix coefficient exceeds the "
                f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit field-element bound",
            )

        multiplication_norm = _multiplication_norm(polynomial, degree, variable)
        maximum_rank = _structural_rank_bound(matrix_coordinates)
        if scaled_height == 0:
            determinant_bound = 1
        else:
            determinant_bound = (
                factorial(maximum_rank)
                * multiplication_norm ** max(maximum_rank - 1, 0)
                * scaled_height**maximum_rank
            )
        determinant_denominator_bound = common_denominator**maximum_rank
        determinant_digits = max(
            _decimal_digits(determinant_bound),
            _decimal_digits(determinant_denominator_bound),
        )
        if determinant_digits > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS:
            raise CyclicRankKernelAdmissionError(
                "elimination_height_bound",
                "fraction-free component elimination can exceed the "
                f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit exact-result envelope",
            )

        idempotent_coefficients, crt_idempotent = _crt_idempotent(
            total_polynomial, polynomial, symbol.period
        )
        idempotent_denominator, idempotent_height = _idempotent_height(
            idempotent_coefficients
        )

        component_matrix = _public_component_matrix(field, matrix_coordinates)
        field_bytes = len(field.model_dump_json().encode("utf-8"))
        fixed_component_bytes = (
            len(component_matrix.model_dump_json().encode("utf-8"))
            + len(crt_idempotent.model_dump_json().encode("utf-8"))
            + 4_096
        )
        maximum_component_bytes = 0
        maximum_global_component_bytes = 0
        for possible_rank in range(maximum_rank + 1):
            possible_determinant_bound = (
                1
                if possible_rank == 0
                else factorial(possible_rank)
                * multiplication_norm ** (possible_rank - 1)
                * scaled_height**possible_rank
            )
            possible_determinant_digits = max(
                _decimal_digits(possible_determinant_bound),
                _decimal_digits(common_denominator**possible_rank),
            )
            nullity = symbol.source_block_dimension - possible_rank
            reconstruction_digits = 1
            if nullity:
                reconstruction_numerator_bound = max(
                    1,
                    degree
                    * multiplication_norm
                    * possible_determinant_bound
                    * max(idempotent_height, 1),
                )
                reconstruction_digits = max(
                    _decimal_digits(reconstruction_numerator_bound),
                    _decimal_digits(idempotent_denominator),
                )
                if reconstruction_digits > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS:
                    raise CyclicRankKernelAdmissionError(
                        "reconstruction_height_bound",
                        "CRT kernel reconstruction can exceed the 256-digit rational basis envelope",
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
        field_work += degree * max(len(symbol.entries), 1)
        field_work += degree**3 + symbol.period * degree**2
        field_work += symbol.source_block_dimension**2 * degree**2 * symbol.period
        field_work += degree**2 * (
            symbol.target_block_dimension
            * symbol.source_block_dimension
            * max(maximum_rank, 1)
            + maximum_rank**3
        )
        if field_work > MAX_CYCLIC_FIELD_WORK:
            raise CyclicRankKernelAdmissionError(
                "field_work_bound",
                "exact cyclotomic elimination exceeds the 100,000,000-unit field-work envelope",
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

    public_kernel = SimpleNumberFieldVectorSpaceBasis(
        presentation=admission.field,
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
