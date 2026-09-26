"""Exact root system operations."""

from __future__ import annotations

from fractions import Fraction
from math import factorial, isqrt, lcm, prod
from typing import cast

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    IncomparablePair,
    OrderedPair,
    _transitive_reduction,
    canonical_poset_ranks,
    finite_poset_digest,
)
from jacobian.math.groups.root_systems._cartan import (
    VALID_CARTAN_TYPE_RANKS,
    cartan_type_matrix,
    connected_components,
    positive_symmetrizer,
)
from jacobian.math.groups.root_systems._cartan import (
    positive_roots as enumerate_positive_roots,
)
from jacobian.math.groups.root_systems._cartan import (
    simple_reflection as _simple_reflection_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_longest_word as _weyl_longest_word_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_word_descents as _weyl_word_descents_kernel,
)
from jacobian.math.groups.root_systems._cartan import (
    weyl_word_inversions as _weyl_word_inversions_kernel,
)
from jacobian.math.groups.root_systems._dynkin_models import (
    MAX_DYNKIN_DIAGRAM_OUTPUT_CELLS,
    DynkinEdge,
    FiniteDynkinDiagram,
)
from jacobian.math.groups.root_systems._models import (
    MAX_LATTICE_COORDINATE_BITS,
    MAX_LATTICE_OUTPUT_COORDINATE_BITS,
    MAX_LATTICE_VECTOR_OUTPUT_CELLS,
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    MAX_REFLECTION_REPRESENTABLE,
    MAX_ROOT_COORDINATE,
    MAX_ROOT_POSET_ROOTS,
    MAX_WEIGHT_ORBIT_OUTPUT_DIGITS,
    MAX_WEIGHT_ORBIT_SIZE,
    MAX_WEYL_GROUP_ORDER,
    MAX_WEYL_WORD_LENGTH,
    CartanMatrix,
    CartanType,
    CartanTypeResult,
    CorootLatticeVector,
    CoweightLatticeVector,
    FiniteCartanDatum,
    PositiveCorootsResult,
    PositiveRootComponentProfile,
    PositiveRootProfileEntry,
    PositiveRootProfileResult,
    PositiveRootsResult,
    RootComponentData,
    RootCorootPair,
    RootLatticeVector,
    RootLengthClass,
    RootLengthComponentProfile,
    RootLengthProfileResult,
    RootPosetResult,
    RootSystemDataResult,
    RootToCorootResult,
    SimpleReflectionResult,
    SimpleReflectionsResult,
    WeightLatticeVector,
    WeylAntidominantRepresentativeResult,
    WeylDescentsResult,
    WeylDominantRepresentativeResult,
    WeylElement,
    WeylElementLengthResult,
    WeylElementOrderResult,
    WeylExponentComponent,
    WeylExponentsResult,
    WeylGroupOrderResult,
    WeylLongestElementResult,
    WeylParabolicResult,
    WeylParabolicWeightOrbitRequest,
    WeylParabolicWeightOrbitResult,
    WeylPoincarePolynomialResult,
    WeylVectorActionResult,
    WeylWeightOrbitResult,
    _FiniteCartanLatticeVector,
)
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

MAX_SIGNED_ROOT_ACTION_DEGREE = 2 * MAX_POSITIVE_ROOTS
MAX_ROOT_POSET_PAIR_COMPARISONS = (
    2 * (MAX_ROOT_POSET_ROOTS * (MAX_ROOT_POSET_ROOTS - 1) // 2) * MAX_RANK
)
MAX_ROOT_POSET_COVER_WORK = MAX_ROOT_POSET_ROOTS**3
MAX_ROOT_POSET_OUTPUT_PAIRS = MAX_ROOT_POSET_ROOTS * (MAX_ROOT_POSET_ROOTS - 1) // 2
MAX_ROOT_POSET_OUTPUT_CELLS = (
    3 * MAX_RANK**2
    + MAX_RANK
    + MAX_ROOT_POSET_ROOTS * (MAX_RANK + 4)
    + 3 * MAX_ROOT_POSET_OUTPUT_PAIRS
)
MAX_ROOT_PROFILE_WORK = 120_000
MAX_ROOT_PROFILE_OUTPUT_CELLS = 64_000
MAX_COXETER_POLYNOMIAL_WORK = 200_000
MAX_COXETER_POLYNOMIAL_OUTPUT_CELLS = 4_096
MAX_ROOT_TO_COROOT_WORK = 10_000
MAX_ROOT_TO_COROOT_OUTPUT_CELLS = 1_024
MAX_ROOT_LENGTH_PROFILE_WORK = 120_000
MAX_ROOT_LENGTH_PROFILE_OUTPUT_CELLS = 64_000
MAX_WEYL_ELEMENT_ORDER_WORK = 2_000_000
MAX_WEYL_ELEMENT_ORDER_OUTPUT_CELLS = 8_192
MAX_DYNKIN_DIAGRAM_WORK = 5_000


def coxeter_polynomial(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> IntegerPolynomial:
    """Return det(tI-c) for c=s_(r-1)...s_1 s_0 on simple roots.

    The ordered product convention means the simple reflections are applied
    left to right, matching ``weyl_word_act_on_root_vector``.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    work = rank * rank * (1 << rank) * (rank + 1)
    # Every admitted finite Cartan entry has magnitude at most three. Bound
    # entries through the product and all characteristic coefficients before
    # any matrix or polynomial expansion.
    product_entry_bound = (3 * rank) ** rank
    coefficient_bound = (1 << rank) * factorial(rank) * product_entry_bound**rank
    output_bytes_bound = (rank + 1) * (len(str(coefficient_bound)) + 2)
    if (
        work > MAX_COXETER_POLYNOMIAL_WORK
        or output_bytes_bound > MAX_COXETER_POLYNOMIAL_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.coxeter_polynomial_bounds",
            message="Coxeter product and characteristic polynomial exceed the admitted work or output bound",
        )

    identity = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    coxeter = identity
    for index in range(rank):
        reflection = [list(row) for row in identity]
        reflection[index] = [
            int(index == column) - rows[index][column] for column in range(rank)
        ]
        coxeter = _integer_matrix_product(reflection, coxeter)

    # Subset dynamic programming computes the determinant over ZZ[t]. Each
    # state sums signed partial permutations with rows 0..k-1 assigned.
    states: dict[int, list[int]] = {0: [1]}
    for mask in range(1 << rank):
        polynomial = states.get(mask)
        if polynomial is None:
            continue
        row = mask.bit_count()
        if row == rank:
            continue
        for column in range(rank):
            bit = 1 << column
            if mask & bit:
                continue
            # Entry of tI-c, in ascending coefficient order.
            entry = [-coxeter[row][column]]
            if row == column:
                entry.append(1)
            sign = -1 if (mask >> (column + 1)).bit_count() % 2 else 1
            target = states.setdefault(mask | bit, [0] * (row + 2))
            for left_degree, left_coefficient in enumerate(polynomial):
                for right_degree, right_coefficient in enumerate(entry):
                    target[left_degree + right_degree] += (
                        sign * left_coefficient * right_coefficient
                    )
    ascending = states[(1 << rank) - 1]
    descending = tuple(reversed(ascending))
    return IntegerPolynomial(coefficients=descending)


def _integer_matrix_product(
    left: list[list[int]], right: tuple[tuple[int, ...], ...] | list[list[int]]
) -> tuple[tuple[int, ...], ...]:
    size = len(left)
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def cartan_datum(matrix: CartanMatrix) -> FiniteCartanDatum:
    """Construct root/coroot/weight basis data for a finite Cartan matrix."""
    cartan = _as_cartan(matrix)
    _admit_cartan_finite_type(cartan.entries)
    return _cartan_datum_from_admitted(cartan)


def _admit_lattice_coordinates(
    coordinates: tuple[int, ...], rank: int, *, output: bool = False
) -> None:
    max_bits = (
        MAX_LATTICE_OUTPUT_COORDINATE_BITS if output else MAX_LATTICE_COORDINATE_BITS
    )
    if (
        len(coordinates) != rank
        or any(type(value) is not int for value in coordinates)
        or any(abs(value).bit_length() > max_bits for value in coordinates)
    ):
        raise OperationResourceAdmissionError(
            location=("coordinates",),
            code="root_system.lattice_coordinates_over_envelope",
            message=f"lattice coordinates must match rank and fit within {max_bits} bits",
        )
    digit_bound = (max_bits * 30103) // 100_000 + 2
    output_bytes_bound = rank * (digit_bound + 3) + MAX_RANK**2 * 16 + 1024
    if output_bytes_bound > MAX_LATTICE_VECTOR_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("coordinates",),
            code="root_system.lattice_output_over_envelope",
            message="the exact lattice vector and retained datum exceed the output envelope",
        )


def _create_lattice_vector[LatticeVectorT: _FiniteCartanLatticeVector](
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...] | list[int],
    result_type: type[LatticeVectorT],
) -> LatticeVectorT:
    cartan = _as_cartan(matrix)
    _admit_cartan_finite_type(cartan.entries)
    coords = tuple(coordinates)
    _admit_lattice_coordinates(coords, len(cartan))
    datum = _cartan_datum_from_admitted(cartan)
    return result_type.model_construct(datum=datum, coordinates=coords)


def root_lattice_vector(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...] | list[int],
) -> RootLatticeVector:
    """Construct a root-lattice vector in simple-root coordinates."""
    return _create_lattice_vector(matrix, coordinates, RootLatticeVector)


def coroot_lattice_vector(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...] | list[int],
) -> CorootLatticeVector:
    """Construct a coroot-lattice vector in simple-coroot coordinates."""
    return _create_lattice_vector(matrix, coordinates, CorootLatticeVector)


def weight_lattice_vector(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...] | list[int],
) -> WeightLatticeVector:
    """Construct a weight-lattice vector in fundamental-weight coordinates."""
    return _create_lattice_vector(matrix, coordinates, WeightLatticeVector)


def coweight_lattice_vector(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...] | list[int],
) -> CoweightLatticeVector:
    """Construct a coweight-lattice vector in fundamental-coweight coordinates."""
    return _create_lattice_vector(matrix, coordinates, CoweightLatticeVector)


def _canonical_lattice_vector(
    vector: _FiniteCartanLatticeVector,
    expected_type: type[_FiniteCartanLatticeVector],
    *,
    output_bound: bool,
) -> tuple[FiniteCartanDatum, tuple[int, ...]]:
    if type(vector) is not expected_type:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.lattice_vector_type",
            message="the vector must have the exact lattice type required by this map",
        )
    try:
        datum = vector.datum
        cartan = _as_cartan(datum.cartan_matrix)
        coordinates = tuple(vector.coordinates)
    except (AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.invalid_lattice_vector",
            message="the lattice vector must retain a canonical finite Cartan datum",
        ) from error
    _admit_cartan_finite_type(cartan.entries)
    _admit_lattice_coordinates(coordinates, len(cartan), output=output_bound)
    canonical_datum = _cartan_datum_from_admitted(cartan)
    if datum != canonical_datum:
        raise OperationDomainValidationError(
            location=("vector", "datum"),
            code="root_system.lattice_vector_datum_mismatch",
            message="the supplied lattice maps and symmetrizer must match the canonical Cartan datum",
        )
    return canonical_datum, coordinates


def _lattice_inclusion[ResultVectorT: _FiniteCartanLatticeVector](
    vector: _FiniteCartanLatticeVector,
    expected_type: type[_FiniteCartanLatticeVector],
    result_type: type[ResultVectorT],
    *,
    transpose: bool,
) -> ResultVectorT:
    if type(vector) is not expected_type:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.lattice_vector_type",
            message="the inclusion requires a vector in its stated source lattice",
        )
    datum, coordinates = _canonical_lattice_vector(
        vector, expected_type, output_bound=False
    )
    rank = len(coordinates)
    matrix = datum.cartan_matrix.entries
    coefficient_bound = max(abs(value) for row in matrix for value in row)
    predicted_bits = (
        max((abs(value).bit_length() for value in coordinates), default=0)
        + (rank * coefficient_bound - 1).bit_length()
    )
    if predicted_bits > MAX_LATTICE_OUTPUT_COORDINATE_BITS:
        raise OperationResourceAdmissionError(
            location=("vector", "coordinates"),
            code="root_system.lattice_map_over_envelope",
            message="the exact basis-map image exceeds the admitted coordinate bound",
        )
    if rank * rank > MAX_RANK**2:
        raise OperationResourceAdmissionError(
            location=("vector", "datum"),
            code="root_system.lattice_map_work_over_envelope",
            message="the exact lattice basis map exceeds the admitted work bound",
        )
    image = tuple(
        sum(
            (matrix[column][row] if transpose else matrix[row][column])
            * coordinates[column]
            for column in range(rank)
        )
        for row in range(rank)
    )
    _admit_lattice_coordinates(image, rank, output=True)
    return result_type.model_construct(datum=datum, coordinates=image)


def root_to_weight_lattice(
    vector: RootLatticeVector,
) -> WeightLatticeVector:
    """Embed Q in P using the exact Cartan matrix in the datum's ordered bases."""
    return _lattice_inclusion(
        vector,
        RootLatticeVector,
        WeightLatticeVector,
        transpose=False,
    )


def coroot_to_coweight_lattice(
    vector: CorootLatticeVector,
) -> CoweightLatticeVector:
    """Embed Q^vee in P^vee using the transpose Cartan basis map."""
    return _lattice_inclusion(
        vector,
        CorootLatticeVector,
        CoweightLatticeVector,
        transpose=True,
    )


def dynkin_diagram(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> FiniteDynkinDiagram:
    """Return the exact directed/multiple-edge Dynkin graph of finite Cartan data."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    # The largest value has eight named nodes and 28 labeled edges. Reserve
    # complete result space before scanning entries or constructing the datum.
    work_bound = 8 * rank**3 + rank**2
    output_bound = MAX_DYNKIN_DIAGRAM_OUTPUT_CELLS
    if work_bound > MAX_DYNKIN_DIAGRAM_WORK or output_bound > (
        MAX_DYNKIN_DIAGRAM_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.dynkin_diagram_bounds",
            message="Dynkin diagram exceeds its admitted work or output envelope",
        )

    edges = tuple(
        DynkinEdge(
            simple_root_indices=(left, right),
            cartan_pairing=(rows[left][right], rows[right][left]),
            edge_multiplicity=rows[left][right] * rows[right][left],
        )
        for left in range(rank)
        for right in range(left + 1, rank)
        if rows[left][right] != 0 or rows[right][left] != 0
    )
    return FiniteDynkinDiagram._from_kernel(_cartan_datum_from_admitted(cartan), edges)


def _cartan_datum_from_admitted(cartan: CartanMatrix) -> FiniteCartanDatum:
    """Build basis maps after the caller has admitted finite Cartan data."""
    from fractions import Fraction
    from math import gcd, lcm

    from jacobian._exact import CanonicalRational
    from jacobian.math.matrices.values import IntegerMatrix

    rows = cartan.entries
    rank = len(rows)
    rational = positive_symmetrizer(rows)
    denominator: int = 1
    for value in rational:
        denominator = lcm(denominator, value.denominator)
    scaled: list[int] = [int(value * denominator) for value in rational]
    common: int = 0
    for scaled_value in scaled:
        common = gcd(common, abs(int(scaled_value)))
    normalized: list[int] = [scaled_value // common for scaled_value in scaled]
    # alpha_j = sum_i A[i,j] omega_i; coroot_j = sum_i A[j,i] omega_i^vee.
    root_to_weight = tuple(
        tuple(rows[row][column] for column in range(rank)) for row in range(rank)
    )
    coroot_to_coweight = tuple(
        tuple(rows[column][row] for column in range(rank)) for row in range(rank)
    )
    return FiniteCartanDatum._from_kernel(
        cartan_matrix=cartan,
        symmetrizer=tuple(
            CanonicalRational.from_fraction(Fraction(value, denominator // common))
            for value in normalized
        ),
        root_to_weight=IntegerMatrix(
            row_count=len(rows), column_count=len(rows), entries=root_to_weight
        ),
        coroot_to_coweight=IntegerMatrix(
            row_count=len(rows), column_count=len(rows), entries=coroot_to_coweight
        ),
    )


def _admit_root_poset_work(root_count: int, rank: int) -> None:
    """Admit all pair comparisons, reduction work, and emitted poset cells."""
    pair_comparisons = 2 * (root_count * (root_count - 1) // 2) * rank
    cover_work = root_count**3
    output_pairs = root_count * (root_count - 1) // 2
    output_cells = root_count * (rank + 4) + 3 * output_pairs
    if (
        not 1 <= root_count <= MAX_ROOT_POSET_ROOTS
        or pair_comparisons > MAX_ROOT_POSET_PAIR_COMPARISONS
        or cover_work > MAX_ROOT_POSET_COVER_WORK
        or output_pairs > MAX_ROOT_POSET_OUTPUT_PAIRS
        or output_cells > MAX_ROOT_POSET_OUTPUT_CELLS
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.root_poset_bounds",
            message=(
                "the positive-root family and its complete root poset must fit "
                f"the {MAX_ROOT_POSET_ROOTS}-root admission envelope"
            ),
        )


def root_poset(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> RootPosetResult:
    """Return the complete positive-root poset in its Cartan coordinate basis."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    roots = enumerate_positive_roots(rows)
    rank = len(rows)
    _admit_root_poset_work(len(roots), rank)

    elements = tuple(f"root_{index:02d}" for index in range(len(roots)))
    strict: set[tuple[str, str]] = set()
    incomparable_pairs: list[IncomparablePair] = []
    for left_index, left in enumerate(roots):
        for right_index in range(left_index + 1, len(roots)):
            right = roots[right_index]
            left_below = all(a <= b for a, b in zip(left, right, strict=True))
            right_below = all(a <= b for a, b in zip(right, left, strict=True))
            left_label = elements[left_index]
            right_label = elements[right_index]
            if left_below:
                strict.add((left_label, right_label))
            elif right_below:
                strict.add((right_label, left_label))
            else:
                incomparable_pairs.append(
                    IncomparablePair(left=left_label, right=right_label)
                )

    covers = _transitive_reduction(elements, strict)
    strict_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in sorted(strict)
    )
    cover_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in sorted(covers)
    )
    incomparable = tuple(incomparable_pairs)
    minimal = tuple(
        element
        for element in elements
        if not any(upper == element for _, upper in strict)
    )
    maximal = tuple(
        element
        for element in elements
        if not any(lower == element for lower, _ in strict)
    )
    ranks = canonical_poset_ranks(elements, covers)
    digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
    )
    poset = FinitePoset(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
        poset_digest=digest,
    )
    return RootPosetResult._from_kernel(
        _cartan_datum_from_admitted(cartan), roots, poset
    )


def _as_cartan(matrix: CartanMatrix | tuple[tuple[int, ...], ...]) -> CartanMatrix:
    """Establish a safe Cartan carrier before finite-type arithmetic.

    Native callers can pass model-constructed values that bypass Pydantic.
    Inspect the nested matrix and axis here so malformed claims become the
    root-system owner error rather than leaking AttributeError/ValidationError.
    """
    try:
        if isinstance(matrix, CartanMatrix):
            cartan = CartanMatrix.model_validate(matrix.model_dump(mode="python"))
        else:
            cartan = CartanMatrix.model_validate(matrix)
        nested = cartan.matrix
        rows = nested.entries
        rank = len(rows)
        axis = cartan.simple_root_axis
        if (
            type(nested.row_count) is not int
            or type(nested.column_count) is not int
            or nested.row_count != rank
            or nested.column_count != rank
        ):
            raise ValueError("nested integer-matrix dimensions are inconsistent")
        if not isinstance(rows, tuple) or not 1 <= rank <= MAX_RANK:
            raise ValueError("Cartan matrix rank is outside the admitted envelope")
        if not isinstance(axis, tuple) or axis != tuple(range(rank)):
            raise ValueError("Cartan matrix axis is not canonical")
        if any(
            not isinstance(row, tuple)
            or len(row) != rank
            or any(type(entry) is not int for entry in row)
            for row in rows
        ):
            raise ValueError(
                "Cartan matrix entries are not a rectangular integer matrix"
            )
        return cartan
    except (ValidationError, TypeError, ValueError, AttributeError, KeyError) as error:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.invalid_cartan_carrier",
            message="matrix must be a canonical bounded integer Cartan carrier",
        ) from error


def _admit_cartan_finite_type(matrix: tuple[tuple[int, ...], ...]) -> None:
    """Admit the finite-type Cartan domain before invoking a root kernel."""
    from jacobian.math.groups.root_systems._cartan import require_finite_type

    rank = len(matrix)
    if not 1 <= rank <= MAX_RANK:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.rank_out_of_range",
            message=f"rank must be between 1 and {MAX_RANK}",
        )
    if any(len(row) != rank for row in matrix):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.not_square",
            message="Cartan matrix must be square",
        )
    if any(matrix[index][index] != 2 for index in range(rank)):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.diagonal_entry",
            message="diagonal entries must be 2",
        )
    for row in range(rank):
        for column in range(rank):
            if row == column:
                continue
            entry = matrix[row][column]
            transpose_entry = matrix[column][row]
            if entry > 0:
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.positive_off_diagonal",
                    message="off-diagonal entries must be non-positive",
                )
            if entry * transpose_entry not in (0, 1, 2, 3):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.off_diagonal_product",
                    message="off-diagonal product must be 0, 1, 2, or 3",
                )
            if (entry == 0) != (transpose_entry == 0):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.zero_pattern",
                    message="generalized Cartan matrix requires a_ij == 0 iff a_ji == 0",
                )

    try:
        require_finite_type(matrix)
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.finite_type",
            message=str(error),
        ) from error


def cartan_matrix_from_type(
    cartan_type: str,
    rank: int,
) -> CartanTypeResult:
    """Build the Cartan matrix of a finite Dynkin type and rank.

    The admitted pairs are ``A_n`` (``n >= 1``), ``B_n``/``C_n``
    (``n >= 2``), ``D_n`` (``n >= 4``), ``E_6``/``E_7``/``E_8``,
    ``F_4``, and ``G_2`` with rank at most 8. The kernel asserts its
    own output is finite-type before trusted result construction.
    """
    if type(cartan_type) is not str or cartan_type not in VALID_CARTAN_TYPE_RANKS:
        raise OperationDomainValidationError(
            location=("cartan_type",),
            code="root_system.unknown_cartan_type",
            message="cartan_type must be one of A, B, C, D, E, F, G",
        )
    if type(rank) is not int or rank not in VALID_CARTAN_TYPE_RANKS[cartan_type]:
        raise OperationDomainValidationError(
            location=("rank",),
            code="root_system.invalid_cartan_type_rank",
            message=(
                f"rank {rank!r} is not a finite type for "
                f"{cartan_type}; admitted ranks are "
                f"{list(VALID_CARTAN_TYPE_RANKS[cartan_type])}"
            ),
        )
    rows = cartan_type_matrix(cartan_type, rank)
    _admit_cartan_finite_type(rows)
    cartan = CartanMatrix.model_validate(rows)
    return CartanTypeResult._from_kernel(cast("CartanType", cartan_type), rank, cartan)


def root_system_data(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> RootSystemDataResult:
    """Compute complete root-system data from a canonical Cartan matrix."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    n = len(rows)
    simple_roots = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    roots = enumerate_positive_roots(rows)
    components: list[RootComponentData] = []
    for indices in connected_components(rows):
        component_roots = tuple(
            root
            for root in roots
            if any(root[index] for index in indices)
            and all(root[index] == 0 for index in range(n) if index not in indices)
        )
        highest = max(component_roots, key=lambda root: sum(root))
        marks = tuple(highest[index] for index in indices)
        components.append(
            RootComponentData(
                simple_root_indices=indices,
                positive_roots=component_roots,
                highest_root=highest,
                marks=marks,
                coxeter_number=sum(marks) + 1,
            )
        )

    return RootSystemDataResult._from_kernel(
        cartan,
        positive_roots=roots,
        negative_roots=tuple(tuple(-value for value in root) for root in roots),
        simple_roots=simple_roots,
        components=tuple(components),
    )


def positive_roots(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> PositiveRootsResult:
    """Compute all positive roots of a root system from its Cartan matrix."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    all_positive = enumerate_positive_roots(rows)
    return PositiveRootsResult._from_kernel(cartan, all_positive)


def positive_root_profile(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> PositiveRootProfileResult:
    """Return exact heights, supports, and componentwise highest roots.

    Positive roots use the existing canonical lexicographic coordinate order.
    Each component's highest root is selected as the unique positive root that
    dominates every positive root of that component in simple-root order.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    # The root closure itself is bounded by the owner's complete-root limit.
    # This upper bound covers each candidate-vs-root coordinate comparison,
    # profile construction, all component index lists, and JSON framing.
    profile_work_bound = MAX_POSITIVE_ROOTS**2 * rank + MAX_POSITIVE_ROOTS * rank
    output_byte_bound = (
        4_096
        + MAX_POSITIVE_ROOTS * (128 + 24 * rank)
        + rank * (128 + 4 * MAX_POSITIVE_ROOTS + 4 * rank)
    )
    if profile_work_bound > MAX_ROOT_PROFILE_WORK or output_byte_bound > (
        MAX_ROOT_PROFILE_OUTPUT_CELLS
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.root_profile_bounds",
            message="positive-root profile exceeds its admitted work or output envelope",
        )

    roots = enumerate_positive_roots(rows)
    components = connected_components(rows)
    simple_index_to_component = {
        index: component_index
        for component_index, indices in enumerate(components)
        for index in indices
    }
    root_supports = tuple(
        tuple(index for index, coefficient in enumerate(root) if coefficient)
        for root in roots
    )
    root_component_indices: list[int] = []
    component_root_indices: list[list[int]] = [[] for _ in components]
    for root_index, support in enumerate(root_supports):
        component_ids = {simple_index_to_component[index] for index in support}
        if len(component_ids) != 1:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        component_index = next(iter(component_ids))
        root_component_indices.append(component_index)
        component_root_indices[component_index].append(root_index)

    component_profiles: list[PositiveRootComponentProfile] = []
    for component_index, indices in enumerate(components):
        root_indices = tuple(component_root_indices[component_index])
        candidates = tuple(
            candidate_index
            for candidate_index in root_indices
            if all(
                all(
                    roots[other_index][coordinate] <= roots[candidate_index][coordinate]
                    for coordinate in range(rank)
                )
                for other_index in root_indices
            )
        )
        if len(candidates) != 1:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        component_profiles.append(
            PositiveRootComponentProfile(
                simple_root_indices=indices,
                positive_root_indices=root_indices,
                highest_root_index=candidates[0],
            )
        )

    root_profiles = tuple(
        PositiveRootProfileEntry(
            root_coefficients=root,
            height=sum(root),
            support_simple_root_indices=support,
            component_index=root_component_indices[root_index],
        )
        for root_index, (root, support) in enumerate(
            zip(roots, root_supports, strict=True)
        )
    )
    return PositiveRootProfileResult(
        datum=_cartan_datum_from_admitted(cartan),
        positive_roots=root_profiles,
        components=tuple(component_profiles),
    )


def _admit_weyl_exponent_work(rows: tuple[tuple[int, ...], ...]) -> None:
    rank = len(rows)
    work_bound = MAX_POSITIVE_ROOTS * (rank + MAX_RANK * MAX_ROOT_COORDINATE)
    output_bytes_bound = 4_096 + rank * (128 + 12 * rank)
    if work_bound > MAX_ROOT_PROFILE_WORK or output_bytes_bound > (
        MAX_ROOT_PROFILE_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.weyl_exponents_bounds",
            message="Weyl exponents exceed the admitted work or output envelope",
        )


def _weyl_exponent_data(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Compute component axes and exponents after finite-type admission."""
    rank = len(rows)
    roots = enumerate_positive_roots(rows)
    components = []
    for indices in connected_components(rows):
        component_roots = tuple(
            root for root in roots if any(root[index] for index in indices)
        )
        if any(
            any(root[index] for index in range(rank) if index not in indices)
            for root in component_roots
        ):
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        max_height = max(sum(root) for root in component_roots)
        height_counts = [0] * (max_height + 2)
        for root in component_roots:
            height_counts[sum(root)] += 1
        exponent_multiplicities = tuple(
            height_counts[height] - height_counts[height + 1]
            for height in range(1, max_height + 1)
        )
        exponents = tuple(
            height
            for height, multiplicity in enumerate(exponent_multiplicities, start=1)
            for _ in range(multiplicity)
        )
        if len(exponents) != len(indices) or any(
            multiplicity < 0 for multiplicity in exponent_multiplicities
        ):
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        components.append((indices, exponents))
    return tuple(components)


def weyl_exponents(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylExponentsResult:
    """Return the Weyl exponents, retaining the factorization of the datum.

    For an irreducible finite crystallographic root system, the number of
    positive roots of height ``k`` equals the number of exponents at least
    ``k``. Successive differences of these height counts recover the exponent
    multiset without enumerating Weyl-group elements.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    _admit_weyl_exponent_work(rows)
    components = tuple(
        WeylExponentComponent(simple_root_indices=indices, exponents=exponents)
        for indices, exponents in _weyl_exponent_data(rows)
    )
    return WeylExponentsResult(
        datum=_cartan_datum_from_admitted(cartan), components=tuple(components)
    )


def weyl_poincare_polynomial(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylPoincarePolynomialResult:
    """Return ``sum_w q^length(w)`` for the finite Weyl group.

    The exact factorization ``product_i [m_i+1]_q`` is computed from the
    positive-root height distribution, so the operation never enumerates the
    group. Dense coefficients use the shared descending-degree ``ZZ[q]``
    polynomial carrier.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    _admit_weyl_exponent_work(rows)
    degree_bound = MAX_POSITIVE_ROOTS
    term_bound = degree_bound + 1
    work_bound = (degree_bound + 1) * (degree_bound + MAX_RANK)
    output_bytes_bound = 512 + term_bound * (len(str(MAX_WEYL_GROUP_ORDER)) + 4)
    if (
        degree_bound >= MAX_POLYNOMIAL_TERMS
        or work_bound > 50_000
        or output_bytes_bound > MAX_COXETER_POLYNOMIAL_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.weyl_poincare_bounds",
            message="Weyl Poincare polynomial exceeds the admitted work or output envelope",
        )

    component_data = _weyl_exponent_data(rows)
    exponents = tuple(
        exponent
        for _indices, component_exponents in component_data
        for exponent in component_exponents
    )
    polynomial = [1]
    for exponent in exponents:
        product = [0] * (len(polynomial) + exponent)
        for degree, coefficient in enumerate(polynomial):
            for shift in range(exponent + 1):
                product[degree + shift] += coefficient
        polynomial = product
    if (
        not exponents
        or len(polynomial) - 1 != sum(exponents)
        or any(
            coefficient <= 0 or coefficient > MAX_WEYL_GROUP_ORDER
            for coefficient in polynomial
        )
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return WeylPoincarePolynomialResult(
        matrix=cartan,
        polynomial=IntegerPolynomial(coefficients=tuple(reversed(polynomial))),
    )


def positive_coroots(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> PositiveCorootsResult:
    """Compute positive coroot coordinates and exact root lengths.

    Coroot coordinates use the simple-coroot basis matching the ordered
    simple-root basis of the source Cartan datum. If ``D A`` is the exact
    symmetrized Cartan matrix and ``c`` is a root coordinate vector, then
    ``(alpha, alpha) = c^T D A c`` and the coefficient of ``alpha_i^vee`` in
    ``alpha^vee`` is ``2 d_i c_i / (alpha, alpha)``.
    """
    datum = cartan_datum(_as_cartan(matrix))
    return _positive_coroots_from_admitted(datum)


def root_length_profile(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> RootLengthProfileResult:
    """Group positive roots by exact squared length within each factor.

    Each irreducible component uses the existing symmetrizer normalization
    whose first simple root has squared length 2. Length ratios are local to
    that component, since independent components have no canonical relative
    scale.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    # Bound the full root closure, quadratic-form evaluations, grouping, and
    # largest possible serialization before enumerating roots.
    work_bound = MAX_POSITIVE_ROOTS * (rank * rank + rank + 4)
    output_bound = (
        4_096
        + MAX_POSITIVE_ROOTS * (128 + 24 * rank)
        + rank * (128 + 4 * MAX_POSITIVE_ROOTS)
    )
    if work_bound > MAX_ROOT_LENGTH_PROFILE_WORK or output_bound > (
        MAX_ROOT_LENGTH_PROFILE_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.root_length_profile_bounds",
            message="root-length profile exceeds its admitted work or output envelope",
        )

    roots = enumerate_positive_roots(rows)
    symmetrizer = positive_symmetrizer(rows)
    bilinear = tuple(
        tuple(symmetrizer[i] * rows[i][j] for j in range(rank)) for i in range(rank)
    )
    squared_lengths = {
        root: sum(
            (
                Fraction(root[i]) * bilinear[i][j] * root[j]
                for i in range(rank)
                for j in range(rank)
            ),
            start=Fraction(0),
        )
        for root in roots
    }
    components: list[RootLengthComponentProfile] = []
    for simple_indices in connected_components(rows):
        component_roots = tuple(
            root for root in roots if any(root[index] for index in simple_indices)
        )
        lengths = sorted({squared_lengths[root] for root in component_roots})
        if not lengths or len(lengths) > 2:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        short_length = lengths[0]
        groups = tuple(
            RootLengthClass(
                squared_length=CanonicalRational.from_fraction(length),
                squared_length_ratio_to_short=CanonicalRational.from_fraction(
                    length / short_length
                ),
                roots=tuple(
                    root for root in component_roots if squared_lengths[root] == length
                ),
            )
            for length in lengths
        )
        components.append(
            RootLengthComponentProfile(
                simple_root_indices=simple_indices,
                length_classes=groups,
            )
        )
    return RootLengthProfileResult(
        datum=_cartan_datum_from_admitted(cartan),
        positive_roots=roots,
        components=tuple(components),
    )


def _positive_coroots_from_admitted(
    datum: FiniteCartanDatum,
) -> PositiveCorootsResult:
    """Build coroot data after finite-type admission and datum construction."""
    rows = datum.cartan_matrix.entries
    positive = enumerate_positive_roots(rows)
    symmetrizer = tuple(value.as_fraction() for value in datum.symmetrizer)
    bilinear = tuple(
        tuple(symmetrizer[row] * rows[row][column] for column in range(len(rows)))
        for row in range(len(rows))
    )
    pairs = [
        _positive_root_coroot_pair(root, symmetrizer, bilinear) for root in positive
    ]
    return PositiveCorootsResult._from_kernel(datum, tuple(pairs))


def _positive_root_coroot_pair(
    root: tuple[int, ...],
    symmetrizer: tuple[Fraction, ...],
    bilinear: tuple[tuple[Fraction, ...], ...],
) -> RootCorootPair:
    """Map one known positive root using the datum's exact bilinear form."""

    rank = len(root)
    squared_length = sum(
        (
            Fraction(root[i]) * bilinear[i][j] * root[j]
            for i in range(rank)
            for j in range(rank)
        ),
        start=Fraction(0),
    )
    if squared_length <= 0:
        raise RuntimeError("finite root has nonpositive squared length")
    coroot_coordinates = tuple(
        Fraction(2) * symmetrizer[index] * coefficient / squared_length
        for index, coefficient in enumerate(root)
    )
    if any(value.denominator != 1 or value < 0 for value in coroot_coordinates):
        raise RuntimeError("finite root produced nonintegral coroot coordinates")
    return RootCorootPair._from_kernel(
        root,
        tuple(int(value) for value in coroot_coordinates),
        CanonicalRational.from_fraction(squared_length),
    )


def root_to_coroot(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    root_coefficients: tuple[int, ...] | list[int],
) -> RootToCorootResult:
    """Convert one positive root to its exact coroot in the same datum."""

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    if (
        not isinstance(root_coefficients, (tuple, list))
        or len(root_coefficients) != rank
        or not any(root_coefficients)
        or any(
            type(coefficient) is not int
            or coefficient < 0
            or coefficient > MAX_ROOT_COORDINATE
            for coefficient in root_coefficients
        )
    ):
        raise OperationDomainValidationError(
            location=("root_coefficients",),
            code="root_system.root_coroot_input_shape",
            message="input must be a nonzero bounded positive-root vector on the Cartan axis",
        )
    root = tuple(root_coefficients)
    # Root closure has at most 120 candidates at the admitted rank. Bound the
    # scan and result framing before constructing that family.
    work_bound = MAX_POSITIVE_ROOTS * rank + rank * rank * rank
    output_bytes_bound = 256 + 64 * rank
    if (
        work_bound > MAX_ROOT_TO_COROOT_WORK
        or output_bytes_bound > MAX_ROOT_TO_COROOT_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.root_to_coroot_bounds",
            message="root-to-coroot conversion exceeds its admitted work or output bound",
        )
    positive = enumerate_positive_roots(rows)
    if root not in positive:
        raise OperationDomainValidationError(
            location=("root_coefficients",),
            code="root_system.root_not_positive_root",
            message="root coefficients must name a positive root of the supplied datum",
        )
    datum = _cartan_datum_from_admitted(cartan)
    symmetrizer = tuple(value.as_fraction() for value in datum.symmetrizer)
    bilinear = tuple(
        tuple(symmetrizer[row] * rows[row][column] for column in range(rank))
        for row in range(rank)
    )
    pair = _positive_root_coroot_pair(root, symmetrizer, bilinear)
    return RootToCorootResult._from_kernel(datum, pair)


def _apply_reflection(
    cartan: list[list[int]], vector: list[int], simple_idx: int
) -> list[int]:
    """Apply simple reflection s_i to a root lattice vector.

    For a vector v = sum v_j alpha_j, s_i(v) = v - (sum_j v_j A[i][j]) alpha_i.
    """
    n = len(cartan)
    inner = sum(vector[j] * cartan[simple_idx][j] for j in range(n))
    result = list(vector)
    result[simple_idx] -= inner
    return result


def _signed_roots(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    positive = enumerate_positive_roots(matrix)
    roots = tuple(
        sorted((*positive, *(tuple(-value for value in root) for root in positive)))
    )
    if len(roots) > MAX_SIGNED_ROOT_ACTION_DEGREE:
        raise ValueError("signed root action exceeds the bounded degree")
    return roots


def _weyl_group_order(matrix: tuple[tuple[int, ...], ...]) -> int:
    """Return |W| through its faithful action on the complete signed root set."""
    from sympy.combinatorics import Permutation, PermutationGroup

    roots = _signed_roots(matrix)
    root_index = {root: index for index, root in enumerate(roots)}
    generators = []
    for simple_index in range(len(matrix)):
        images = tuple(
            root_index[_simple_reflection_kernel(root, simple_index, matrix)]
            for root in roots
        )
        generators.append(Permutation(images))
    return int(PermutationGroup(*generators).order())


def _weyl_order_from_exponents(rows: tuple[tuple[int, ...], ...]) -> int:
    """Return the finite Weyl order from the bounded positive-root profile."""
    if not rows:
        return 1
    _admit_weyl_exponent_work(rows)
    return prod(
        exponent + 1
        for _indices, exponents in _weyl_exponent_data(rows)
        for exponent in exponents
    )


def simple_reflection(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    vector: tuple[int, ...],
    simple_index: int,
) -> SimpleReflectionResult:
    """Apply a simple reflection to a root lattice vector."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    if type(simple_index) is not int or simple_index < 0 or simple_index >= rank:
        raise OperationDomainValidationError(
            location=("simple_index",),
            code="root_system.simple_index_out_of_range",
            message="simple_index out of range",
        )
    if len(vector) != rank:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.vector_length_mismatch",
            message="vector length must match rank",
        )
    if any(
        type(coordinate) is not int or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
        for coordinate in vector
    ):
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.vector_coordinate_out_of_range",
            message="vector coordinates exceed the bounded root-lattice axis",
        )
    reflected = tuple(
        _apply_reflection(
            [list(row) for row in rows],
            list(vector),
            simple_index,
        )
    )
    if any(abs(coordinate) > MAX_REFLECTION_REPRESENTABLE for coordinate in reflected):
        raise OperationDomainValidationError(
            location=("reflected_vector",),
            code="root_system.reflected_vector_out_of_range",
            message="reflection image exceeds the interoperable root-lattice axis",
        )
    return SimpleReflectionResult._from_kernel(cartan, vector, simple_index, reflected)


def _admit_weyl_word(word: tuple[int, ...] | list[int], rank: int) -> tuple[int, ...]:
    """Admit a bounded Weyl word against a Cartan rank."""
    if isinstance(word, list):
        word = tuple(word)
    if (
        not isinstance(word, tuple)
        or len(word) > MAX_WEYL_WORD_LENGTH
        or any(type(index) is not int or index < 0 or index >= rank for index in word)
    ):
        raise OperationDomainValidationError(
            location=("word",),
            code="root_system.invalid_weyl_word",
            message=(
                "word must hold at most "
                f"{MAX_WEYL_WORD_LENGTH} integer simple-reflection indices "
                f"below the Cartan rank {rank}"
            ),
        )
    return word


def weyl_element_length(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylElementLengthResult:
    """Compute the length, reducedness, and inversion set of a Weyl word.

    The word lists simple-reflection indices applied left to right. The
    length is the number of positive roots the word sends negative; the
    word is reduced exactly when that count equals its factor count.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    word = _admit_weyl_word(word, len(rows))
    inversions = _weyl_word_inversions_kernel(rows, word)
    return WeylElementLengthResult._from_kernel(
        cartan, word, len(inversions), len(inversions) == len(word), inversions
    )


def weyl_element_order(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylElementOrderResult:
    """Return the exact order of a Weyl word via its faithful signed-root action."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    admitted_word = _admit_weyl_word(word, len(rows))
    rank = len(rows)
    signed_root_count = MAX_SIGNED_ROOT_ACTION_DEGREE
    work_bound = (
        MAX_POSITIVE_ROOTS * rank * rank
        + signed_root_count * len(admitted_word) * rank
        + signed_root_count * 4
    )
    output_bound = MAX_WEYL_ELEMENT_ORDER_OUTPUT_CELLS
    if work_bound > MAX_WEYL_ELEMENT_ORDER_WORK or output_bound > (
        MAX_WEYL_ELEMENT_ORDER_OUTPUT_CELLS
    ):
        raise OperationResourceAdmissionError(
            location=("word",),
            code="root_system.weyl_element_order_bounds",
            message="Weyl element order exceeds its admitted work or output envelope",
        )

    signed_roots = _signed_roots(rows)
    root_index = {root: index for index, root in enumerate(signed_roots)}
    images: list[int] = []
    for root in signed_roots:
        image = root
        for simple_index in admitted_word:
            image = _simple_reflection_kernel(image, simple_index, rows)
        image_index = root_index.get(image)
        if image_index is None:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        images.append(image_index)

    if len(set(images)) != len(images):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    # A Weyl transformation fixing every root fixes the simple-root basis,
    # hence the action on the complete signed-root set is faithful. Its order
    # is the lcm of the cycle lengths of this exact permutation.
    visited = bytearray(len(images))
    element_order = 1
    for start in range(len(images)):
        if visited[start]:
            continue
        length = 0
        index = start
        while not visited[index]:
            visited[index] = 1
            length += 1
            index = images[index]
        element_order = lcm(element_order, length)
    return WeylElementOrderResult._from_kernel(cartan, admitted_word, element_order)


def weyl_element_from_word(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylElement:
    """Construct the canonical root-lattice action represented by a word."""
    from jacobian.math.matrices.values import IntegerMatrix

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    admitted_word = _admit_weyl_word(word, len(rows))
    rank = len(rows)
    action = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    for index in admitted_word:
        reflection = _reflection_matrix(rows, index, transpose=False)
        action = _integer_matrix_product([list(row) for row in reflection], action)
    return WeylElement.model_construct(
        matrix=cartan,
        root_action=IntegerMatrix(row_count=rank, column_count=rank, entries=action),
    )


def _integer_inverse(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Invert an admitted unimodular integer matrix exactly."""
    from fractions import Fraction

    rank = len(matrix)
    work = [
        [Fraction(value) for value in row]
        + [Fraction(int(i == j)) for j in range(rank)]
        for i, row in enumerate(matrix)
    ]
    for column in range(rank):
        pivot = next((row for row in range(column, rank) if work[row][column]), None)
        if pivot is None:
            raise OperationDomainValidationError(
                location=("element", "root_action"),
                code="root_system.noninvertible_weyl_action",
                message="a Weyl action matrix must be invertible",
            )
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(rank):
            if row != column and work[row][column]:
                scale = work[row][column]
                work[row] = [
                    a - scale * b for a, b in zip(work[row], work[column], strict=True)
                ]
    inverse = tuple(tuple(value for value in row[rank:]) for row in work)
    if any(value.denominator != 1 for row in inverse for value in row):
        raise OperationDomainValidationError(
            location=("element", "root_action"),
            code="root_system.nonintegral_weyl_inverse",
            message="a Weyl action matrix must have an integral inverse",
        )
    return tuple(tuple(int(value) for value in row) for row in inverse)


def _admit_weyl_element_value(element: WeylElement) -> tuple[tuple[int, ...], ...]:
    """Check a caller-supplied action matrix belongs to this finite Weyl group."""
    from jacobian.math.groups.root_systems._cartan import positive_roots

    rows = element.matrix.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    action = element.root_action.entries
    if (
        element.root_action.row_count != rank
        or element.root_action.column_count != rank
        or any(abs(value) > MAX_ROOT_COORDINATE for row in action for value in row)
    ):
        raise OperationDomainValidationError(
            location=("element", "root_action"),
            code="root_system.invalid_weyl_action_shape",
            message="root action must be a bounded square integer matrix on the Cartan axis",
        )
    roots = positive_roots(rows)
    root_set = set(roots) | {tuple(-value for value in root) for root in roots}
    for root in roots:
        image = tuple(
            sum(action[i][j] * root[j] for j in range(rank)) for i in range(rank)
        )
        if image not in root_set:
            raise OperationDomainValidationError(
                location=("element", "root_action"),
                code="root_system.action_not_root_automorphism",
                message="root action must permute the root system",
            )
    # Root-system automorphisms include diagram symmetries. Repeated left
    # descents must reach identity to establish membership in the Weyl subgroup.
    reduced = action
    for _ in range(len(roots) + 1):
        if reduced == tuple(
            tuple(int(i == j) for j in range(rank)) for i in range(rank)
        ):
            return action
        inverse = _integer_inverse(reduced)
        descent = next(
            (
                index
                for index in range(rank)
                if all(
                    value <= 0 for value in (inverse[row][index] for row in range(rank))
                )
                and any(
                    value < 0 for value in (inverse[row][index] for row in range(rank))
                )
            ),
            None,
        )
        if descent is None:
            break
        reflection = _reflection_matrix(rows, descent, transpose=False)
        reduced = _integer_matrix_product([list(row) for row in reflection], reduced)
    raise OperationDomainValidationError(
        location=("element", "root_action"),
        code="root_system.action_not_weyl_element",
        message="root action is not generated by the simple reflections",
    )


def weyl_element_compose(first: WeylElement, then: WeylElement) -> WeylElement:
    """Compose elements in the declared order: apply ``first``, then ``then``."""
    from jacobian.math.matrices.values import IntegerMatrix

    if first.matrix != then.matrix:
        raise OperationDomainValidationError(
            location=("then", "matrix"),
            code="root_system.weyl_parent_mismatch",
            message="Weyl elements must use the same ordered Cartan parent",
        )
    left = _admit_weyl_element_value(first)
    right = _admit_weyl_element_value(then)
    rank = len(left)
    result = tuple(
        tuple(sum(right[i][k] * left[k][j] for k in range(rank)) for j in range(rank))
        for i in range(rank)
    )
    return WeylElement.model_construct(
        matrix=first.matrix,
        root_action=IntegerMatrix(row_count=rank, column_count=rank, entries=result),
    )


def weyl_element_inverse(element: WeylElement) -> WeylElement:
    """Return the inverse of an admitted finite Weyl element."""
    from jacobian.math.matrices.values import IntegerMatrix

    action = _admit_weyl_element_value(element)
    inverse = _integer_inverse(action)
    rank = len(action)
    return WeylElement.model_construct(
        matrix=element.matrix,
        root_action=IntegerMatrix(row_count=rank, column_count=rank, entries=inverse),
    )


def weyl_word_act_on_root_vector(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
    vector: tuple[int, ...] | list[int],
) -> WeylVectorActionResult:
    """Apply a bounded simple-reflection word to a root-lattice vector.

    Words are applied left to right, matching ``weyl_element_length``. In
    every finite crystallographic system admitted here, each simple root is
    sent to a root whose simple-root coefficients have absolute value at most
    six (the E8 maximum). Thus every prefix image has coordinates bounded by
    ``6 * rank * max(abs(vector))``. This conservative result bound is checked
    before applying any reflection.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    admitted_word = _admit_weyl_word(word, rank)
    if isinstance(vector, list):
        vector = tuple(vector)
    if (
        not isinstance(vector, tuple)
        or len(vector) != rank
        or any(
            type(coordinate) is not int
            or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
            for coordinate in vector
        )
    ):
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.invalid_root_vector",
            message=(
                "vector must have one representable integer coordinate per simple root"
            ),
        )
    if not admitted_word:
        return WeylVectorActionResult._from_kernel(
            cartan, admitted_word, vector, vector
        )
    max_coordinate = max(map(abs, vector), default=0)
    result_bound = MAX_ROOT_COORDINATE * rank * max_coordinate
    if result_bound > MAX_REFLECTION_REPRESENTABLE:
        raise OperationDomainValidationError(
            location=("vector",),
            code="root_system.weyl_vector_output_bound",
            message=(
                "the root-lattice image is not guaranteed to fit the "
                "interoperable coordinate bound"
            ),
        )
    image = list(vector)
    cartan_rows = [list(row) for row in rows]
    for simple_index in admitted_word:
        image = _apply_reflection(cartan_rows, image, simple_index)
    return WeylVectorActionResult._from_kernel(
        cartan, admitted_word, vector, tuple(image)
    )


def _weight_reflect(
    weight: tuple[int, ...], index: int, rows: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Apply s_i in fundamental-weight coordinates."""
    pairing = weight[index]
    return tuple(
        weight[target] - pairing * rows[target][index] for target in range(len(rows))
    )


def _fraction_inverse(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Invert a small nonsingular rational matrix by exact elimination."""
    size = len(matrix)
    work = [
        list(row) + [Fraction(int(i == j)) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            raise RuntimeError("finite weight Gram matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            scale = work[row][column]
            if scale:
                work[row] = [
                    left - scale * right
                    for left, right in zip(work[row], work[column], strict=True)
                ]
    return tuple(tuple(row[size:]) for row in work)


def _weight_coordinate_bounds(
    rows: tuple[tuple[int, ...], ...], weight: tuple[int, ...]
) -> tuple[int, ...]:
    """Bound every Weyl image coordinate using its invariant exact norm."""
    rank = len(rows)
    symmetrizer = positive_symmetrizer(rows)
    inverse_cartan = _fraction_inverse(
        tuple(tuple(Fraction(value) for value in row) for row in rows)
    )
    # In fundamental-weight coordinates G = A^{-T} D; this is the exact
    # invariant positive-definite inner product induced by the root form D A.
    gram = tuple(
        tuple(
            inverse_cartan[column][row] * symmetrizer[column] for column in range(rank)
        )
        for row in range(rank)
    )
    inverse_gram = _fraction_inverse(gram)
    norm_squared = sum(
        Fraction(weight[row]) * gram[row][column] * weight[column]
        for row in range(rank)
        for column in range(rank)
    )
    limit_squared = MAX_REFLECTION_REPRESENTABLE**2
    bounds: list[int] = []
    for index in range(rank):
        coordinate_bound_squared = norm_squared * inverse_gram[index][index]
        if coordinate_bound_squared > limit_squared:
            raise OperationDomainValidationError(
                location=("weight",),
                code="root_system.weight_orbit_coordinate_bound",
                message="some Weyl image coordinate may exceed the interoperable integer bound",
            )
        # floor(sqrt(p/q)) using integer arithmetic only.
        numerator = coordinate_bound_squared.numerator
        denominator = coordinate_bound_squared.denominator
        bound = isqrt(numerator // denominator)
        bounds.append(bound)
    return tuple(bounds)


def _dominant_weight(
    rows: tuple[tuple[int, ...], ...],
    weight: tuple[int, ...],
    *,
    word: list[int] | None = None,
    max_steps: int = MAX_WEIGHT_ORBIT_SIZE,
    bound_code: str = "root_system.weight_orbit_size_bound",
    bound_message: str = (
        f"the complete weight orbit exceeds {MAX_WEIGHT_ORBIT_SIZE} values"
    ),
) -> tuple[int, ...]:
    """Reflect into the dominant chamber, optionally recording its word."""
    dominant = weight
    normalization_steps = 0
    while True:
        negative = next(
            (index for index, value in enumerate(dominant) if value < 0), None
        )
        if negative is None:
            return dominant
        if normalization_steps >= max_steps:
            raise OperationDomainValidationError(
                location=("weight",),
                code=bound_code,
                message=bound_message,
            )
        dominant = _weight_reflect(dominant, negative, rows)
        if word is not None:
            word.append(negative)
        normalization_steps += 1


def weyl_dominant_representative(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    weight: tuple[int, ...] | list[int],
) -> WeylDominantRepresentativeResult:
    """Return the dominant representative and a Weyl element mapping to it."""
    from jacobian.math.matrices.values import IntegerMatrix

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    if isinstance(weight, list):
        weight = tuple(weight)
    if (
        not isinstance(weight, tuple)
        or len(weight) != rank
        or any(
            type(coordinate) is not int
            or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
            for coordinate in weight
        )
    ):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.invalid_integral_weight",
            message=(
                "weight must have one bounded integer fundamental-weight "
                "coordinate per simple coroot"
            ),
        )

    # Every prefix remains in the finite orbit. Admit its exact coordinate,
    # matrix-work, and output envelopes before performing any reflections.
    coordinate_bounds = _weight_coordinate_bounds(rows, weight)
    if any(bound > MAX_REFLECTION_REPRESENTABLE for bound in coordinate_bounds):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.dominant_representative_coordinate_bound",
            message="some Weyl image coordinate may exceed the interoperable integer bound",
        )
    work_bound = MAX_WEYL_WORD_LENGTH * rank**3
    output_bytes_bound = rank * rank * 16 + rank * 32 + 2048
    if work_bound > 1_000_000 or output_bytes_bound > 16_384:
        raise OperationResourceAdmissionError(
            location=("weight",),
            code="root_system.dominant_representative_bounds",
            message="the dominant representative and transporter exceed the admitted work or output envelope",
        )

    word: list[int] = []
    dominant = _dominant_weight(
        rows,
        weight,
        word=word,
        max_steps=MAX_WEYL_WORD_LENGTH,
        bound_code="root_system.dominant_representative_word_bound",
        bound_message=(
            "the dominant transporter exceeds the admitted "
            f"{MAX_WEYL_WORD_LENGTH}-reflection word bound"
        ),
    )
    root_action = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    for index in word:
        reflection = _reflection_matrix(rows, index, transpose=False)
        root_action = _integer_matrix_product(
            [list(row) for row in reflection], root_action
        )
    element = WeylElement.model_construct(
        matrix=cartan,
        root_action=IntegerMatrix(
            row_count=rank,
            column_count=rank,
            entries=root_action,
        ),
    )
    return WeylDominantRepresentativeResult._from_kernel(
        cartan, weight, dominant, element
    )


def weyl_antidominant_representative(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    weight: tuple[int, ...] | list[int],
) -> WeylAntidominantRepresentativeResult:
    """Return the unique antidominant orbit representative and transporter.

    For a weight not already in the negative chamber, first move it to the
    dominant chamber and then apply the longest Weyl element. This uses the
    finite-type chamber duality ``w0(C+) = C-`` and retains the resulting
    exact action on the root lattice as the transporter value.
    """
    from jacobian.math.matrices.values import IntegerMatrix

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    if isinstance(weight, list):
        weight = tuple(weight)
    if (
        not isinstance(weight, tuple)
        or len(weight) != rank
        or any(
            type(coordinate) is not int
            or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
            for coordinate in weight
        )
    ):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.invalid_integral_weight",
            message=(
                "weight must have one bounded integer fundamental-weight "
                "coordinate per simple coroot"
            ),
        )

    identity = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    # Both chambers include their shared walls. Preserve the identity map when
    # the source is already antidominant, including zero.
    if all(coordinate <= 0 for coordinate in weight):
        element = WeylElement.model_construct(
            matrix=cartan,
            root_action=IntegerMatrix(
                row_count=rank, column_count=rank, entries=identity
            ),
        )
        return WeylAntidominantRepresentativeResult._from_kernel(
            cartan, weight, weight, element
        )

    # Each performed reflection checks its actual resulting coordinates in
    # _weight_reflect. An orbit-wide norm estimate is only an upper bound and
    # can reject a representable chamber-normalization path.
    longest_word_work_bound = (
        (MAX_POSITIVE_ROOTS + 1) * rank * MAX_POSITIVE_ROOTS * rank
    )
    inversion_work_bound = MAX_POSITIVE_ROOTS * MAX_POSITIVE_ROOTS * rank
    action_work_bound = (MAX_WEYL_WORD_LENGTH + MAX_POSITIVE_ROOTS) * rank**2
    work_bound = (
        longest_word_work_bound
        + inversion_work_bound
        + action_work_bound
        + MAX_POSITIVE_ROOTS * rank**2
        + MAX_WEYL_WORD_LENGTH * rank
    )
    output_cells_bound = rank * rank + 2 * rank
    output_bytes_bound = rank * rank * 32 + rank * 64 + 2048
    if (
        work_bound > 2_000_000
        or output_cells_bound > 1_000
        or output_bytes_bound > 16_384
    ):
        raise OperationResourceAdmissionError(
            location=("weight",),
            code="root_system.antidominant_representative_bounds",
            message=(
                "the antidominant representative and transporter exceed the "
                "admitted work or output envelope"
            ),
        )

    dominant_word: list[int] = []
    dominant = _dominant_weight(
        rows,
        weight,
        word=dominant_word,
        max_steps=MAX_WEYL_WORD_LENGTH,
        bound_code="root_system.antidominant_representative_word_bound",
        bound_message=(
            "the antidominant transporter exceeds the admitted "
            f"{MAX_WEYL_WORD_LENGTH}-reflection chamber-normalization bound"
        ),
    )
    longest_word = _weyl_longest_word_kernel(rows)
    positive_root_count = len(enumerate_positive_roots(rows))
    if (
        len(longest_word) != positive_root_count
        or len(_weyl_word_inversions_kernel(rows, longest_word)) != positive_root_count
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    antidominant = dominant
    root_action = identity
    for index in dominant_word:
        root_action = _left_apply_root_reflection_to_action(root_action, index, rows)
    for index in longest_word:
        antidominant = _weight_reflect(antidominant, index, rows)
        root_action = _left_apply_root_reflection_to_action(root_action, index, rows)
    if any(value > 0 for value in antidominant):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    element = WeylElement.model_construct(
        matrix=cartan,
        root_action=IntegerMatrix(
            row_count=rank,
            column_count=rank,
            entries=tuple(tuple(row) for row in root_action),
        ),
    )
    return WeylAntidominantRepresentativeResult._from_kernel(
        cartan, weight, antidominant, element
    )


def _stabilizer_order(
    rows: tuple[tuple[int, ...], ...], dominant: tuple[int, ...]
) -> int:
    """Return the Weyl order of the zero-pairing parabolic subgroup."""
    indices = tuple(index for index, value in enumerate(dominant) if value == 0)
    if not indices:
        return 1
    parabolic = tuple(tuple(rows[row][column] for column in indices) for row in indices)
    return _weyl_group_order(parabolic)


def _enumerate_weight_orbit(
    rows: tuple[tuple[int, ...], ...],
    weight: tuple[int, ...],
    coordinate_bounds: tuple[int, ...],
    expected_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate all weight images after exact size and coordinate admission."""
    rank = len(rows)
    seen = {weight}
    pending = [weight]
    for current in pending:
        for index in range(rank):
            image = _weight_reflect(current, index, rows)
            if any(
                abs(value) > coordinate_bounds[coordinate_index]
                for coordinate_index, value in enumerate(image)
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            if image not in seen:
                seen.add(image)
                if len(seen) > expected_size:
                    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
                pending.append(image)
    if len(seen) != expected_size:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return tuple(sorted(seen))


def weyl_weight_orbit(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    weight: tuple[int, ...] | list[int],
) -> WeylWeightOrbitResult:
    """Return a complete finite Weyl orbit in fundamental-weight coordinates.

    The fundamental-weight coordinates are the pairings with the ordered
    simple coroots. Exact norm and orbit-stabilizer bounds are computed before
    orbit expansion; at most ``MAX_WEIGHT_ORBIT_SIZE`` values are admitted.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)
    if isinstance(weight, list):
        weight = tuple(weight)
    if (
        not isinstance(weight, tuple)
        or len(weight) != rank
        or any(
            type(coordinate) is not int
            or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
            for coordinate in weight
        )
    ):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.invalid_integral_weight",
            message="weight must have one bounded integer fundamental-weight coordinate per simple coroot",
        )

    # The norm gives a representation-safe coordinate bound for every orbit
    # element before any reflection or output construction.
    coordinate_bounds = _weight_coordinate_bounds(rows, weight)
    if any(bound > MAX_REFLECTION_REPRESENTABLE for bound in coordinate_bounds):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_coordinate_bound",
            message="some Weyl image coordinate may exceed the interoperable integer bound",
        )

    group_order = _weyl_group_order(rows)
    if not 1 <= group_order <= MAX_WEYL_GROUP_ORDER:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    # Move to the dominant chamber using strictly increasing pairing with
    # rho^vee. The finite 4096-step cap is admitted before this normalization;
    # exceeding it proves the orbit itself cannot fit the public orbit cap.
    dominant = _dominant_weight(rows, weight)

    # For dominant lambda, its stabilizer is the parabolic subgroup generated
    # by the zero simple-coroot pairings. Orbit-stabilizer gives the exact
    # cardinality before enumeration.
    stabilizer_order = _stabilizer_order(rows, dominant)
    if group_order % stabilizer_order:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    orbit_size = group_order // stabilizer_order
    if not 1 <= orbit_size <= MAX_WEIGHT_ORBIT_SIZE:
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_size_bound",
            message=f"the complete weight orbit has {orbit_size} values; maximum is {MAX_WEIGHT_ORBIT_SIZE}",
        )
    # Each value stores rank bounded safe integers, whose decimal form has a
    # fixed digit width, so the admitted output digit volume precedes the BFS.
    coordinate_digit_bound = 18
    max_output_digits = (orbit_size + 1) * rank * coordinate_digit_bound
    if max_output_digits > MAX_WEIGHT_ORBIT_OUTPUT_DIGITS:
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_output_bound",
            message="the complete weight orbit exceeds the admitted output size",
        )

    orbit = _enumerate_weight_orbit(rows, weight, coordinate_bounds, orbit_size)
    return WeylWeightOrbitResult._from_kernel(cartan, weight, orbit)


def weyl_parabolic_weight_orbit(
    request: WeylParabolicWeightOrbitRequest,
) -> WeylParabolicWeightOrbitResult:
    """Return the complete orbit of a typed weight under a standard parabolic.

    The subgroup order and the stabilizer order in its induced Cartan datum
    determine the exact orbit size before the ambient-coordinate BFS begins.
    """
    datum, weight = _canonical_lattice_vector(
        request.weight, WeightLatticeVector, output_bound=False
    )
    rows = datum.cartan_matrix.entries
    indices = request.simple_root_indices
    if tuple(sorted(set(indices))) != indices or any(
        index >= len(rows) for index in indices
    ):
        raise OperationDomainValidationError(
            location=("simple_root_indices",),
            code="root_system.parabolic_simple_indices",
            message="parabolic simple-root indices must be a strictly increasing subset of the Cartan axis",
        )

    # The full Weyl norm bounds every parabolic image coordinate and is
    # admitted before any subgroup or orbit expansion.
    coordinate_bounds = _weight_coordinate_bounds(rows, weight)
    if any(bound > MAX_REFLECTION_REPRESENTABLE for bound in coordinate_bounds):
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_coordinate_bound",
            message="some parabolic weight image coordinate may exceed the interoperable integer bound",
        )

    subgroup = tuple(tuple(rows[i][j] for j in indices) for i in indices)
    if subgroup:
        _admit_cartan_finite_type(subgroup)
        subgroup_order = _weyl_order_from_exponents(subgroup)
        restricted_dominant = _dominant_weight(
            subgroup, tuple(weight[index] for index in indices)
        )
        zero_indices = tuple(
            index for index, value in enumerate(restricted_dominant) if value == 0
        )
        stabilizer_matrix = tuple(
            tuple(subgroup[i][j] for j in zero_indices) for i in zero_indices
        )
        stabilizer_order = _weyl_order_from_exponents(stabilizer_matrix)
    else:
        subgroup_order = 1
        stabilizer_order = 1
    if (
        not 1 <= subgroup_order <= MAX_WEYL_GROUP_ORDER
        or subgroup_order % stabilizer_order
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    orbit_size = subgroup_order // stabilizer_order
    if not 1 <= orbit_size <= MAX_WEIGHT_ORBIT_SIZE:
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_size_bound",
            message=f"the complete parabolic weight orbit has {orbit_size} values; maximum is {MAX_WEIGHT_ORBIT_SIZE}",
        )
    max_output_digits = (orbit_size + 1) * len(rows) * 18
    if max_output_digits > MAX_WEIGHT_ORBIT_OUTPUT_DIGITS:
        raise OperationDomainValidationError(
            location=("weight",),
            code="root_system.weight_orbit_output_bound",
            message="the complete parabolic weight orbit exceeds the admitted output size",
        )

    seen = {weight}
    pending = [weight]
    for current in pending:
        for index in indices:
            image = _weight_reflect(current, index, rows)
            if any(
                abs(value) > coordinate_bounds[coordinate]
                for coordinate, value in enumerate(image)
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            if image not in seen:
                seen.add(image)
                if len(seen) > orbit_size:
                    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
                pending.append(image)
    if len(seen) != orbit_size:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return WeylParabolicWeightOrbitResult._from_kernel(
        datum, indices, weight, tuple(sorted(seen))
    )


def weyl_element_descents(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    word: tuple[int, ...] | list[int],
) -> WeylDescentsResult:
    """Compute the left and right descent sets of a Weyl-group word.

    A word lists its factors in application order, so prepending is
    right multiplication and appending is left multiplication: a right
    descent lowers the length when prepended, a left descent when
    appended. Both sets are carried as sorted simple-root index sets.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    admitted = _admit_weyl_word(word, len(rows))
    left_descents, right_descents = _weyl_word_descents_kernel(rows, admitted)
    return WeylDescentsResult._from_kernel(
        cartan, admitted, left_descents, right_descents
    )


def weyl_longest_element(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylLongestElementResult:
    """Compute a reduced word for the longest Weyl-group element.

    Greedy weak-order ascent appends one length-raising simple
    reflection at a time, so the word is reduced by construction. The
    kernel asserts maximality before trusted construction: the word's
    inversion count must equal the positive-root count.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    word = _weyl_longest_word_kernel(rows)
    num_positive_roots = len(enumerate_positive_roots(rows))
    inversions = _weyl_word_inversions_kernel(rows, word)
    if len(word) != len(inversions) or len(word) != num_positive_roots:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return WeylLongestElementResult._from_kernel(
        cartan, word, len(word), num_positive_roots
    )


def weyl_group_order(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> WeylGroupOrderResult:
    """Compute the exact order of a finite Weyl group without enumeration."""
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    return WeylGroupOrderResult._from_kernel(cartan, _weyl_group_order(rows))


def weyl_parabolic(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
    simple_root_indices: tuple[int, ...] | list[int],
) -> WeylParabolicResult:
    """Return standard-parabolic data embedded in the parent Cartan datum.

    The subgroup is generated by the selected simple reflections. Its order is
    the product of ``m + 1`` over the exponents of the induced principal
    Cartan submatrix, with the empty subset representing the trivial group.
    """
    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    if isinstance(simple_root_indices, list):
        simple_root_indices = tuple(simple_root_indices)
    rank = len(rows)
    if (
        not isinstance(simple_root_indices, tuple)
        or len(simple_root_indices) > rank
        or any(
            type(index) is not int or not 0 <= index < rank
            for index in simple_root_indices
        )
        or tuple(sorted(set(simple_root_indices))) != simple_root_indices
    ):
        raise OperationDomainValidationError(
            location=("simple_root_indices",),
            code="root_system.parabolic_simple_indices",
            message="simple_root_indices must be a strictly increasing subset of the Cartan axis",
        )

    parabolic = tuple(
        tuple(rows[row][column] for column in simple_root_indices)
        for row in simple_root_indices
    )
    output_bytes_bound = 256 + len(simple_root_indices) ** 2 * 12
    if output_bytes_bound > 4_096:
        raise OperationResourceAdmissionError(
            location=("simple_root_indices",),
            code="root_system.parabolic_output_bounds",
            message="standard parabolic data exceed the admitted output envelope",
        )
    if parabolic:
        _admit_cartan_finite_type(parabolic)
        group_order = _weyl_order_from_exponents(parabolic)
    else:
        group_order = 1
    if not 1 <= group_order <= MAX_WEYL_GROUP_ORDER:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return WeylParabolicResult(
        matrix=cartan,
        simple_root_indices=simple_root_indices,
        parabolic_cartan_matrix=parabolic,
        group_order=group_order,
    )


def _reflection_matrix(
    rows: tuple[tuple[int, ...], ...], index: int, *, transpose: bool
) -> tuple[tuple[int, ...], ...]:
    """Return the exact matrix of ``s_index`` on a root-like lattice.

    On the root lattice ``s_i(v) = v - <v, alpha_i^vee> alpha_i`` with
    ``<alpha_j, alpha_i^vee> = A[i][j]``; the coroot lattice uses the
    transposed pairing ``A[j][i]``.
    """
    rank = len(rows)
    matrix: list[list[int]] = [[int(k == j) for j in range(rank)] for k in range(rank)]
    for target in range(rank):
        for source in range(rank):
            pairing = rows[index][source] if not transpose else rows[source][index]
            matrix[target][source] -= int(target == index) * pairing
    return tuple(tuple(row) for row in matrix)


def _left_apply_root_reflection_to_action(
    action: tuple[tuple[int, ...], ...],
    index: int,
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Left-multiply a root action by ``s_index`` using its one changed row."""
    reflected = list(action)
    reflected[index] = tuple(
        action[index][column]
        - sum(
            rows[index][source] * action[source][column] for source in range(len(rows))
        )
        for column in range(len(rows))
    )
    return tuple(reflected)


def _weight_reflection_matrix(
    rows: tuple[tuple[int, ...], ...], index: int, *, transpose: bool
) -> tuple[tuple[int, ...], ...]:
    """Return the exact matrix of ``s_index`` on a weight-like lattice.

    For weights ``s_i(lambda) = lambda - lambda_i alpha_i`` with
    ``alpha_i = sum_k A[k][i] omega_k``; coweights use the transposed
    coefficients ``A[i][k]``.
    """
    rank = len(rows)
    matrix: list[list[int]] = [[int(k == j) for j in range(rank)] for k in range(rank)]
    for target in range(rank):
        coefficient = rows[target][index] if not transpose else rows[index][target]
        matrix[target][index] -= coefficient
    return tuple(tuple(row) for row in matrix)


def _square_is_identity(matrix: tuple[tuple[int, ...], ...]) -> bool:
    rank = len(matrix)
    return all(
        sum(matrix[row][k] * matrix[k][column] for k in range(rank))
        == int(row == column)
        for row in range(rank)
        for column in range(rank)
    )


def simple_reflections(
    matrix: CartanMatrix | tuple[tuple[int, ...], ...],
) -> SimpleReflectionsResult:
    """Compute every simple-reflection matrix on the four root-datum lattices."""
    from jacobian.math.matrices.values import IntegerMatrix

    cartan = _as_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)
    rank = len(rows)

    def _as_integer_matrix(
        entries: tuple[tuple[int, ...], ...],
    ) -> IntegerMatrix:
        return IntegerMatrix(
            row_count=rank,
            column_count=rank,
            entries=entries,
        )

    root = tuple(
        _as_integer_matrix(_reflection_matrix(rows, i, transpose=False))
        for i in range(rank)
    )
    coroot = tuple(
        _as_integer_matrix(_reflection_matrix(rows, i, transpose=True))
        for i in range(rank)
    )
    weight = tuple(
        _as_integer_matrix(_weight_reflection_matrix(rows, i, transpose=False))
        for i in range(rank)
    )
    coweight = tuple(
        _as_integer_matrix(_weight_reflection_matrix(rows, i, transpose=True))
        for i in range(rank)
    )
    for family in (root, coroot, weight, coweight):
        for reflection in family:
            if not _square_is_identity(reflection.entries):
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="root_system.reflection_not_involution",
                    message="simple reflections must square to the identity",
                )
    return SimpleReflectionsResult._from_kernel(
        cartan,
        root_matrices=root,
        coroot_matrices=coroot,
        weight_matrices=weight,
        coweight_matrices=coweight,
    )


__all__ = [
    "cartan_matrix_from_type",
    "coroot_lattice_vector",
    "coroot_to_coweight_lattice",
    "coweight_lattice_vector",
    "positive_coroots",
    "positive_root_profile",
    "positive_roots",
    "root_lattice_vector",
    "root_system_data",
    "root_to_weight_lattice",
    "simple_reflection",
    "simple_reflections",
    "weight_lattice_vector",
    "weyl_dominant_representative",
    "weyl_element_descents",
    "weyl_element_length",
    "weyl_group_order",
    "weyl_longest_element",
    "weyl_word_act_on_root_vector",
]
