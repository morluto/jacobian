"""Domain functions for cohomology operations over GF(2) and Z/p."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    HomologyConvention,
)
from jacobian.math.topology.cohomology.operations._models import (
    MAX_AMBIENT_SIMPLEX_VERTICES,
    MAX_COCHAIN_DEGREE,
    MAX_RESULT_COCHAIN_DEGREE,
    BocksteinResult,
    CohomologyRingResult,
    CupProductEntry,
    CupProductResult,
    InducedCohomologyMapResult,
    InducedCohomologyMatrix,
    SimplicialCochain,
    SimplicialMap,
    SteenrodSquareResult,
    _effective_ambient,
    _validate_simplex_entries,
    _validation_error,
)


def _run_admission(admission: Callable[[], None]) -> None:
    """Expose owner admission as a typed native-domain failure."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("request",), code=exc.type, message=exc.message()
        ) from exc


def _require_downward_closed(simplices: tuple[tuple[int, ...], ...]) -> None:
    """Require every codimension-one face of each ambient simplex."""

    known = set(simplices)
    for simplex in simplices:
        for index in range(len(simplex)):
            face = simplex[:index] + simplex[index + 1 :]
            if face and face not in known:
                raise _validation_error(
                    "ambient_not_downward_closed",
                    "ambient_simplices must be downward closed: the face "
                    f"{face} of {simplex} is absent",
                )


def _require_cocycle(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...],
) -> None:
    """Require the GF(2) coboundary to vanish on ambient simplices."""

    values_by_face: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        values_by_face[key] = (values_by_face.get(key, 0) + coefficient) % 2
    for sigma in ambient_simplices:
        if len(sigma) != cochain_degree + 2:
            continue
        coboundary = 0
        for index in range(len(sigma)):
            face = sigma[:index] + sigma[index + 1 :]
            coboundary = (coboundary + values_by_face.get(face, 0)) % 2
        if coboundary:
            raise _validation_error(
                "not_cocycle",
                "the supplied cochain is not a cocycle: its coboundary does not "
                "vanish on the ambient complex",
            )


def _validate_ambient_complex(
    support: tuple[tuple[int, ...], ...],
    ambient: tuple[tuple[int, ...], ...],
) -> None:
    """Verify coefficient-independent ambient shape, support, and closure."""

    _validate_simplex_entries(ambient, "ambient simplex")
    if any(len(simplex) > MAX_AMBIENT_SIMPLEX_VERTICES for simplex in ambient):
        raise _validation_error(
            "ambient_simplex_bound",
            "each ambient simplex may carry at most "
            f"{MAX_AMBIENT_SIMPLEX_VERTICES} vertices",
        )
    known = set(ambient)
    for simplex in support:
        if simplex not in known:
            raise _validation_error(
                "support_outside_ambient",
                "cochain support must lie inside the ambient complex",
            )
    _require_downward_closed(ambient)


def _is_zero_mod2_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
) -> bool:
    """Return whether the GF(2) cochain represented by sparse support is zero."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % 2
    return not any(value != 0 for value in merged.values())


def _is_zero_mod_prime_cochain(
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    prime: int,
) -> bool:
    """Return whether the Z/p cochain represented by sparse support is zero."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplex_values, simplex_coefficients, strict=True):
        key = tuple(simplex)
        merged[key] = (merged.get(key, 0) + coefficient) % prime
    return not any(value != 0 for value in merged.values())


def _admit_steenrod_square(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...],
    ambient_complex: FiniteSimplicialComplex | None,
) -> None:
    """Admit one exact Steenrod-square invocation and verify its cocycle."""

    def admission() -> None:
        result_degree = cochain_degree + square_degree
        if result_degree > MAX_RESULT_COCHAIN_DEGREE:
            raise _validation_error(
                "result_degree_bound",
                f"Sq^{square_degree} of a degree-{cochain_degree} "
                f"cochain returns degree {result_degree}, above the "
                f"{MAX_RESULT_COCHAIN_DEGREE}-degree exact-result budget",
            )
        if 0 < square_degree < cochain_degree:
            raise _validation_error(
                "intermediate_square_unsupported",
                "intermediate Steenrod squares 0<k<deg require cup-i products "
                "and are not supported",
            )
        effective_ambient = _effective_ambient(ambient_simplices, ambient_complex)
        if (
            not _is_zero_mod2_cochain(simplex_values, simplex_coefficients)
            and not effective_ambient
        ):
            raise _validation_error(
                "ambient_required_for_nonzero",
                "Steenrod squares are cohomology operations: the supplied cochain "
                "must be verified as a cocycle against an ambient simplicial "
                "complex; supply ambient_simplices or ambient_complex",
            )
        if (
            square_degree == cochain_degree
            and cochain_degree >= 1
            and not effective_ambient
        ):
            raise _validation_error(
                "ambient_required_for_top_square",
                "the top Steenrod square requires the ambient simplicial complex; "
                "supply ambient_simplices or ambient_complex",
            )
        if effective_ambient:
            _validate_ambient_complex(simplex_values, effective_ambient)
            _require_cocycle(
                cochain_degree,
                simplex_values,
                simplex_coefficients,
                effective_ambient,
            )

    _run_admission(admission)


def _admit_bockstein(
    prime: int,
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...],
    ambient_complex: FiniteSimplicialComplex | None,
) -> None:
    """Admit the supported zero-cocycle Bockstein branch."""

    def admission() -> None:
        from sympy import isprime

        if not isprime(prime):
            raise _validation_error("prime_not_prime", "prime must be a prime integer")
        if not _is_zero_mod_prime_cochain(
            simplex_values,
            simplex_coefficients,
            prime,
        ):
            raise _validation_error(
                "nonzero_bockstein_unsupported",
                "non-zero Bockstein requires the ambient simplicial complex; "
                "unsupported in this bounded operation",
            )
        effective_ambient = _effective_ambient(ambient_simplices, ambient_complex)
        if effective_ambient:
            # Zero modulo prime already establishes cocyclicity over Z/p.
            _validate_ambient_complex(simplex_values, effective_ambient)

    _run_admission(admission)


def _reduce_support(
    simplices: tuple[tuple[int, ...], ...],
    coeffs: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Merge duplicate simplex keys, summing coefficients in GF(2)."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplices, coeffs, strict=True):
        key = tuple(sorted(simplex))
        merged[key] = (merged.get(key, 0) + coefficient) % 2
    surviving = sorted(key for key, value in merged.items() if value != 0)
    values = tuple(surviving)
    coefficients = tuple(merged[key] for key in surviving)
    return values, coefficients


def _reduce_support_mod_prime(
    simplices: tuple[tuple[int, ...], ...],
    coeffs: tuple[int, ...],
    prime: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Merge duplicate simplex keys, summing coefficients modulo ``prime``."""

    merged: dict[tuple[int, ...], int] = {}
    for simplex, coefficient in zip(simplices, coeffs, strict=True):
        key = tuple(sorted(simplex))
        merged[key] = (merged.get(key, 0) + coefficient) % prime
    surviving = sorted(key for key, value in merged.items() if value != 0)
    values = tuple(surviving)
    coefficients = tuple(merged[key] for key in surviving)
    return values, coefficients


def _combined_face(
    front: tuple[int, ...],
    back: tuple[int, ...],
    left_degree: int,
    right_degree: int,
) -> tuple[int, ...] | None:
    """Return the simplex carrying ``front cup back`` via Alexander-Whitney.

    The faces must meet at exactly one shared vertex that closes the front
    face and opens the back face, and the sorted union must have the front
    ``left_degree + 1`` vertices followed by the back ``right_degree + 1``.
    """
    if front[-1] != back[0]:
        return None
    combined = tuple(sorted(set(front) | set(back)))
    if len(combined) != left_degree + right_degree + 1:
        return None
    if combined[: left_degree + 1] != front or combined[left_degree:] != back:
        return None
    return combined


def _cup_product(
    left_simplices: tuple[tuple[int, ...], ...],
    left_coeffs: tuple[int, ...],
    left_degree: int,
    right_simplices: tuple[tuple[int, ...], ...],
    right_coeffs: tuple[int, ...],
    right_degree: int,
    allowed_faces: frozenset[tuple[int, ...]] | None = None,
) -> tuple[list[tuple[int, ...]], list[int]]:
    """Compute the cup product of two cochains over GF(2) via Alexander-Whitney.

    For simplicial cochains, the cup product of alpha (degree p) and
    beta (degree q) on a (p+q)-simplex [v_0, ..., v_{p+q}] is:
    (alpha cup beta)([v_0, ..., v_{p+q}]) = alpha([v_0, ..., v_p]) * beta([v_p, ..., v_{p+q}])

    Over GF(2), all signs are 1. Only pairs where the front face of the
    combined simplex equals the left simplex and the back face equals the
    right simplex contribute. When ``allowed_faces`` is provided, target
    simplices outside the ambient complex are dropped: the cup product lives
    on the ambient complex's simplices only.
    """
    result_map: dict[tuple[int, ...], int] = {}
    for ls, lc in zip(left_simplices, left_coeffs, strict=False):
        lc_mod = lc % 2
        if lc_mod == 0:
            continue
        ls_sorted = tuple(sorted(ls))
        if len(ls_sorted) != left_degree + 1:
            continue
        for rs, rc in zip(right_simplices, right_coeffs, strict=False):
            rc_mod = rc % 2
            if rc_mod == 0:
                continue
            rs_sorted = tuple(sorted(rs))
            if len(rs_sorted) != right_degree + 1:
                continue
            combined = _combined_face(ls_sorted, rs_sorted, left_degree, right_degree)
            if combined is None:
                continue
            if allowed_faces is not None and combined not in allowed_faces:
                continue
            result_map[combined] = (result_map.get(combined, 0) + lc_mod * rc_mod) % 2

    # Filter zero results
    result_map = {k: v for k, v in result_map.items() if v % 2 != 0}
    simplices = sorted(result_map.keys())
    coeffs = [result_map[s] for s in simplices]
    return simplices, coeffs


def steenrod_square_fields(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...],
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[int, ...], bool]:
    """Pure Sq^k core returning ``(degree, values, coefficients, is_zero)``.

    Kept free of models so the explicit claim verifier can replay the exact
    computation without re-entering result validation.
    """
    p = cochain_degree
    k = square_degree

    if k > p:
        return (p + k, (), (), True)

    support_values, support_coeffs = _reduce_support(
        simplex_values, simplex_coefficients
    )

    if k == 0:
        is_zero = not support_coeffs
        return (p, support_values, support_coeffs, is_zero)

    if k == p:
        allowed = frozenset(ambient_simplices) if ambient_simplices else None
        simplices, coeffs = _cup_product(
            support_values,
            support_coeffs,
            p,
            support_values,
            support_coeffs,
            p,
            allowed_faces=allowed,
        )
        is_zero = len(coeffs) == 0 or all(c == 0 for c in coeffs)
        return (
            2 * p,
            tuple(simplices) if not is_zero else (),
            tuple(coeffs) if not is_zero else (),
            is_zero,
        )

    # Intermediate squares 0<k<p are not supported; request validation should have rejected.
    raise ValueError(
        "intermediate Steenrod squares 0<k<deg require cup-i products and are not supported"
    )


def steenrod_square(
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    square_degree: int,
    ambient_simplices: tuple[tuple[int, ...], ...] = (),
    ambient_complex: FiniteSimplicialComplex | None = None,
) -> SteenrodSquareResult:
    """Compute the Steenrod square Sq^k(x) for a cocycle x over GF(2).

    Sq^k(x) is nonzero only when 0 <= k <= deg(x).
    Sq^0(x) = x (identity)
    Sq^n(x) = x cup x when n = deg(x) (where x is a degree-n cocycle)
    Sq^k(x) = 0 when k > deg(x) (instability / cessation)

    For a cocycle x of degree n:
    - Sq^0(x) = x
    - Sq^n(x) = x cup x
    - Sq^k(x) = 0 for k > n (instability)
    - Sq^k(x) for 0 < k < n requires the cup-i product structure
    """
    _admit_steenrod_square(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        ambient_simplices,
        ambient_complex,
    )
    effective = _effective_ambient(ambient_simplices, ambient_complex)
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = steenrod_square_fields(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        effective,
    )
    return SteenrodSquareResult._from_kernel(
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        square_degree,
        ambient_simplices,
        ambient_complex,
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    )


def bockstein_fields(
    prime: int,
    cochain_degree: int,
    simplex_coefficients: tuple[int, ...],
    simplex_values: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[int, ...], bool]:
    """Pure Bockstein core returning ``(degree, values, coefficients, is_zero)``.

    When ``simplex_values`` is supplied duplicate simplex keys are merged
    modulo ``prime`` before the zero test, so a cochain whose sparse
    support cancels to zero is correctly classified as the zero cocycle.
    """

    if simplex_values is not None:
        # Merge duplicate keys modulo prime before the zero check.
        _, merged_coeffs = _reduce_support_mod_prime(
            simplex_values, simplex_coefficients, prime
        )
        # ``merged_coeffs`` is empty iff every residue is 0
        if not merged_coeffs:
            return (cochain_degree + 1, (), (), True)
        raise ValueError(
            "non-zero Bockstein requires the ambient simplicial complex and is not supported"
        )

    if not simplex_coefficients or all(c % prime == 0 for c in simplex_coefficients):
        return (cochain_degree + 1, (), (), True)
    raise ValueError(
        "non-zero Bockstein requires the ambient simplicial complex and is not supported"
    )


def bockstein(
    prime: int,
    cochain_degree: int,
    simplex_values: tuple[tuple[int, ...], ...],
    simplex_coefficients: tuple[int, ...],
    ambient_simplices: tuple[tuple[int, ...], ...] = (),
    ambient_complex: FiniteSimplicialComplex | None = None,
) -> BocksteinResult:
    """Compute the Bockstein homomorphism beta: H^n(Z/p) -> H^{n+1}(Z/p).

    For the short exact sequence 0 -> Z/p -> Z/p^2 -> Z/p -> 0,
    the Bockstein of a cocycle x is beta(x) = (1/p) * dx where dx is
    the coboundary of x modulo p. Computing it requires the ambient
    simplicial complex to evaluate the coboundary. This bounded operation
    only supports the trivial cocycle (all coefficients 0 mod p), for which
    the Bockstein is provably zero. Non-zero inputs are rejected at the
    request boundary as unsupported.
    """
    _admit_bockstein(
        prime,
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        ambient_simplices,
        ambient_complex,
    )
    (
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    ) = bockstein_fields(
        prime,
        cochain_degree,
        simplex_coefficients,
        simplex_values,
    )
    return BocksteinResult._from_kernel(
        prime,
        cochain_degree,
        simplex_values,
        simplex_coefficients,
        ambient_simplices,
        ambient_complex,
        result_degree,
        result_simplex_values,
        result_simplex_coefficients,
        is_zero,
    )


def verify_steenrod_square(claim: SteenrodSquareResult) -> bool:
    """Verify a serialized supported Steenrod-square claim."""

    try:
        return (
            steenrod_square(
                claim.cochain_degree,
                claim.simplex_values,
                claim.simplex_coefficients,
                claim.square_degree,
                claim.ambient_simplices,
                claim.ambient_complex,
            )
            == claim
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def _admit_simplicial_cochain(
    complex_: FiniteSimplicialComplex,
    prime: int,
    cochain: SimplicialCochain,
    location: str,
) -> None:
    """Require one cochain bound to its complex and prime field."""

    if cochain.complex != complex_ or cochain.prime != prime:
        raise OperationDomainValidationError(
            location=(location,),
            code="topology.simplicial_cochain_source_binding",
            message="the cochain must bind the source complex and prime",
        )


def _face_index(
    complex_: FiniteSimplicialComplex, degree: int
) -> dict[tuple[str, ...], int]:
    """Map canonical degree faces to cochain positions."""

    for entry in complex_.faces_by_dimension:
        if entry.dimension == degree:
            return {face: index for index, face in enumerate(entry.faces)}
    return {}


def _coboundary_matrix(
    complex_: FiniteSimplicialComplex, degree: int, *, prime: int
) -> list[list[int]]:
    """Return the dense coboundary delta^degree over GF(prime)."""

    from jacobian.math.topology.cohomology.operations._simplicial import (
        _dense_boundary,
        _transpose,
    )

    if degree >= complex_.dimension:
        faces = _face_index(complex_, degree)
        return [[0] * len(faces)]
    boundary = _dense_boundary(complex_, degree + 1, prime=prime)
    return _transpose(boundary)


def _is_cocycle(
    complex_: FiniteSimplicialComplex,
    prime: int,
    degree: int,
    coefficients: tuple[int, ...],
) -> bool:
    """Check the coboundary vanishes on one cochain vector."""

    from jacobian.math.topology.cohomology.operations._simplicial import (
        _mat_vec_mod,
    )

    matrix = _coboundary_matrix(complex_, degree, prime=prime)
    if not matrix:
        return True
    if len(matrix[0]) != len(coefficients):
        return False
    return all(value == 0 for value in _mat_vec_mod(matrix, coefficients, prime=prime))


def cup_product(
    complex_: FiniteSimplicialComplex,
    prime: int,
    left: SimplicialCochain,
    right: SimplicialCochain,
) -> CupProductResult:
    """Multiply two simplicial cochains by Alexander-Whitney over GF(prime).

    On an ordered simplex ``[v_0, ..., v_{p+q}]`` the product evaluates the
    left factor on the front face and the right factor on the back face;
    the formula carries no signs.  Degrees above the complex dimension
    yield the empty (zero) cochain of degree ``p + q``; a product degree
    beyond the representable ``MAX_COCHAIN_DEGREE`` envelope is rejected
    before any work.
    """

    from jacobian.math.topology.cohomology.operations._simplicial import (
        require_simplicial_cohomology_admission,
    )

    _admit_simplicial_cochain(complex_, prime, left, "left")
    _admit_simplicial_cochain(complex_, prime, right, "right")
    require_simplicial_cohomology_admission(
        complex_, prime, HomologyConvention.UNREDUCED
    )
    left_degree, right_degree = left.degree, right.degree
    total = left_degree + right_degree
    if total > MAX_COCHAIN_DEGREE:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="topology.cup_product_degree_bound",
            message="the product degree exceeds the representable cochain envelope",
        )
    if total > complex_.dimension:
        product = SimplicialCochain(
            complex=complex_, prime=prime, degree=total, coefficients=()
        )
        return CupProductResult._from_kernel(
            complex=complex_,
            prime=prime,
            left=left,
            right=right,
            product=product,
        )
    left_values = dict(
        zip(_face_index(complex_, left_degree), left.coefficients, strict=True)
    )
    right_values = dict(
        zip(_face_index(complex_, right_degree), right.coefficients, strict=True)
    )
    target_faces = next(
        entry.faces for entry in complex_.faces_by_dimension if entry.dimension == total
    )
    coefficients: list[int] = []
    for face in target_faces:
        front = face[: left_degree + 1]
        back = face[left_degree:]
        coefficients.append(
            (left_values.get(front, 0) * right_values.get(back, 0)) % prime
        )
    product = SimplicialCochain(
        complex=complex_, prime=prime, degree=total, coefficients=tuple(coefficients)
    )
    return CupProductResult._from_kernel(
        complex=complex_,
        prime=prime,
        left=left,
        right=right,
        product=product,
    )


def verify_cup_product(claim: CupProductResult) -> bool:
    """Verify a cup-product claim by recomputing it from its factors."""
    try:
        return cup_product(claim.complex, claim.prime, claim.left, claim.right) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def _quotient_coordinates(
    class_basis: tuple[tuple[int, ...], ...],
    coboundary_basis: tuple[tuple[int, ...], ...],
    target: tuple[int, ...],
    *,
    prime: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Express a cocycle in class plus coboundary coordinates over GF(prime).

    Solves the augmented system by exact reduced row-echelon form; raises
    RuntimeError when the target lies outside the cocycle span.
    """

    from jacobian.math.matrices.finite_fields import linear_algebra as prime_field
    from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

    columns = [*class_basis, *coboundary_basis]
    if not columns:
        if any(value % prime for value in target):
            raise RuntimeError("a nonzero cocycle needs basis coordinates")
        return (), ()
    rows = len(columns[0])
    augmented = tuple(
        (
            *[columns[column][row] % prime for column in range(len(columns))],
            target[row] % prime,
        )
        for row in range(rows)
    )
    reduced, pivots = prime_field.rref(
        PrimeFieldMatrix(prime=prime, entries=augmented, columns=len(columns) + 1)
    )
    pivot_set = set(pivots)
    if len(pivots) != len(columns):
        raise RuntimeError("class and coboundary bases must be independent")
    for row_index, row in enumerate(reduced):
        if row_index not in pivot_set and row[-1] % prime != 0:
            raise RuntimeError("a cocycle must lie in the class-coboundary span")
    solution = [0] * len(columns)
    for row_index, column in enumerate(pivots):
        if column < len(columns):
            solution[column] = reduced[row_index][-1] % prime
    class_count = len(class_basis)
    return (
        tuple(solution[:class_count]),
        tuple(solution[class_count:]),
    )


def verify_bockstein(claim: BocksteinResult) -> bool:
    """Verify the supported serialized Bockstein relation."""

    try:
        return (
            bockstein(
                claim.prime,
                claim.cochain_degree,
                claim.simplex_values,
                claim.simplex_coefficients,
                claim.ambient_simplices,
                claim.ambient_complex,
            )
            == claim
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def cohomology_ring(
    complex_: FiniteSimplicialComplex,
    prime: int,
    convention: HomologyConvention = HomologyConvention.UNREDUCED,
) -> CohomologyRingResult:
    """Compute the prime-field cohomology ring multiplication table.

    Every cohomology-basis pair within the complex dimension multiplies by
    Alexander-Whitney on cocycle representatives; the product is a cocycle
    by the Leibniz rule (asserted) and is expressed in class plus
    coboundary coordinates by exact row-echelon form, giving the full
    graded ring structure constants.
    """

    from jacobian.math.topology.cohomology.operations._simplicial import (
        require_simplicial_cohomology_admission,
        simplicial_cohomology,
    )

    require_simplicial_cohomology_admission(complex_, prime, convention)
    cohomology = simplicial_cohomology(complex_, prime, convention)
    groups = {group.dimension: group for group in cohomology.groups}
    class_vectors: dict[tuple[int, int], tuple[int, ...]] = {}
    for degree, group in groups.items():
        for index, vector in enumerate(group.cohomology_basis):
            class_vectors[(degree, index)] = vector.coefficients
    entries: list[CupProductEntry] = []
    for left_degree in sorted(groups):
        for left_index in range(groups[left_degree].betti_number):
            for right_degree in sorted(groups):
                total = left_degree + right_degree
                if total > complex_.dimension or total not in groups:
                    continue
                for right_index in range(groups[right_degree].betti_number):
                    product = cup_product(
                        complex_,
                        prime,
                        SimplicialCochain(
                            complex=complex_,
                            prime=prime,
                            degree=left_degree,
                            coefficients=class_vectors[(left_degree, left_index)],
                        ),
                        SimplicialCochain(
                            complex=complex_,
                            prime=prime,
                            degree=right_degree,
                            coefficients=class_vectors[(right_degree, right_index)],
                        ),
                    ).product
                    if not _is_cocycle(complex_, prime, total, product.coefficients):
                        raise RuntimeError("a product of cocycles must be a cocycle")
                    target = groups[total]
                    class_coords, coboundary_coords = _quotient_coordinates(
                        tuple(
                            vector.coefficients for vector in target.cohomology_basis
                        ),
                        tuple(
                            vector.coefficients for vector in target.coboundary_basis
                        ),
                        product.coefficients,
                        prime=prime,
                    )
                    entries.append(
                        CupProductEntry(
                            left_degree=left_degree,
                            left_index=left_index,
                            right_degree=right_degree,
                            right_index=right_index,
                            class_components=class_coords,
                            coboundary_components=coboundary_coords,
                        )
                    )
    return CohomologyRingResult._from_kernel(
        complex=complex_,
        prime=prime,
        convention=convention,
        cohomology=cohomology,
        products=tuple(entries),
    )


def verify_cohomology_ring(claim: CohomologyRingResult) -> bool:
    """Verify a cohomology-ring claim by recomputing its table."""
    try:
        return cohomology_ring(claim.complex, claim.prime, claim.convention) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def _permutation_sign(order: tuple[int, ...]) -> int:
    """Return the sign of the permutation taking a tuple to sorted order."""

    inversions = sum(
        1
        for first in range(len(order))
        for second in range(first + 1, len(order))
        if order[first] > order[second]
    )
    return -1 if inversions % 2 else 1


def _pullback_vector(
    source: FiniteSimplicialComplex,
    target: FiniteSimplicialComplex,
    vertex_map: tuple[str, ...],
    degree: int,
    coefficients: tuple[int, ...],
    *,
    prime: int,
) -> tuple[int, ...]:
    """Pull one target cochain back along a simplicial vertex map.

    A source face maps to the sorted image with the permutation sign, or
    to zero when the image is degenerate.
    """

    source_faces = _face_index(source, degree)
    target_faces = _face_index(target, degree)
    source_vertex_index = {
        vertex: index for index, vertex in enumerate(source.vertices)
    }
    target_values = dict(zip(target_faces, coefficients, strict=True))
    pulled: list[int] = []
    for face in source.faces_by_dimension:
        if face.dimension != degree:
            continue
        for simplex in face.faces:
            image = tuple(vertex_map[source_vertex_index[vertex]] for vertex in simplex)
            if len(set(image)) != len(image):
                pulled.append(0)
                continue
            sorted_image = tuple(sorted(image))
            image_position = {label: index for index, label in enumerate(sorted_image)}
            order = tuple(image_position[label] for label in image)
            pulled.append(
                (_permutation_sign(order) * target_values.get(sorted_image, 0)) % prime
            )
    if len(pulled) != len(source_faces):
        raise RuntimeError("pullback must cover every source face exactly once")
    return tuple(pulled)


def induced_cohomology_map(
    simplicial_map: SimplicialMap,
    prime: int,
    convention: HomologyConvention = HomologyConvention.UNREDUCED,
) -> InducedCohomologyMapResult:
    """Pull cohomology classes back along a simplicial map over GF(prime).

    Each target class basis cocycle pulls back to a source cocycle (the
    pullback commutes with the coboundary, asserted per class) and is
    expressed in source class coordinates by exact row-echelon form.  The
    per-degree matrices map target-class coordinates to source-class
    coordinates.
    """

    from jacobian.math.topology.cohomology.operations._simplicial import (
        require_simplicial_cohomology_admission,
        simplicial_cohomology,
    )

    require_simplicial_cohomology_admission(simplicial_map.source, prime, convention)
    require_simplicial_cohomology_admission(simplicial_map.target, prime, convention)
    source_cohomology = simplicial_cohomology(simplicial_map.source, prime, convention)
    target_cohomology = simplicial_cohomology(simplicial_map.target, prime, convention)
    source_groups = {group.dimension: group for group in source_cohomology.groups}
    target_groups = {group.dimension: group for group in target_cohomology.groups}
    matrices: list[InducedCohomologyMatrix] = []
    top = max(simplicial_map.source.dimension, simplicial_map.target.dimension)
    for degree in range(top + 1):
        source_group = source_groups.get(degree)
        target_group = target_groups.get(degree)
        source_basis = (
            tuple(vector.coefficients for vector in source_group.cohomology_basis)
            if source_group is not None
            else ()
        )
        source_coboundaries = (
            tuple(vector.coefficients for vector in source_group.coboundary_basis)
            if source_group is not None
            else ()
        )
        columns: list[tuple[int, ...]] = []
        if target_group is not None:
            for vector in target_group.cohomology_basis:
                pulled = _pullback_vector(
                    simplicial_map.source,
                    simplicial_map.target,
                    simplicial_map.vertex_map,
                    degree,
                    vector.coefficients,
                    prime=prime,
                )
                if source_group is not None and not _is_cocycle(
                    simplicial_map.source, prime, degree, pulled
                ):
                    raise RuntimeError("a pulled-back cocycle must be a cocycle")
                if not source_basis and not source_coboundaries:
                    # No source cochains at this degree: the pullback must be
                    # the zero cochain, hence has no class coordinates.
                    if any(value % prime for value in pulled):
                        raise RuntimeError(
                            "a pullback beyond the source dimension must vanish"
                        )
                    columns.append(())
                    continue
                class_coords, _ = _quotient_coordinates(
                    source_basis, source_coboundaries, pulled, prime=prime
                )
                columns.append(class_coords)
        rows: list[tuple[int, ...]] = []
        for row_index in range(len(source_basis)):
            rows.append(tuple(column[row_index] for column in columns))
        matrices.append(InducedCohomologyMatrix(degree=degree, rows=tuple(rows)))
    return InducedCohomologyMapResult._from_kernel(
        map=simplicial_map,
        prime=prime,
        convention=convention,
        source_cohomology=source_cohomology,
        target_cohomology=target_cohomology,
        matrices=tuple(matrices),
    )


def verify_induced_cohomology_map(claim: InducedCohomologyMapResult) -> bool:
    """Verify an induced-map claim by recomputing its matrices."""
    try:
        return induced_cohomology_map(claim.map, claim.prime, claim.convention) == claim
    except (OperationDomainValidationError, RuntimeError, TypeError, ValueError):
        return False


__all__ = [
    "bockstein",
    "bockstein_fields",
    "cohomology_ring",
    "cup_product",
    "induced_cohomology_map",
    "steenrod_square",
    "steenrod_square_fields",
    "verify_bockstein",
    "verify_cohomology_ring",
    "verify_cup_product",
    "verify_induced_cohomology_map",
    "verify_steenrod_square",
]
