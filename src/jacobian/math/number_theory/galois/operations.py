"""Domain functions for Galois theory operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._factor_process import factor_mod_prime

if TYPE_CHECKING:
    from sympy.combinatorics.perm_groups import PermutationGroup

from jacobian.math.number_theory.galois._models import (
    MAX_FACTOR_DEGREE,
    MAX_FIELD_ORDER,
    AutomorphismResult,
    FiniteFieldFactor,
    FinitePermutationGroup,
    FrobeniusCycleResult,
    GaloisFactorResult,
    GaloisGroupResult,
    GaloisRootAxis,
    QQFieldAutomorphism,
    QQRoot,
    QQSplittingField,
    SolvableResult,
    SplittingFieldResult,
    _require_prime,
    _supported_galois_polynomial,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _admit(operation: Callable[[], None], *, location: tuple[str | int, ...]) -> None:
    try:
        operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def _admit_factor(field_order: int, coefficients: tuple[int, ...]) -> None:
    if type(field_order) is not int or not 2 <= field_order <= MAX_FIELD_ORDER:
        raise PydanticCustomError(
            "galois_theory.field_order_bound",
            "field_order must be an integer in 2..251",
        )
    if not 2 <= len(coefficients) <= MAX_FACTOR_DEGREE + 1:
        raise PydanticCustomError(
            "galois_theory.degree_bound",
            "factorization admits degree one through 128",
        )
    if any(type(coefficient) is not int for coefficient in coefficients):
        raise PydanticCustomError(
            "galois_theory.coefficient_type",
            "coefficients must be strict integers",
        )
    _require_prime(field_order)
    if any(not 0 <= coefficient < field_order for coefficient in coefficients):
        raise PydanticCustomError(
            "galois_theory.coefficients_not_canonical",
            "coefficients must be canonical field residues",
        )
    if not coefficients or coefficients[-1] == 0:
        raise PydanticCustomError(
            "galois_theory.polynomial_zero",
            "factorization requires a nonzero polynomial with canonical degree",
        )


def _admit_frobenius(
    field_order: int,
    polynomial_degree: int,
    factorization_degrees: tuple[int, ...],
) -> None:
    if type(field_order) is not int or not 2 <= field_order <= MAX_FIELD_ORDER:
        raise PydanticCustomError(
            "galois_theory.field_order_bound",
            "field_order must be an integer in 2..251",
        )
    if (
        type(polynomial_degree) is not int
        or not 1 <= polynomial_degree <= MAX_FACTOR_DEGREE
        or not 1 <= len(factorization_degrees) <= MAX_FACTOR_DEGREE
        or any(
            type(degree) is not int or not 1 <= degree <= MAX_FACTOR_DEGREE
            for degree in factorization_degrees
        )
    ):
        raise PydanticCustomError(
            "galois_theory.degree_bound",
            "Frobenius partitions admit total degree one through 128",
        )
    from collections import Counter

    from sympy import divisors, mobius

    _require_prime(field_order)
    if sum(factorization_degrees) != polynomial_degree:
        raise PydanticCustomError(
            "galois_theory.partition_degree_mismatch",
            "factorization degrees must sum to polynomial degree",
        )
    for degree, count in Counter(factorization_degrees).items():
        available = (
            sum(
                int(mobius(divisor)) * field_order ** (degree // divisor)
                for divisor in divisors(degree)
            )
            // degree
        )
        if count > available:
            raise PydanticCustomError(
                "galois_theory.partition_unrealizable",
                "factorization pattern exceeds the available distinct "
                f"degree-{degree} irreducible factors over the field",
            )


def galois_factor(
    field_order: int, coefficients: tuple[int, ...]
) -> GaloisFactorResult:
    """Factor canonical ascending coefficients over GF(p)."""
    _admit(
        lambda: _admit_factor(field_order, coefficients),
        location=("field_order", "coefficients"),
    )
    unit, factor_polys = factor_mod_prime(field_order, coefficients)
    result_factors = tuple(
        FiniteFieldFactor(
            coefficients=factor_poly,
            multiplicity=int(multiplicity),
        )
        for factor_poly, multiplicity in factor_polys
    )
    return GaloisFactorResult._from_kernel(
        field_order=field_order,
        source_coefficients=coefficients,
        unit=int(unit) % field_order,
        factors=result_factors,
    )


def frobenius_cycle(
    field_order: int,
    polynomial_degree: int,
    factorization_degrees: tuple[int, ...],
) -> FrobeniusCycleResult:
    _admit(
        lambda: _admit_frobenius(field_order, polynomial_degree, factorization_degrees),
        location=("field_order", "factorization_degrees"),
    )
    cycle_type = tuple(sorted(factorization_degrees, reverse=True))
    return FrobeniusCycleResult(cycle_type=cycle_type)


def _galois_group_from_coeffs(coeffs: tuple[int, ...]) -> PermutationGroup:
    """Return the SymPy permutation group for a polynomial over Q.

    Coefficients are in ascending order: coeffs[0] is the constant term,
    coeffs[-1] is the leading coefficient.  SymPy's ``Poly`` expects
    descending order, so we reverse.
    """
    from sympy import Poly, Symbol, galois_group

    x = Symbol("x")
    # coefficients[0] = constant, coefficients[-1] = leading
    # Poly expects highest-degree first
    descending = list(reversed(coeffs))
    poly = Poly(descending, x, domain="QQ")
    perm_group, _alt = galois_group(poly)
    return perm_group


def _polynomial_from_coefficients(coefficients: tuple[int, ...]) -> RationalPolynomial:
    from jacobian.math.polynomials.values import RationalPolynomial

    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": value, "den": 1},
                        "exponents": [index],
                    }
                    for index, value in reversed(tuple(enumerate(coefficients)))
                    if value
                ]
            },
        }
    )


def _wire_group(
    perm_group: PermutationGroup,
    degree: int,
    polynomial: RationalPolynomial,
) -> FinitePermutationGroup:
    """Project a SymPy group onto the source polynomial's ordered root axis."""

    return FinitePermutationGroup(
        root_axis=GaloisRootAxis(
            polynomial=polynomial,
            indices=tuple(range(degree)),
        ),
        generators=tuple(
            tuple(int(generator(index)) for index in range(degree))
            for generator in perm_group.generators
        ),
    )


def galois_group(coefficients: tuple[int, ...]) -> GaloisGroupResult:
    """Compute the Galois group of a polynomial over Q."""
    _admit(
        lambda: _supported_galois_polynomial(coefficients),
        location=("coefficients",),
    )
    perm_group = _galois_group_from_coeffs(coefficients)
    source = _polynomial_from_coefficients(coefficients)
    group_name = str(perm_group)
    order = int(perm_group.order())
    is_solvable = bool(perm_group.is_solvable)

    return GaloisGroupResult._from_kernel(
        group=_wire_group(perm_group, len(coefficients) - 1, source),
        group_name=group_name,
        order=order,
        degree=len(coefficients) - 1,
        is_solvable=is_solvable,
    )


def solvable(coefficients: tuple[int, ...]) -> SolvableResult:
    """Determine if a polynomial is solvable by radicals.

    A polynomial is solvable by radicals iff its Galois group is solvable.
    This is computed from the actual Galois group, not from the degree alone.
    """
    _admit(
        lambda: _supported_galois_polynomial(coefficients),
        location=("coefficients",),
    )
    perm_group = _galois_group_from_coeffs(coefficients)
    source = _polynomial_from_coefficients(coefficients)
    is_solvable = bool(perm_group.is_solvable)
    return SolvableResult._from_kernel(
        solvable_by_radicals=is_solvable,
        group=_wire_group(perm_group, len(coefficients) - 1, source),
    )


def _canonical_splitting_field(
    field: QQSplittingField, *, location: tuple[str | int, ...]
) -> QQSplittingField:
    """Re-admit a possibly model-constructed field before any backend work."""

    if not isinstance(field, QQSplittingField):
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.splitting_field_type",
            message="field must be a QQ splitting-field value",
        )
    # These checks are deliberately strict for native callers: model_construct
    # must not turn a list, bool, or malformed axis into a trusted carrier.
    degree = getattr(field, "degree", None)
    basis_labels = getattr(field, "basis_labels", None)
    root_labels = getattr(field, "root_labels", None)
    if (
        type(degree) is not int
        or type(basis_labels) is not tuple
        or type(root_labels) is not tuple
        or any(type(label) is not str for label in basis_labels)
        or any(type(label) is not str for label in root_labels)
    ):
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.splitting_field_shape",
            message="splitting-field axes and degree must retain their canonical types",
        )
    try:
        # Re-run the structural model boundary for nested model_construct values.
        return QQSplittingField.model_validate(field.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.invalid_splitting_field",
            message="splitting field has malformed source or axes",
        ) from exc


def _canonical_automorphism(
    automorphism: QQFieldAutomorphism,
) -> tuple[QQFieldAutomorphism, QQSplittingField]:
    """Validate a root permutation before constructing a backend permutation."""

    if not isinstance(automorphism, QQFieldAutomorphism):
        raise OperationDomainValidationError(
            location=("automorphism",),
            code="galois_theory.automorphism_type",
            message="automorphism must be a QQ field automorphism value",
        )
    field = _canonical_splitting_field(
        cast(QQSplittingField, getattr(automorphism, "field", None)),
        location=("automorphism", "field"),
    )
    permutation = getattr(automorphism, "root_permutation", None)
    axis = tuple(range(len(field.root_labels)))
    if type(permutation) is not tuple or any(
        type(value) is not int for value in permutation
    ):
        raise OperationDomainValidationError(
            location=("automorphism", "root_permutation"),
            code="galois_theory.automorphism_axis",
            message="automorphism must carry a strict integer root permutation",
        )
    if len(permutation) != len(axis) or tuple(sorted(permutation)) != axis:
        raise OperationDomainValidationError(
            location=("automorphism", "root_permutation"),
            code="galois_theory.automorphism_axis",
            message="automorphism must carry a complete root permutation",
        )
    return (
        QQFieldAutomorphism(field=field, root_permutation=permutation),
        field,
    )


def _require_splitting_field(field: QQSplittingField) -> tuple[int, PermutationGroup]:
    """Re-establish the source/action relation at every public consumer."""

    field = _canonical_splitting_field(field, location=("field",))
    try:
        coefficients = _coefficients_from_polynomial(field.source)
        _supported_galois_polynomial(coefficients)
        group = _galois_group_from_coeffs(coefficients)
    except (
        ArithmeticError,
        IndexError,
        PydanticCustomError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        code = getattr(exc, "type", "galois_theory.invalid_splitting_field")
        message = exc.message() if isinstance(exc, PydanticCustomError) else str(exc)
        raise OperationDomainValidationError(
            location=("field",), code=code, message=message
        ) from exc
    source_degree = len(coefficients) - 1
    expected_degree = int(group.order())
    if (
        len(field.root_labels) != source_degree
        or len(field.basis_labels) != expected_degree
        or field.degree != expected_degree
    ):
        raise OperationDomainValidationError(
            location=("field",),
            code="galois_theory.splitting_field_axis_mismatch",
            message="field axes do not match the source polynomial and exact group degree",
        )
    return source_degree, group


def _require_automorphism(automorphism: QQFieldAutomorphism) -> PermutationGroup:
    canonical, field = _canonical_automorphism(automorphism)
    _degree, group = _require_splitting_field(field)
    from sympy.combinatorics import Permutation

    try:
        permutation = Permutation(canonical.root_permutation)
    except (TypeError, ValueError) as exc:
        # Keep malformed authored values in the owner domain even if a backend
        # changes its exception class or diagnostics.
        raise OperationDomainValidationError(
            location=("automorphism", "root_permutation"),
            code="galois_theory.automorphism_axis",
            message="automorphism must carry a complete root permutation",
        ) from exc
    if not group.contains(permutation):
        raise OperationDomainValidationError(
            location=("automorphism", "root_permutation"),
            code="galois_theory.automorphism_not_in_group",
            message="root permutation is not an automorphism of the source field",
        )
    return group


def splitting_field(coefficients: tuple[int, ...]) -> SplittingFieldResult:
    """Construct a bounded exact carrier for an irreducible QQ polynomial."""
    if type(coefficients) is not tuple or any(
        type(value) is not int for value in coefficients
    ):
        raise OperationDomainValidationError(
            location=("coefficients",),
            code="galois_theory.splitting_field_coefficients",
            message="splitting-field coefficients must be a tuple of integers",
        )
    try:
        _supported_galois_polynomial(coefficients)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("coefficients",), code=exc.type, message=exc.message()
        ) from exc
    group = _galois_group_from_coeffs(coefficients)
    degree = int(group.order())
    source = _polynomial_from_coefficients(coefficients)
    source_degree = len(coefficients) - 1
    field = QQSplittingField(
        source=source,
        basis_labels=tuple(f"b_{i}" for i in range(degree)),
        root_labels=tuple(f"root_{i}" for i in range(source_degree)),
        degree=degree,
    )
    roots = tuple(
        QQRoot(field=field, index=i, multiplicity=1) for i in range(source_degree)
    )
    return SplittingFieldResult(
        field=field,
        roots=roots,
        source_coefficients=coefficients,
        factor_reconstruction=coefficients,
    )


def automorphisms(field: QQSplittingField) -> AutomorphismResult:
    canonical_field = _canonical_splitting_field(field, location=("field",))
    _degree, group = _require_splitting_field(canonical_field)
    autos = tuple(
        QQFieldAutomorphism(
            field=canonical_field,
            root_permutation=tuple(
                int(g(i)) for i in range(len(canonical_field.root_labels))
            ),
        )
        for g in group.generators
    )
    identity = tuple(range(len(canonical_field.root_labels)))
    if not autos:
        autos = (QQFieldAutomorphism(field=canonical_field, root_permutation=identity),)
    return AutomorphismResult(field=canonical_field, automorphisms=autos)


def compose_automorphisms(
    first: QQFieldAutomorphism, second: QQFieldAutomorphism
) -> QQFieldAutomorphism:
    canonical_first, first_field = _canonical_automorphism(first)
    canonical_second, second_field = _canonical_automorphism(second)
    if first_field != second_field:
        raise OperationDomainValidationError(
            location=("second", "field"),
            code="galois_theory.parent_mismatch",
            message="automorphisms must share the exact splitting field",
        )
    _require_automorphism(canonical_first)
    _require_automorphism(canonical_second)
    composed = tuple(
        canonical_first.root_permutation[canonical_second.root_permutation[i]]
        for i in range(len(canonical_first.root_permutation))
    )
    # The result is checked again so this remains true for model-constructed
    # values and for any future change to the composition convention.
    result = QQFieldAutomorphism(field=first_field, root_permutation=composed)
    _require_automorphism(result)
    return result


def _canonical_root(root: QQRoot) -> tuple[QQRoot, QQSplittingField]:
    """Validate a root index before it can be used for Python indexing."""

    if not isinstance(root, QQRoot):
        raise OperationDomainValidationError(
            location=("root",),
            code="galois_theory.root_type",
            message="root must be a QQ splitting-field root value",
        )
    field = _canonical_splitting_field(
        cast(QQSplittingField, getattr(root, "field", None)),
        location=("root", "field"),
    )
    index = getattr(root, "index", None)
    multiplicity = getattr(root, "multiplicity", None)
    if type(index) is not int or type(multiplicity) is not int:
        raise OperationDomainValidationError(
            location=("root",),
            code="galois_theory.root_axis_mismatch",
            message="root index and multiplicity must be strict integers",
        )
    if multiplicity != 1 or not 0 <= index < len(field.root_labels):
        raise OperationDomainValidationError(
            location=("root",),
            code="galois_theory.root_axis_mismatch",
            message="root must be a simple root on the complete source axis",
        )
    return QQRoot(field=field, index=index, multiplicity=1), field


def apply_automorphism(automorphism: QQFieldAutomorphism, root: QQRoot) -> QQRoot:
    # Both carriers are admitted structurally before the source Galois group or
    # any root-axis indexing is touched.  This is the native analogue of wire
    # model validation for model_construct-authored values.
    canonical_root, root_field = _canonical_root(root)
    canonical_automorphism, automorphism_field = _canonical_automorphism(automorphism)
    if root_field != automorphism_field:
        raise OperationDomainValidationError(
            location=("root", "field"),
            code="galois_theory.parent_mismatch",
            message="root must belong to the automorphism field",
        )
    _require_automorphism(canonical_automorphism)
    return QQRoot(
        field=automorphism_field,
        index=canonical_automorphism.root_permutation[canonical_root.index],
        multiplicity=1,
    )


def _canonical_galois_group_claim(
    claim: GaloisGroupResult,
) -> GaloisGroupResult | None:
    """Re-admit the complete nested claim before reading its source axis."""

    if not isinstance(claim, GaloisGroupResult):
        return None
    try:
        return GaloisGroupResult.model_validate(claim.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError):
        return None


def _canonical_solvable_claim(claim: SolvableResult) -> SolvableResult | None:
    if not isinstance(claim, SolvableResult):
        return None
    try:
        return SolvableResult.model_validate(claim.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError):
        return None


def verify_galois_group(claim: GaloisGroupResult) -> bool:
    """Verify a serialized Galois-group claim against its source polynomial."""

    canonical_claim = _canonical_galois_group_claim(claim)
    if canonical_claim is None:
        return False
    claim = canonical_claim
    try:
        coefficients = _coefficients_from_polynomial(claim.polynomial)
        _supported_galois_polynomial(coefficients)
        perm_group = _galois_group_from_coeffs(coefficients)
        expected_group = _wire_group(
            perm_group, len(coefficients) - 1, claim.polynomial
        )
        return (
            claim.degree == len(coefficients) - 1
            and claim.order == int(perm_group.order())
            and claim.group_name == str(perm_group)
            and claim.is_solvable == bool(perm_group.is_solvable)
            and claim.group == expected_group
        )
    except (ArithmeticError, IndexError, PydanticCustomError, TypeError, ValueError):
        return False


def verify_solvable(claim: SolvableResult) -> bool:
    """Verify a serialized radical-solvability claim against its source."""

    canonical_claim = _canonical_solvable_claim(claim)
    if canonical_claim is None:
        return False
    claim = canonical_claim
    try:
        coefficients = _coefficients_from_polynomial(claim.polynomial)
        _supported_galois_polynomial(coefficients)
        perm_group = _galois_group_from_coeffs(coefficients)
        expected_group = _wire_group(
            perm_group, len(coefficients) - 1, claim.polynomial
        )
        return (
            claim.solvable_by_radicals == bool(perm_group.is_solvable)
            and claim.group == expected_group
        )
    except (ArithmeticError, IndexError, PydanticCustomError, TypeError, ValueError):
        return False


def _coefficients_from_polynomial(polynomial: RationalPolynomial) -> tuple[int, ...]:
    terms = polynomial.polynomial.terms
    degree = terms[0].exponents[0]
    coefficients = [0] * (degree + 1)
    for term in terms:
        if term.coefficient.den != 1:
            raise ValueError("Galois source coefficients must be integral")
        coefficients[term.exponents[0]] = term.coefficient.num
    return tuple(coefficients)


__all__ = [
    "frobenius_cycle",
    "galois_factor",
    "galois_group",
    "solvable",
    "verify_galois_group",
    "verify_solvable",
]
