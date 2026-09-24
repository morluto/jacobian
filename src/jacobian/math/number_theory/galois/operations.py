"""Domain functions for Galois theory operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import gcd, isqrt
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
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
    GaloisAutomorphismSubgroup,
    GaloisFactorResult,
    GaloisFixedFieldRequest,
    GaloisFixedFieldResult,
    GaloisGroupResult,
    GaloisRootAxis,
    GaloisSubgroupRequest,
    IntermediateFieldStabilizerRequest,
    IntermediateFieldStabilizerResult,
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    QQFieldAutomorphism,
    QQRoot,
    QQSplittingField,
    SolvableResult,
    SplittingFieldRequest,
    SplittingFieldResult,
    _require_prime,
    _supported_galois_polynomial,
)
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbedding,
    SimpleNumberFieldEmbeddingRequest,
    apply_simple_number_field_embedding,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _admit(operation: Callable[[], None], *, location: tuple[str | int, ...]) -> None:
    try:
        operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def polynomial_discriminant(
    polynomial: RationalPolynomial,
) -> PolynomialDiscriminantResult:
    """Compute a bounded exact integer polynomial discriminant over QQ."""
    try:
        canonical_polynomial = RationalPolynomial.model_validate(
            polynomial.model_dump()
        )
        request = PolynomialDiscriminantRequest(polynomial=canonical_polynomial)
    except ValidationError as exc:
        details = exc.errors(include_url=False, include_context=False)[0]
        raise OperationDomainValidationError(
            location=("polynomial",),
            code=str(details["type"]),
            message=str(details["msg"]),
        ) from exc
    coefficients = request.coefficients
    # A Sylvester determinant for f and f' has size at most 11. Hadamard's
    # bound with entry magnitude <= degree * 10^12 proves the discriminant
    # has fewer than 160 decimal digits throughout the admitted domain.
    from sympy import Poly, Symbol

    poly = Poly.from_list(list(reversed(coefficients)), Symbol("x"), domain="QQ")
    discriminant = int(poly.discriminant())
    from math import isqrt

    is_square = discriminant >= 0 and isqrt(discriminant) ** 2 == discriminant
    return PolynomialDiscriminantResult(
        polynomial=canonical_polynomial,
        discriminant=discriminant,
        is_rational_square=is_square,
    )


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


def _field_element(presentation, coefficients: tuple[Fraction, ...]):
    from jacobian.math.number_theory.number_fields.values import (
        SimpleNumberFieldElement,
    )

    if len(coefficients) != presentation.degree:
        raise ValueError("field coordinates must be reduced to the power basis")
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in coefficients
        ),
    )


def _coords(element) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in element.coefficients_ascending)


def _zero(presentation):
    return _field_element(presentation, (Fraction(0),) * presentation.degree)


def _one(presentation):
    return _field_element(
        presentation,
        (Fraction(1),) + (Fraction(0),) * (presentation.degree - 1),
    )


def _add_elements(left, right):
    if left.presentation != right.presentation:
        raise ValueError("number-field element parents must agree")
    return _field_element(
        left.presentation,
        tuple(a + b for a, b in zip(_coords(left), _coords(right), strict=True)),
    )


def _scale_element(element, scalar: Fraction):
    return _field_element(
        element.presentation, tuple(scalar * c for c in _coords(element))
    )


def _multiply_elements(left, right):
    presentation = left.presentation
    if right.presentation != presentation:
        raise ValueError("number-field element parents must agree")
    a = _coords(left)
    b = _coords(right)
    if presentation.degree == 1:
        return _field_element(presentation, (a[0] * b[0],))
    leading, linear, constant = map(Fraction, presentation.coefficients_descending)
    product = [Fraction(0)] * 3
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    # The quotient relation is leading*alpha^2 + linear*alpha + constant = 0.
    product[1] += product[2] * (-linear / leading)
    product[0] += product[2] * (-constant / leading)
    return _field_element(presentation, (product[0], product[1]))


def _source_coefficients(polynomial: RationalPolynomial) -> tuple[int, ...]:
    if len(polynomial.variables) != 1 or not polynomial.polynomial.terms:
        raise ValueError(
            "splitting-field source must be a nonzero univariate polynomial"
        )
    degree = polynomial.polynomial.terms[0].exponents[0]
    values = [0] * (degree + 1)
    for term in polynomial.polynomial.terms:
        if term.coefficient.den != 1:
            raise ValueError("splitting-field source coefficients must be integers")
        values[term.exponents[0]] = term.coefficient.num
    return tuple(values)


def _linear_factor_coefficients(field: QQSplittingField):
    presentation = field.extension
    coefficients = [_one(presentation)]
    for root, multiplicity in zip(
        field.root_values, field.root_multiplicities, strict=True
    ):
        for _ in range(multiplicity):
            updated = [_zero(presentation) for _ in range(len(coefficients) + 1)]
            for power, coefficient in enumerate(coefficients):
                updated[power] = _add_elements(
                    updated[power],
                    _multiply_elements(coefficient, _scale_element(root, Fraction(-1))),
                )
                updated[power + 1] = _add_elements(updated[power + 1], coefficient)
            coefficients = updated
    return tuple(coefficients)


def _construct_splitting_field(source: RationalPolynomial) -> SplittingFieldResult:
    from jacobian.math.number_theory.number_fields.values import (
        SimpleNumberFieldPresentation,
    )

    coefficients = _source_coefficients(source)
    degree = len(coefficients) - 1
    if degree not in (1, 2) or coefficients[-1] == 0:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="galois_theory.splitting_field_degree_bound",
            message="exact splitting fields currently admit degree one or two",
        )
    if any(type(value) is not int or abs(value) > 10**12 for value in coefficients):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="galois_theory.splitting_field_coefficient_bound",
            message="splitting-field coefficients must be integers of magnitude at most 10^12",
        )

    if degree == 1:
        extension = SimpleNumberFieldPresentation(coefficients_descending=(1, 0))
        root = _field_element(extension, (Fraction(-coefficients[0], coefficients[1]),))
        root_values = (root,)
        multiplicities = (1,)
    else:
        c, b, a = coefficients
        discriminant = b * b - 4 * a * c
        square_root = isqrt(discriminant) if discriminant >= 0 else -1
        if discriminant >= 0 and square_root * square_root == discriminant:
            extension = SimpleNumberFieldPresentation(coefficients_descending=(1, 0))
            roots = sorted(
                (
                    Fraction(-b - square_root, 2 * a),
                    Fraction(-b + square_root, 2 * a),
                )
            )
            root_values = tuple(
                _field_element(extension, (root,)) for root in sorted(set(roots))
            )
            multiplicities = (2,) if square_root == 0 else (1, 1)
        else:
            content = gcd(gcd(abs(a), abs(b)), abs(c))
            normalized = (a // content, b // content, c // content)
            if normalized[0] < 0:
                normalized = tuple(-value for value in normalized)
            extension = SimpleNumberFieldPresentation(
                coefficients_descending=normalized
            )
            alpha = _field_element(extension, (Fraction(0), Fraction(1)))
            conjugate = _field_element(
                extension,
                (Fraction(-b, a), Fraction(-1)),
            )
            root_values = (alpha, conjugate)
            multiplicities = (1, 1)

    field = QQSplittingField(
        source=source,
        extension=extension,
        root_values=root_values,
        root_multiplicities=multiplicities,
    )
    factor_reconstruction = _linear_factor_coefficients(field)
    reconstructed_source = tuple(
        _scale_element(value, Fraction(coefficients[-1]))
        for value in factor_reconstruction
    )
    expected_source = tuple(
        _field_element(
            extension, (Fraction(value),) + (Fraction(0),) * (extension.degree - 1)
        )
        for value in coefficients
    )
    if reconstructed_source != expected_source:
        raise ArithmeticError(
            "exact root factors do not reconstruct the source polynomial"
        )
    roots = tuple(
        QQRoot(
            field=field,
            index=index,
            multiplicity=multiplicity,
            value=value,
        )
        for index, (value, multiplicity) in enumerate(
            zip(root_values, multiplicities, strict=True)
        )
    )
    return SplittingFieldResult(
        field=field,
        roots=roots,
        source_coefficients=coefficients,
        factor_reconstruction=factor_reconstruction,
    )


def _canonical_splitting_field(
    field: QQSplittingField, *, location: tuple[str | int, ...]
) -> QQSplittingField:
    if not isinstance(field, QQSplittingField):
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.splitting_field_type",
            message="field must be an exact bounded QQ splitting-field value",
        )
    try:
        canonical = QQSplittingField.model_validate(field.model_dump())
        expected = _construct_splitting_field(canonical.source).field
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.invalid_splitting_field",
            message="splitting field has malformed source or exact root coordinates",
        ) from exc
    if canonical != expected:
        raise OperationDomainValidationError(
            location=location,
            code="galois_theory.splitting_field_claim_mismatch",
            message="splitting-field extension, roots, or multiplicities do not match the exact source construction",
        )
    return canonical


def _canonical_automorphism(
    automorphism: QQFieldAutomorphism,
) -> tuple[QQFieldAutomorphism, QQSplittingField]:
    if not isinstance(automorphism, QQFieldAutomorphism):
        raise OperationDomainValidationError(
            location=("automorphism",),
            code="galois_theory.automorphism_type",
            message="automorphism must be an exact QQ field map",
        )
    field = _canonical_splitting_field(
        cast(QQSplittingField, getattr(automorphism, "field", None)),
        location=("automorphism", "field"),
    )
    try:
        canonical = QQFieldAutomorphism.model_validate(
            {**automorphism.model_dump(), "field": field.model_dump()}
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("automorphism",),
            code="galois_theory.invalid_automorphism",
            message="automorphism has malformed basis images or root permutation",
        ) from exc
    _require_automorphism(canonical, field)
    return canonical, field


def _map_element(automorphism: QQFieldAutomorphism, element):
    presentation = automorphism.field.extension
    coords = _coords(element)
    value = _zero(presentation)
    power = _one(presentation)
    generator_image = automorphism.basis_images[1] if presentation.degree == 2 else None
    for coefficient in coords:
        value = _add_elements(value, _scale_element(power, coefficient))
        if generator_image is not None:
            power = _multiply_elements(power, generator_image)
    return value


def _automorphism_for_generator_image(
    field: QQSplittingField, generator_image
) -> QQFieldAutomorphism:
    presentation = field.extension
    basis_images = (
        (_one(presentation),)
        if presentation.degree == 1
        else (_one(presentation), generator_image)
    )
    permutation = []
    for root in field.root_values:
        image = _map_element(
            QQFieldAutomorphism.model_construct(
                field=field,
                root_permutation=tuple(range(len(field.root_values))),
                basis_images=basis_images,
            ),
            root,
        )
        try:
            permutation.append(field.root_values.index(image))
        except ValueError as exc:
            raise ArithmeticError(
                "field map does not preserve the exact root family"
            ) from exc
    return QQFieldAutomorphism(
        field=field,
        root_permutation=tuple(permutation),
        basis_images=basis_images,
    )


def _require_automorphism(
    automorphism: QQFieldAutomorphism, field: QQSplittingField
) -> None:
    presentation = field.extension
    if automorphism.basis_images[0] != _one(presentation):
        raise OperationDomainValidationError(
            location=("automorphism", "basis_images", 0),
            code="galois_theory.automorphism_not_unital",
            message="a QQ-field automorphism must fix one",
        )
    if presentation.degree == 2:
        image = automorphism.basis_images[1]
        a, b, c = map(Fraction, presentation.coefficients_descending)
        relation = _add_elements(
            _add_elements(
                _scale_element(_multiply_elements(image, image), a),
                _scale_element(image, b),
            ),
            _field_element(presentation, (c, Fraction(0))),
        )
        if relation != _zero(presentation):
            raise OperationDomainValidationError(
                location=("automorphism", "basis_images", 1),
                code="galois_theory.automorphism_not_homomorphism",
                message="the generator image must satisfy the defining field relation",
            )
        if image not in field.root_values:
            raise OperationDomainValidationError(
                location=("automorphism", "basis_images", 1),
                code="galois_theory.automorphism_not_surjective",
                message="the generator image must be a root in the exact source root family",
            )
    for index, root in enumerate(field.root_values):
        image = _map_element(automorphism, root)
        expected_index = automorphism.root_permutation[index]
        if image != field.root_values[expected_index]:
            raise OperationDomainValidationError(
                location=("automorphism", "root_permutation", index),
                code="galois_theory.automorphism_action_mismatch",
                message="the root permutation must agree with the exact field map",
            )


def splitting_field(request: SplittingFieldRequest) -> SplittingFieldResult:
    """Build the exact splitting field for every rational polynomial of degree <=2."""
    try:
        canonical_request = SplittingFieldRequest.model_validate(request.model_dump())
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        details = (
            exc.errors(include_url=False, include_context=False)[0]
            if isinstance(exc, ValidationError)
            else None
        )
        raise OperationDomainValidationError(
            location=("polynomial",),
            code=str(details["type"])
            if details
            else "galois_theory.invalid_splitting_field_request",
            message=str(details["msg"])
            if details
            else "invalid splitting-field request",
        ) from exc
    return _construct_splitting_field(canonical_request.polynomial)


def automorphisms(field: QQSplittingField) -> AutomorphismResult:
    canonical_field = _canonical_splitting_field(field, location=("field",))
    presentation = canonical_field.extension
    candidates = (
        [_automorphism_for_generator_image(canonical_field, _one(presentation))]
        if presentation.degree == 1
        else [
            _automorphism_for_generator_image(
                canonical_field,
                _field_element(presentation, (Fraction(0), Fraction(1))),
            ),
            *(
                _automorphism_for_generator_image(canonical_field, root)
                for root in canonical_field.root_values
                if root != _field_element(presentation, (Fraction(0), Fraction(1)))
            ),
        ]
    )
    # On a degree-two field the generator's conjugate is the only other image;
    # rational-root polynomials have the degree-one QQ presentation.
    unique = {auto.root_permutation: auto for auto in candidates}
    return AutomorphismResult(
        field=canonical_field,
        automorphisms=tuple(unique[key] for key in sorted(unique)),
    )


def compose_automorphisms(
    first: QQFieldAutomorphism, second: QQFieldAutomorphism
) -> QQFieldAutomorphism:
    """Return ``first ∘ second`` on the exact field and root axis."""
    canonical_first, first_field = _canonical_automorphism(first)
    canonical_second, second_field = _canonical_automorphism(second)
    if first_field != second_field:
        raise OperationDomainValidationError(
            location=("second", "field"),
            code="galois_theory.parent_mismatch",
            message="automorphisms must share the exact splitting field",
        )
    images = tuple(
        _map_element(canonical_first, image) for image in canonical_second.basis_images
    )
    provisional = QQFieldAutomorphism.model_construct(
        field=first_field,
        root_permutation=tuple(range(len(first_field.root_values))),
        basis_images=images,
    )
    permutation = tuple(
        first_field.root_values.index(_map_element(provisional, root))
        for root in first_field.root_values
    )
    return QQFieldAutomorphism(
        field=first_field,
        root_permutation=permutation,
        basis_images=images,
    )


def inverse_automorphism(
    automorphism: QQFieldAutomorphism,
) -> QQFieldAutomorphism:
    """Return the exact inverse map on a supported splitting field."""
    canonical, field = _canonical_automorphism(automorphism)
    inverse_permutation = [0] * len(canonical.root_permutation)
    for source, target in enumerate(canonical.root_permutation):
        inverse_permutation[target] = source
    if field.degree == 1:
        return _automorphism_for_generator_image(field, _one(field.extension))

    try:
        generator = _field_element(field.extension, (Fraction(0), Fraction(1)))
        generator_index = field.root_values.index(generator)
        inverse_generator = field.root_values[inverse_permutation[generator_index]]
    except ValueError as exc:
        raise ArithmeticError(
            "automorphism generator image is absent from root axis"
        ) from exc
    result = _automorphism_for_generator_image(field, inverse_generator)
    if result.root_permutation != tuple(inverse_permutation):
        raise ArithmeticError("inverse field map and root permutation disagree")
    return result


def _canonical_automorphism_subgroup(
    subgroup: GaloisAutomorphismSubgroup,
) -> GaloisAutomorphismSubgroup:
    """Replay field membership and subgroup closure for a supplied value."""
    if not isinstance(subgroup, GaloisAutomorphismSubgroup):
        raise OperationDomainValidationError(
            location=("subgroup",),
            code="galois_theory.subgroup_type",
            message="subgroup must be a typed exact automorphism subgroup",
        )
    try:
        field = _canonical_splitting_field(
            subgroup.field, location=("subgroup", "field")
        )
        candidate = GaloisAutomorphismSubgroup.model_validate(
            {**subgroup.model_dump(), "field": field.model_dump()}
        )
        elements = tuple(
            _canonical_automorphism(element)[0] for element in candidate.elements
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("subgroup",),
            code="galois_theory.invalid_subgroup",
            message="subgroup has malformed field or exact automorphism values",
        ) from exc
    if any(element.field != field for element in elements):
        raise OperationDomainValidationError(
            location=("subgroup", "elements"),
            code="galois_theory.subgroup_parent_mismatch",
            message="every subgroup element must belong to the same exact field",
        )

    full_group = automorphisms(field).automorphisms
    full_by_action = {element.root_permutation: element for element in full_group}
    by_action = {element.root_permutation: element for element in elements}
    if len(by_action) != len(elements) or any(
        action not in full_by_action for action in by_action
    ):
        raise OperationDomainValidationError(
            location=("subgroup", "elements"),
            code="galois_theory.subgroup_not_in_parent_group",
            message="subgroup elements must be distinct members of the complete automorphism group",
        )
    identity = tuple(range(len(field.root_values)))
    if identity not in by_action:
        raise OperationDomainValidationError(
            location=("subgroup", "elements"),
            code="galois_theory.subgroup_missing_identity",
            message="a subgroup must contain the identity automorphism",
        )
    for left in elements:
        inverse = inverse_automorphism(left)
        if inverse.root_permutation not in by_action:
            raise OperationDomainValidationError(
                location=("subgroup", "elements"),
                code="galois_theory.subgroup_not_closed",
                message="the supplied automorphisms are not closed under inverses",
            )
        for right in elements:
            product = compose_automorphisms(left, right)
            if product.root_permutation not in by_action:
                raise OperationDomainValidationError(
                    location=("subgroup", "elements"),
                    code="galois_theory.subgroup_not_closed",
                    message="the supplied automorphisms are not closed under composition",
                )
    return GaloisAutomorphismSubgroup(
        field=field,
        elements=tuple(by_action[key] for key in sorted(by_action)),
    )


def galois_subgroup(request: GaloisSubgroupRequest) -> GaloisAutomorphismSubgroup:
    """Admit a complete subgroup of the exact supported automorphism group."""
    try:
        canonical_request = GaloisSubgroupRequest.model_validate(request.model_dump())
        field = _canonical_splitting_field(canonical_request.field, location=("field",))
        candidate = GaloisAutomorphismSubgroup(
            field=field, elements=canonical_request.elements
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("subgroup",),
            code="galois_theory.invalid_subgroup_request",
            message="subgroup request must retain one exact supported field",
        ) from exc
    return _canonical_automorphism_subgroup(candidate)


def galois_fixed_field(
    request: GaloisFixedFieldRequest,
) -> GaloisFixedFieldResult:
    """Return the exact embedded fixed field of a supported QQ subgroup."""
    try:
        canonical_request = GaloisFixedFieldRequest.model_validate(request.model_dump())
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("subgroup",),
            code="galois_theory.invalid_fixed_field_request",
            message="fixed-field request must contain one typed automorphism subgroup",
        ) from exc
    subgroup = _canonical_automorphism_subgroup(canonical_request.subgroup)
    extension = subgroup.field.extension
    if len(subgroup.elements) == 1:
        # The trivial subgroup fixes all of L. Its canonical primitive element
        # maps to itself, including when L=QQ has degree one.
        fixed = extension
        image = (
            _field_element(extension, (Fraction(0), Fraction(1)))
            if extension.degree == 2
            else _field_element(extension, (Fraction(0),))
        )
    else:
        if extension.degree != 2:
            raise ArithmeticError("a nontrivial automorphism subgroup cannot act on QQ")
        fixed = SimpleNumberFieldPresentation(coefficients_descending=(1, 0))
        # The canonical degree-one presentation is QQ[x]/(x), so its generator
        # is the zero element of the target field.
        image = _zero(extension)
    inclusion = SimpleNumberFieldEmbedding(
        source=fixed, target=extension, generator_image=image
    )
    return GaloisFixedFieldResult(
        subgroup=subgroup, fixed_field=fixed, inclusion=inclusion
    )


def intermediate_field_stabilizer(
    request: IntermediateFieldStabilizerRequest,
) -> IntermediateFieldStabilizerResult:
    """Return automorphisms fixing a supplied embedded intermediate field."""
    try:
        canonical_request = IntermediateFieldStabilizerRequest.model_validate(
            request.model_dump()
        )
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="galois_theory.invalid_intermediate_field_request",
            message="stabilizer request must contain a typed field inclusion",
        ) from exc
    field = _canonical_splitting_field(canonical_request.field, location=("field",))
    inclusion = canonical_request.inclusion
    if inclusion.target != field.extension or inclusion.source.degree > field.degree:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="galois_theory.intermediate_field_parent_mismatch",
            message="the intermediate-field map must land in the retained extension and cannot increase degree",
        )
    source_one = _field_element(
        inclusion.source,
        (Fraction(1),) + (Fraction(0),) * (inclusion.source.degree - 1),
    )
    try:
        mapped_one = apply_simple_number_field_embedding(
            SimpleNumberFieldEmbeddingRequest(
                source=inclusion.source,
                target=inclusion.target,
                generator_image=inclusion.generator_image,
                element=source_one,
            )
        )
    except (
        OperationDomainValidationError,
        ValidationError,
        TypeError,
        ValueError,
    ) as exc:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="galois_theory.invalid_intermediate_field_embedding",
            message="intermediate-field inclusion must be an exact injective QQ-field map",
        ) from exc
    if mapped_one.image != _one(field.extension):
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="galois_theory.intermediate_field_nonunital",
            message="intermediate-field inclusion must preserve one",
        )
    fixed_elements = tuple(
        automorphism
        for automorphism in automorphisms(field).automorphisms
        if _map_element(automorphism, inclusion.generator_image)
        == inclusion.generator_image
    )
    subgroup = _canonical_automorphism_subgroup(
        GaloisAutomorphismSubgroup(field=field, elements=fixed_elements)
    )
    return IntermediateFieldStabilizerResult(inclusion=inclusion, subgroup=subgroup)


def _canonical_root(root: QQRoot) -> tuple[QQRoot, QQSplittingField]:
    if not isinstance(root, QQRoot):
        raise OperationDomainValidationError(
            location=("root",),
            code="galois_theory.root_type",
            message="root must be an exact QQ splitting-field root value",
        )
    field = _canonical_splitting_field(
        cast(QQSplittingField, getattr(root, "field", None)),
        location=("root", "field"),
    )
    try:
        canonical = QQRoot.model_validate(
            {**root.model_dump(), "field": field.model_dump()}
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("root",),
            code="galois_theory.invalid_root",
            message="root does not match the field's exact root and multiplicity axes",
        ) from exc
    return canonical, field


def apply_automorphism(automorphism: QQFieldAutomorphism, root: QQRoot) -> QQRoot:
    canonical_root, root_field = _canonical_root(root)
    canonical_automorphism, automorphism_field = _canonical_automorphism(automorphism)
    if root_field != automorphism_field:
        raise OperationDomainValidationError(
            location=("root", "field"),
            code="galois_theory.parent_mismatch",
            message="root must belong to the automorphism field",
        )
    image = _map_element(canonical_automorphism, canonical_root.value)
    index = canonical_automorphism.root_permutation[canonical_root.index]
    if image != root_field.root_values[index]:
        raise ArithmeticError("exact field action and root permutation disagree")
    return QQRoot(
        field=root_field,
        index=index,
        multiplicity=root_field.root_multiplicities[index],
        value=image,
    )


def apply_automorphism_to_element(automorphism: QQFieldAutomorphism, element):
    """Apply an exact field automorphism to any element in its source field."""
    from jacobian.math.number_theory.number_fields.values import (
        MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
        SimpleNumberFieldElement,
    )

    canonical_automorphism, field = _canonical_automorphism(automorphism)
    try:
        canonical_element = SimpleNumberFieldElement.model_validate(
            element.model_dump()
        )
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("element",),
            code="galois_theory.invalid_field_element",
            message="element must be a canonical exact element of the simple field",
        ) from exc
    if canonical_element.presentation != field.extension:
        raise OperationDomainValidationError(
            location=("element", "presentation"),
            code="galois_theory.parent_mismatch",
            message="element must belong to the automorphism field",
        )

    # For a quadratic field, the image of a+b*alpha is a+b*u+b*v*alpha.
    # Bound unreduced rational coordinates before any exact multiplication.
    element_coords = _coords(canonical_element)
    if field.extension.degree == 2:
        scalar, alpha = element_coords
        image_scalar, image_alpha = _coords(canonical_automorphism.basis_images[1])

        def product_digit_pair(left: Fraction, right: Fraction) -> tuple[int, int]:
            return (
                len(str(abs(left.numerator))) + len(str(abs(right.numerator))),
                len(str(left.denominator)) + len(str(right.denominator)),
            )

        def sum_digit_bound(
            left: Fraction, right_digits: tuple[int, int]
        ) -> tuple[int, int]:
            left_numerator = len(str(abs(left.numerator)))
            left_denominator = len(str(left.denominator))
            product_numerator, product_denominator = right_digits
            return max(
                left_numerator + product_denominator,
                product_numerator + left_denominator,
            ) + 1, max(left_denominator + product_denominator, 1)

        scalar_sum_bounds = sum_digit_bound(
            scalar, product_digit_pair(alpha, image_scalar)
        )
        alpha_bounds = product_digit_pair(alpha, image_alpha)
        scalar_bound = max(scalar_sum_bounds)
        alpha_bound = max(alpha_bounds)
        if max(scalar_bound, alpha_bound) > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS:
            raise OperationDomainValidationError(
                location=("element",),
                code="galois_theory.element_image_over_envelope",
                message=(
                    "the conservative exact automorphism-image coordinate bound "
                    f"exceeds {MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS} digits"
                ),
            )
    image = _map_element(canonical_automorphism, canonical_element)
    return _field_element(field.extension, _coords(image))


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
    "inverse_automorphism",
    "polynomial_discriminant",
    "solvable",
    "verify_galois_group",
    "verify_solvable",
]
