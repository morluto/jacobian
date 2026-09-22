"""Exact finite function-field operations over GF(p)(x)[y]/(f)."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._gfpx import (
    RF,
    ZERO_RF,
    KPoly,
    is_irreducible_over_gf,
    is_irreducible_over_rational_function,
    kp_derivative,
    kp_divmod,
    kp_gcd,
    kp_normalize,
    kp_xgcd,
    rf_add,
    rf_evaluate,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_sub,
)
from jacobian.math.function_fields._models import (
    MAX_EXTENSION_DEGREE,
    MAX_FIELD_ADMISSION_WORK,
    MAX_MULTIPLICATION_WORK,
    MAX_POLYNOMIAL_X_DEGREE,
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldDivisor,
    FunctionFieldDivisorDegreeResult,
    FunctionFieldDivisorTerm,
    FunctionFieldElementMultiplyResult,
    FunctionFieldPlace,
    FunctionFieldPrincipalDivisorRequest,
    FunctionFieldPrincipalDivisorResult,
    FunctionFieldProductTerm,
    FunctionFieldReductionStep,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _to_internal_polynomial(polynomial: PrimeFieldPolynomial) -> tuple[int, ...]:
    coefficients = polynomial.coefficients
    if coefficients == (0,):
        return ()
    return coefficients


def _to_internal_rational_function(value: PrimeFieldRationalFunction) -> RF:
    prime = value.characteristic
    return rf_normalize(
        _to_internal_polynomial(value.numerator),
        _to_internal_polynomial(value.denominator),
        prime,
    )


def _from_internal_polynomial(
    coefficients: tuple[int, ...], prime: int
) -> PrimeFieldPolynomial:
    return PrimeFieldPolynomial(
        characteristic=prime,
        coefficients=coefficients if coefficients else (0,),
    )


def _from_internal_rational_function(
    value: RF, prime: int
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=_from_internal_polynomial(value[0], prime),
        denominator=_from_internal_polynomial(value[1], prime),
    )


def _canonical_field(field: FiniteFunctionField) -> FiniteFunctionField:
    prime = field.characteristic
    return FiniteFunctionField.model_construct(
        characteristic=prime,
        variable=field.variable,
        generator=field.generator,
        defining_polynomial=tuple(
            _from_internal_rational_function(
                _to_internal_rational_function(coefficient), prime
            )
            for coefficient in field.defining_polynomial
        ),
    )


def _canonical_element(
    element: FiniteFunctionFieldElement, field: FiniteFunctionField
) -> FiniteFunctionFieldElement:
    prime = field.characteristic
    return FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(
                _to_internal_rational_function(coordinate), prime
            )
            for coordinate in element.coordinates
        ),
    )


def _field_kpoly(field: FiniteFunctionField) -> KPoly:
    return tuple(
        _to_internal_rational_function(coefficient)
        for coefficient in field.defining_polynomial
    )


def _admit_irreducibility(field: FiniteFunctionField, kpoly: KPoly) -> bool:
    """Admit irreducibility by specialization, then exact Gauss-lemma factoring.

    A monic factorization over GF(p)(x) specializes at a point with nonvanishing
    denominators to a monic factorization over GF(p); therefore an irreducible
    specialization proves irreducibility.  If all bounded specializations are
    reducible, clear rational-function denominators, remove the GF(p)[x]
    content, and factor the primitive lift exactly over GF(p)[x,y].
    """

    prime = field.characteristic
    degree_bound = len(kpoly) - 1
    for point in range(prime):
        specialization: list[int] = []
        admissible = True
        for coefficient in kpoly:
            evaluated = rf_evaluate(coefficient, point, prime)
            if evaluated is None:
                admissible = False
                break
            specialization.append(evaluated)
        if not admissible or len(specialization) - 1 != degree_bound:
            continue
        if is_irreducible_over_gf(tuple(specialization), prime):
            return True
    return is_irreducible_over_rational_function(kpoly, prime)


def _admit_field(field: FiniteFunctionField) -> None:
    prime = field.characteristic
    if not _is_prime(prime):
        raise OperationDomainValidationError(
            location=("field", "characteristic"),
            code="function_field.characteristic_not_prime",
            message="the constant field characteristic must be prime",
        )
    degree = field.degree
    if not 1 <= degree <= MAX_EXTENSION_DEGREE:
        raise OperationResourceAdmissionError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_degree_exceeds_envelope",
            message=(
                f"function fields admit extension degree at most {MAX_EXTENSION_DEGREE}"
            ),
        )
    degree_work = 0
    for coefficient in field.defining_polynomial:
        for polynomial in (coefficient.numerator, coefficient.denominator):
            if polynomial.degree > MAX_POLYNOMIAL_X_DEGREE:
                raise OperationResourceAdmissionError(
                    location=("field", "defining_polynomial"),
                    code="function_field.coefficient_degree_exceeds_envelope",
                    message=(
                        "defining-polynomial coefficient degree exceeds the "
                        f"{MAX_POLYNOMIAL_X_DEGREE} envelope"
                    ),
                )
            degree_work += polynomial.degree + 1
    if degree_work * degree * prime > MAX_FIELD_ADMISSION_WORK:
        raise OperationResourceAdmissionError(
            location=("field", "defining_polynomial"),
            code="function_field.admission_work_exceeds_envelope",
            message=(
                "function-field admission work exceeds the "
                f"{MAX_FIELD_ADMISSION_WORK} unit envelope"
            ),
        )
    canonical = _canonical_field(field)
    kpoly = _field_kpoly(canonical)
    derivative = kp_derivative(kpoly, prime)
    if len(kpoly) <= 1 or len(kp_gcd(kpoly, derivative, prime)) > 1:
        raise OperationDomainValidationError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_not_separable",
            message="the defining polynomial must be separable over GF(p)(x)",
        )
    if not _admit_irreducibility(canonical, kpoly):
        raise OperationDomainValidationError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_not_admitted_irreducible",
            message=("the defining polynomial is reducible over GF(p)(x)"),
        )


def _admit_elements(
    left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement
) -> tuple[FiniteFunctionField, FiniteFunctionFieldElement, FiniteFunctionFieldElement]:
    # Native model_construct values bypass Pydantic's coordinate-count and
    # nested rational-function checks.  Re-run that structural boundary before
    # canonicalization or any irreducibility/backend work.
    validated: list[FiniteFunctionFieldElement] = []
    for label, element in (("left", left), ("right", right)):
        if not isinstance(element, FiniteFunctionFieldElement):
            raise OperationDomainValidationError(
                location=(label,),
                code="function_field.element_type",
                message="element must be a finite function-field element value",
            )
        # Preserve the public resource boundary for an authored field whose
        # degree is intentionally above the wire model's representability cap.
        # It is admitted as a resource rejection before nested element shape is
        # inspected; malformed in-envelope carriers still take the owner error.
        defining_polynomial = getattr(
            getattr(element, "field", None), "defining_polynomial", None
        )
        if type(defining_polynomial) is tuple and len(defining_polynomial) > (
            MAX_EXTENSION_DEGREE + 1
        ):
            raise OperationResourceAdmissionError(
                location=(label, "field", "defining_polynomial"),
                code="function_field.extension_degree_exceeds_envelope",
                message=(
                    f"function fields admit extension degree at most {MAX_EXTENSION_DEGREE}"
                ),
            )
        try:
            validated.append(
                FiniteFunctionFieldElement.model_validate(element.model_dump())
            )
        except (
            ValidationError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise OperationDomainValidationError(
                location=(label,),
                code="function_field.invalid_element",
                message="element has malformed field or coordinate data",
            ) from exc
    left, right = validated
    left_field = _validated_field(left.field)
    right_field = _validated_field(right.field)
    if left_field != right_field:
        raise OperationDomainValidationError(
            location=("right", "field"),
            code="function_field.element_field_mismatch",
            message="both elements must be bound to the identical function field",
        )
    _admit_field(left_field)
    canonical_left = _canonical_element(left, left_field)
    canonical_right = _canonical_element(right, left_field)
    max_terms = 1
    total_degree = 0
    max_numerator_degree = 0
    max_denominator_degree = 0
    for element in (canonical_left, canonical_right):
        for coordinate in element.coordinates:
            max_numerator_degree = max(
                max_numerator_degree, coordinate.numerator.degree
            )
            max_denominator_degree = max(
                max_denominator_degree, coordinate.denominator.degree
            )
            for polynomial in (coordinate.numerator, coordinate.denominator):
                max_terms = max(max_terms, len(polynomial.coefficients))
                total_degree += polynomial.degree
    field_coefficient_degree = max(
        coefficient.numerator.degree + coefficient.denominator.degree
        for coefficient in left_field.defining_polynomial
    )
    # A single convolution plus one reduction step can at most square the input
    # denominator degree and add the defining-polynomial coefficient degree.
    growth_bound = (
        2 * (max_numerator_degree + max_denominator_degree) + field_coefficient_degree
    )
    if growth_bound > MAX_POLYNOMIAL_X_DEGREE:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.coefficient_growth_exceeds_envelope",
            message=(
                "the reduced product can exceed the "
                f"{MAX_POLYNOMIAL_X_DEGREE}-degree coefficient envelope"
            ),
        )
    work = (
        (2 * left_field.degree - 1)
        * max_terms
        * max(1, total_degree + left_field.degree)
    )
    if work > MAX_MULTIPLICATION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.multiplication_work_exceeds_envelope",
            message=(
                "function-field multiplication work exceeds the "
                f"{MAX_MULTIPLICATION_WORK} unit envelope"
            ),
        )
    return left_field, canonical_left, canonical_right


def _internal_coordinates(
    element: FiniteFunctionFieldElement,
) -> tuple[RF, ...]:
    return tuple(
        _to_internal_rational_function(coordinate) for coordinate in element.coordinates
    )


def _multiply_internal(
    left: tuple[RF, ...],
    right: tuple[RF, ...],
    kpoly: KPoly,
    prime: int,
) -> tuple[
    tuple[RF, ...],
    tuple[tuple[int, RF], ...],
    tuple[tuple[int, RF, tuple[tuple[int, RF], ...]], ...],
]:
    degree = len(kpoly) - 1
    raw: list[RF] = [ZERO_RF] * (2 * degree - 1)
    for left_index, left_value in enumerate(left):
        if rf_is_zero(left_value):
            continue
        for right_index, right_value in enumerate(right):
            if rf_is_zero(right_value):
                continue
            position = left_index + right_index
            raw[position] = rf_add(
                raw[position], rf_mul(left_value, right_value, prime), prime
            )
    raw_terms = tuple(
        (position, raw[position])
        for position in range(len(raw))
        if not rf_is_zero(raw[position])
    )
    reduction_steps: list[tuple[int, RF, tuple[tuple[int, RF], ...]]] = []
    for position in range(2 * degree - 2, degree - 1, -1):
        coefficient = raw[position]
        if rf_is_zero(coefficient):
            continue
        raw[position] = ZERO_RF
        replacement: list[tuple[int, RF]] = []
        for index in range(degree):
            field_coefficient = kpoly[index]
            if rf_is_zero(field_coefficient):
                continue
            contribution = rf_mul(coefficient, field_coefficient, prime)
            target = position - degree + index
            raw[target] = rf_sub(raw[target], contribution, prime)
            replacement.append((target, contribution))
        reduction_steps.append((position, coefficient, tuple(replacement)))
    reduced = tuple(raw[index] for index in range(degree))
    reduction_steps.sort(key=lambda step: step[0])
    return reduced, raw_terms, tuple(reduction_steps)


def function_field_element_multiply(
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
) -> FunctionFieldElementMultiplyResult:
    """Exact product in GF(p)(x)[y]/(f) with a complete reduction ledger."""

    field, left, right = _admit_elements(left, right)
    prime = field.characteristic
    kpoly = _field_kpoly(field)
    reduced, raw_terms, reduction_steps = _multiply_internal(
        _internal_coordinates(left),
        _internal_coordinates(right),
        kpoly,
        prime,
    )
    product = FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(value, prime) for value in reduced
        ),
    )
    return FunctionFieldElementMultiplyResult._from_kernel(
        field=field,
        left=left,
        right=right,
        raw_product_terms=tuple(
            FunctionFieldProductTerm(
                y_power=position,
                coefficient=_from_internal_rational_function(value, prime),
            )
            for position, value in raw_terms
        ),
        reduction_steps=tuple(
            FunctionFieldReductionStep(
                y_power=position,
                coefficient=_from_internal_rational_function(coefficient, prime),
                replacement_terms=tuple(
                    FunctionFieldProductTerm(
                        y_power=target,
                        coefficient=_from_internal_rational_function(value, prime),
                    )
                    for target, value in replacement
                ),
            )
            for position, coefficient, replacement in reduction_steps
        ),
        product=product,
    )


def _element_inverse(
    element: FiniteFunctionFieldElement,
) -> FiniteFunctionFieldElement:
    """Private exact inverse via extended Euclid in GF(p)(x)[y]/(f)."""

    field, canonical, _ = _admit_elements(element, element)
    prime = field.characteristic
    kpoly = _field_kpoly(field)
    value = _internal_coordinates(canonical)
    if all(rf_is_zero(coordinate) for coordinate in value):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.zero_not_invertible",
            message="the zero element has no multiplicative inverse",
        )
    _, inverse, _ = kp_xgcd(kp_normalize(value), kpoly, prime)
    _, remainder = kp_divmod(inverse, kpoly, prime)
    degree = field.degree
    padded = tuple(
        remainder[index] if index < len(remainder) else ZERO_RF
        for index in range(degree)
    )
    return FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(coordinate, prime) for coordinate in padded
        ),
    )


def _validated_field(field: FiniteFunctionField) -> FiniteFunctionField:
    """Re-admit a native/model_construct field without doing backend work."""

    if not isinstance(field, FiniteFunctionField):
        raise OperationDomainValidationError(
            location=("field",),
            code="function_field.field_type",
            message="field must be a finite function-field value",
        )
    try:
        return FiniteFunctionField.model_validate(field.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("field",),
            code="function_field.invalid_field",
            message="field has malformed defining-polynomial data",
        ) from exc


def _canonical_place(place: FunctionFieldPlace) -> FunctionFieldPlace:
    """Re-admit place shape before factoring or any other backend call."""

    if not isinstance(place, FunctionFieldPlace):
        raise OperationDomainValidationError(
            location=("place",),
            code="function_field.place_type",
            message="place must be a function-field place value",
        )
    try:
        return FunctionFieldPlace.model_validate(place.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("place",),
            code="function_field.invalid_place",
            message="place has malformed field or place-shape data",
        ) from exc


def _admit_rational_place_field(field: FiniteFunctionField) -> FiniteFunctionField:
    field = _validated_field(field)
    if not _is_prime(field.characteristic):
        raise OperationDomainValidationError(
            location=("field",),
            code="function_field.characteristic_not_prime",
            message="constant characteristic must be prime",
        )
    if len(field.defining_polynomial) != 1 or not (
        field.defining_polynomial[0].numerator.is_one()
        and field.defining_polynomial[0].denominator.is_one()
    ):
        raise OperationDomainValidationError(
            location=("field",),
            code="function_field.place_extension_unsupported",
            message="places in this exact slice require the rational field GF(p)(x)",
        )
    return field


def _monic_polynomial(poly: PrimeFieldPolynomial) -> PrimeFieldPolynomial:
    """Return the monic associate defining the same finite place."""

    leading = poly.coefficients[-1] % poly.characteristic
    inverse = pow(leading, -1, poly.characteristic)
    return PrimeFieldPolynomial(
        characteristic=poly.characteristic,
        coefficients=tuple(
            (coefficient * inverse) % poly.characteristic
            for coefficient in poly.coefficients
        ),
    )


def _admit_place(place: FunctionFieldPlace) -> FunctionFieldPlace:
    place = _canonical_place(place)
    field = _admit_rational_place_field(place.field)
    prime_polynomial = place.prime_polynomial
    if place.kind == "FINITE":
        assert prime_polynomial is not None
        # Polynomial associates define the same prime ideal/place.  The public
        # carrier permits either spelling, while valuation and factorization use
        # the canonical monic representative.
        prime_polynomial = _monic_polynomial(prime_polynomial)
    # Preserve the canonical field and monic place polynomial through the
    # returned value so parent and valuation operations see one representation.
    place = FunctionFieldPlace.model_construct(
        field=field,
        kind=place.kind,
        prime_polynomial=prime_polynomial,
        degree=place.degree,
    )
    if place.kind == "FINITE":
        assert place.prime_polynomial is not None
        factors = _factor_polynomial(place.prime_polynomial)
        if factors != ((place.prime_polynomial, 1),):
            raise OperationDomainValidationError(
                location=("place",),
                code="function_field.place_not_prime",
                message="finite place polynomial must be irreducible",
            )
    return place


def _rf_valuation(value: PrimeFieldRationalFunction, place: FunctionFieldPlace) -> int:
    prime = value.characteristic
    num = list(value.numerator.coefficients)
    den = list(value.denominator.coefficients)
    if place.kind == "INFINITE":
        return (len(den) - 1) - (len(num) - 1)
    assert place.prime_polynomial is not None
    divisor = list(place.prime_polynomial.coefficients)

    def order(poly: list[int]) -> int:
        count = 0
        while len(poly) >= len(divisor):
            quotient, remainder = _poly_divmod_local(poly, divisor, prime)
            if any(remainder):
                break
            count += 1
            poly = quotient
        return count

    return order(num) - order(den)


def _poly_divmod_local(
    dividend: list[int], divisor: list[int], prime: int
) -> tuple[list[int], list[int]]:
    dividend = [x % prime for x in dividend]
    while len(dividend) > 1 and dividend[-1] == 0:
        dividend.pop()
    divisor = [x % prime for x in divisor]
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    if len(dividend) < len(divisor):
        return [0], dividend
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    inv = pow(divisor[-1], -1, prime)
    while len(dividend) >= len(divisor) and any(dividend):
        shift = len(dividend) - len(divisor)
        factor = dividend[-1] * inv % prime
        quotient[shift] = factor
        for i, c in enumerate(divisor):
            dividend[shift + i] = (dividend[shift + i] - factor * c) % prime
        while len(dividend) > 1 and dividend[-1] == 0:
            dividend.pop()
    return quotient, dividend


def function_field_place_valuation(
    place: FunctionFieldPlace, element: FiniteFunctionFieldElement
) -> int | None:
    place = _admit_place(place)
    if not isinstance(element, FiniteFunctionFieldElement):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.element_type",
            message="element must be a finite function-field element value",
        )
    try:
        element = FiniteFunctionFieldElement.model_validate(element.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.invalid_element",
            message="element has malformed coordinate data",
        ) from exc
    if element.field != place.field:
        raise OperationDomainValidationError(
            location=("element", "field"),
            code="function_field.parent_mismatch",
            message="place and element must share the exact function field",
        )
    coordinate = element.coordinates[0]
    if coordinate.numerator.is_zero():
        return None
    return _rf_valuation(coordinate, place)


def _factor_polynomial(
    poly: PrimeFieldPolynomial,
) -> tuple[tuple[PrimeFieldPolynomial, int], ...]:
    """Factor one bounded GF(p)[x] polynomial through the stable Poly API.

    ``sympy.factor_list(Poly(...))`` delegates to the installed flint domain in
    some environments and can raise ``nmods cannot be ordered``.  The owning
    adapter uses ``Poly.factor_list`` instead, which keeps the finite-field
    domain explicit and returns ordinary coefficient data before any public
    value is constructed.
    """

    from sympy import Poly, symbols

    x = symbols("x")
    expression = sum(c * x**i for i, c in enumerate(poly.coefficients))
    try:
        _unit, factors = Poly(expression, x, modulus=poly.characteristic).factor_list()
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="function_field.factorization_backend_failure",
            message="finite-place factorization failed in the admitted GF(p) domain",
        ) from exc
    return tuple(
        (
            PrimeFieldPolynomial(
                characteristic=poly.characteristic,
                coefficients=tuple(
                    int(value) % poly.characteristic
                    for value in reversed(f.all_coeffs())
                ),
            ),
            int(multiplicity),
        )
        for f, multiplicity in factors
    )


def function_field_principal_divisor(
    request: FunctionFieldPrincipalDivisorRequest,
) -> FunctionFieldPrincipalDivisorResult:
    if not isinstance(request, FunctionFieldPrincipalDivisorRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="function_field.principal_divisor_request_type",
            message="request must contain a function field and one of its elements",
        )
    try:
        request = FunctionFieldPrincipalDivisorRequest.model_validate(
            request.model_dump()
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="function_field.invalid_principal_divisor_request",
            message="principal-divisor request has malformed field or element data",
        ) from exc
    field = _admit_rational_place_field(request.field)
    if not isinstance(request.element, FiniteFunctionFieldElement):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.element_type",
            message="element must be a finite function-field element value",
        )
    try:
        element = FiniteFunctionFieldElement.model_validate(
            request.element.model_dump()
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.invalid_element",
            message="element has malformed coordinate data",
        ) from exc
    if element.field != field:
        raise OperationDomainValidationError(
            location=("element", "field"),
            code="function_field.parent_mismatch",
            message="element must belong to the supplied field",
        )
    # Canonicalize once before factoring.  This performs numerator/denominator
    # cancellation and makes scalar denominators monic, so equal finite places
    # cannot be emitted twice by a merely different presentation of the same
    # rational function.
    canonical_value = _to_internal_rational_function(element.coordinates[0])
    value = _from_internal_rational_function(canonical_value, field.characteristic)
    canonical_element = FiniteFunctionFieldElement.model_construct(
        field=field, coordinates=(value,)
    )
    if value.numerator.is_zero():
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.zero_principal_divisor",
            message="zero has no principal divisor",
        )
    support: dict[str, tuple[FunctionFieldPlace, int]] = {}
    for polynomial, sign in ((value.numerator, 1), (value.denominator, -1)):
        for factor, multiplicity in _factor_polynomial(polynomial):
            place = FunctionFieldPlace(
                field=field,
                kind="FINITE",
                prime_polynomial=factor,
                degree=factor.degree,
            )
            key = place.model_dump_json()
            previous = support.get(key)
            support[key] = (
                place,
                (previous[1] if previous else 0) + sign * multiplicity,
            )
    infinity_place = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)
    infinity = _rf_valuation(value, infinity_place)
    if infinity:
        key = infinity_place.model_dump_json()
        support[key] = (infinity_place, infinity)
    terms = tuple(
        FunctionFieldDivisorTerm(place=place, multiplicity=multiplicity)
        for place, multiplicity in support.values()
        if multiplicity
    )
    divisor = FunctionFieldDivisor(field=field, terms=terms)
    return FunctionFieldPrincipalDivisorResult(
        field=field, element=canonical_element, divisor=divisor, degree=divisor.degree
    )


def _admit_divisor(divisor: FunctionFieldDivisor) -> FunctionFieldDivisor:
    """Bind every place to the divisor parent before place factorization."""

    if not isinstance(divisor, FunctionFieldDivisor):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.divisor_type",
            message="divisor must be a function-field divisor value",
        )
    try:
        field_value = divisor.field
        terms = divisor.terms
    except AttributeError as exc:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.invalid_divisor",
            message="divisor has malformed field or support data",
        ) from exc
    field = _validated_field(field_value)
    if type(terms) is not tuple or len(terms) > 256:
        raise OperationDomainValidationError(
            location=("divisor", "terms"),
            code="function_field.divisor_shape",
            message="divisor terms must be a bounded canonical tuple",
        )
    admitted_terms: list[FunctionFieldDivisorTerm] = []
    for index, term in enumerate(terms):
        location = ("divisor", "terms", index)
        if not isinstance(term, FunctionFieldDivisorTerm):
            raise OperationDomainValidationError(
                location=location,
                code="function_field.divisor_term_type",
                message="divisor terms must be typed place/multiplicity values",
            )
        if type(term.multiplicity) is not int or term.multiplicity == 0:
            raise OperationDomainValidationError(
                location=(*location, "multiplicity"),
                code="function_field.divisor_multiplicity",
                message="divisor multiplicities must be nonzero strict integers",
            )
        # Check the parent using only structural admission.  In particular, a
        # mismatched place must not reach finite-place factorization first.
        place = _canonical_place(term.place)
        if place.field != field:
            raise OperationDomainValidationError(
                location=(*location, "place", "field"),
                code="function_field.divisor_parent",
                message="every divisor place must belong to divisor.field",
            )
        admitted_place = _admit_place(place)
        admitted_terms.append(
            FunctionFieldDivisorTerm(
                place=admitted_place, multiplicity=term.multiplicity
            )
        )
    try:
        return FunctionFieldDivisor(field=field, terms=tuple(admitted_terms))
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.invalid_divisor",
            message="divisor support must be canonical and parent-bound",
        ) from exc


def function_field_divisor_degree(
    divisor: FunctionFieldDivisor,
) -> FunctionFieldDivisorDegreeResult:
    divisor = _admit_divisor(divisor)
    return FunctionFieldDivisorDegreeResult(divisor=divisor, degree=divisor.degree)


__all__ = [
    "_element_inverse",
    "function_field_divisor_degree",
    "function_field_element_multiply",
    "function_field_place_valuation",
    "function_field_principal_divisor",
]
