"""Exact finite function-field operations over GF(p)(x)[y]/(f)."""

from __future__ import annotations

from itertools import product

from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.function_fields._gfpx import (
    ONE_POLY,
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
    poly_derivative,
    poly_divmod,
    poly_gcd,
    poly_mul,
    poly_powmod,
    rf_add,
    rf_evaluate,
    rf_inv,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_sub,
)
from jacobian.math.function_fields._models import (
    MAX_BASE_EMBEDDING_VALUE_BYTES,
    MAX_DIVISOR_MULTIPLICITY_BITS,
    MAX_ELEMENT_ADDITION_WORK,
    MAX_ELEMENT_VALUE_BYTES,
    MAX_EXTENSION_DEGREE,
    MAX_FIELD_ADMISSION_WORK,
    MAX_INVERSION_WORK,
    MAX_MULTIPLICATION_WORK,
    MAX_POLYNOMIAL_X_DEGREE,
    MAX_RATIONAL_PLACE_CANDIDATES,
    MAX_RATIONAL_PLACE_DEGREE,
    MAX_RATIONAL_PLACE_OUTPUT,
    MAX_RATIONAL_PLACE_WORK,
    MAX_RIEMANN_ROCH_BASIS_DIMENSION,
    MAX_RIEMANN_ROCH_CONSTRUCTION_WORK,
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldBaseEmbedding,
    FunctionFieldBaseEmbeddingApplyResult,
    FunctionFieldBaseEmbeddingResult,
    FunctionFieldDivisor,
    FunctionFieldDivisorDegreeResult,
    FunctionFieldDivisorEffectivePartsResult,
    FunctionFieldDivisorTerm,
    FunctionFieldElementMultiplyResult,
    FunctionFieldGenusResult,
    FunctionFieldPlace,
    FunctionFieldPlaceEnumerationResult,
    FunctionFieldPrincipalDivisorResult,
    FunctionFieldProductTerm,
    FunctionFieldReductionStep,
    FunctionFieldResidueResult,
    FunctionFieldRiemannRochSpace,
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
    _admit_field_resources(field)
    _admit_field_algebra(field)


def _admit_field_resources(field: FiniteFunctionField) -> None:
    """Check finite coefficient and work bounds without backend expansion."""

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


def _admit_field_algebra(field: FiniteFunctionField) -> None:
    """Check separability and irreducibility after resource admission."""

    prime = field.characteristic
    canonical = _canonical_field(field)
    kpoly = _field_kpoly(canonical)
    if len(kpoly) == 1:
        # The degree-one value denotes the rational function field GF(p)(x),
        # represented structurally by the constant polynomial 1.
        return
    derivative = kp_derivative(kpoly, prime)
    if len(kp_gcd(kpoly, derivative, prime)) > 1:
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


def function_field_base_embedding(
    target: FiniteFunctionField,
) -> FunctionFieldBaseEmbeddingResult:
    """Construct the natural inclusion of GF(p)(x) into a presented extension."""

    target = _validated_field(target)
    if target.degree <= 1:
        raise OperationDomainValidationError(
            location=("target",),
            code="function_field.base_embedding_requires_extension",
            message="the target must be a nontrivial algebraic extension of GF(p)(x)",
        )
    field_bytes = 128 + len(target.variable) + len(target.generator)
    for coefficient in target.defining_polynomial:
        field_bytes += 96
        for polynomial in (coefficient.numerator, coefficient.denominator):
            field_bytes += 8 + 4 * len(polynomial.coefficients)
    encoded_bytes = 256 + 2 * field_bytes + 128 + 160 * target.degree
    if encoded_bytes > MAX_BASE_EMBEDDING_VALUE_BYTES:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="function_field.base_embedding_output_exceeds_envelope",
            message=(
                "the base embedding result exceeds the "
                f"{MAX_BASE_EMBEDDING_VALUE_BYTES}-byte output envelope"
            ),
        )
    target = _canonical_field(target)
    _admit_field(target)
    one = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(1,)
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(1,)
        ),
    )
    source = FiniteFunctionField(
        characteristic=target.characteristic,
        variable=target.variable,
        generator=target.generator,
        defining_polynomial=(one,),
    )
    variable = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(0, 1)
        ),
        denominator=one.denominator,
    )
    zero = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(0,)
        ),
        denominator=one.denominator,
    )
    variable_image = FiniteFunctionFieldElement(
        field=target,
        coordinates=(variable, *(zero for _ in range(target.degree - 1))),
    )
    embedding = FunctionFieldBaseEmbedding(
        source=source,
        target=target,
        variable_image=variable_image,
    )
    return FunctionFieldBaseEmbeddingResult(embedding=embedding)


def function_field_base_embedding_apply(
    embedding: FunctionFieldBaseEmbedding,
    element: FiniteFunctionFieldElement,
) -> FunctionFieldBaseEmbeddingApplyResult:
    """Apply the canonical rational-base inclusion to one rational function."""

    if not isinstance(embedding, FunctionFieldBaseEmbedding) or not isinstance(
        element, FiniteFunctionFieldElement
    ):
        raise OperationDomainValidationError(
            location=(),
            code="function_field.base_embedding_request_type",
            message="embedding and element must be function-field values",
        )
    # A model_construct carrier bypasses Pydantic's nested checks, so
    # re-admission may fail while dumping or revalidating.  Translate those
    # malformed native values into the operation's stable domain errors.
    try:
        embedding = FunctionFieldBaseEmbedding.model_validate(embedding.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("embedding",),
            code="function_field.invalid_base_embedding",
            message="embedding has malformed field, target, or variable-image data",
        ) from exc
    try:
        element = FiniteFunctionFieldElement.model_validate(element.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.invalid_element",
            message="element has malformed coordinate data",
        ) from exc
    if element.field != embedding.source:
        raise OperationDomainValidationError(
            location=("element", "field"),
            code="function_field.base_embedding_element_parent",
            message="element must belong to the embedding source field",
        )
    target = _validated_field(embedding.target)
    target_bytes = 128 + len(target.variable) + len(target.generator)
    for coefficient in target.defining_polynomial:
        target_bytes += 96
        for polynomial in (coefficient.numerator, coefficient.denominator):
            target_bytes += 8 + 4 * len(polynomial.coefficients)
    encoded_bytes = 1024 + 4 * target_bytes + 320 * target.degree
    if encoded_bytes > MAX_BASE_EMBEDDING_VALUE_BYTES:
        raise OperationResourceAdmissionError(
            location=("embedding", "target"),
            code="function_field.base_embedding_output_exceeds_envelope",
            message=(
                "the base embedding application exceeds the "
                f"{MAX_BASE_EMBEDDING_VALUE_BYTES}-byte output envelope"
            ),
        )
    target = _canonical_field(target)
    _admit_field(target)
    source = FiniteFunctionField(
        characteristic=target.characteristic,
        variable=target.variable,
        generator=target.generator,
        defining_polynomial=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(
                    characteristic=target.characteristic, coefficients=(1,)
                ),
                denominator=PrimeFieldPolynomial(
                    characteristic=target.characteristic, coefficients=(1,)
                ),
            ),
        ),
    )
    variable = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(0, 1)
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(1,)
        ),
    )
    zero = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=target.characteristic, coefficients=(0,)
        ),
        denominator=variable.denominator,
    )
    canonical_embedding = FunctionFieldBaseEmbedding(
        source=source,
        target=target,
        variable_image=FiniteFunctionFieldElement(
            field=target,
            coordinates=(variable, *(zero for _ in range(target.degree - 1))),
        ),
    )
    source_element = FiniteFunctionFieldElement(
        field=source,
        coordinates=(
            _from_internal_rational_function(
                _to_internal_rational_function(element.coordinates[0]),
                target.characteristic,
            ),
        ),
    )
    image = FiniteFunctionFieldElement(
        field=target,
        coordinates=(
            source_element.coordinates[0],
            *(zero for _ in range(target.degree - 1)),
        ),
    )
    return FunctionFieldBaseEmbeddingApplyResult._from_kernel(
        embedding=canonical_embedding, source_element=source_element, image=image
    )


def _admit_addition_resources(
    field: FiniteFunctionField,
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
) -> None:
    """Preflight rational-function addition after source canonicalization."""

    output_degrees: list[int] = []
    work = 0
    prime = field.characteristic
    for left_coordinate, right_coordinate in zip(
        left.coordinates, right.coordinates, strict=True
    ):
        ln, ld = left_coordinate.numerator.degree, left_coordinate.denominator.degree
        rn, rd = right_coordinate.numerator.degree, right_coordinate.denominator.degree
        if left_coordinate.numerator.is_zero():
            output_degree = max(rn, rd)
            coordinate_work = rn + rd + 2
        elif right_coordinate.numerator.is_zero():
            output_degree = max(ln, ld)
            coordinate_work = ln + ld + 2
        else:
            numerator_degree = max(ln + rd, rn + ld)
            denominator_degree = ld + rd
            output_degree = max(numerator_degree, denominator_degree)
            # Bound the two cross products, numerator addition, denominator
            # product, and bounded Euclidean normalization in GF(p)[x].
            cross_products = (ln + 1) * (rd + 1) + (rn + 1) * (ld + 1)
            denominator_product = (ld + 1) * (rd + 1)
            euclidean_normalization = 3 * (output_degree + 1) ** 3
            coordinate_work = (
                cross_products
                + denominator_product
                + numerator_degree
                + 1
                + euclidean_normalization
            )
        output_degrees.append(output_degree)
        work += coordinate_work
    maximum_degree = max(output_degrees, default=0)
    if maximum_degree > MAX_POLYNOMIAL_X_DEGREE:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.addition_coefficient_growth_exceeds_envelope",
            message=(
                "the exact rational-function sum can exceed the "
                f"{MAX_POLYNOMIAL_X_DEGREE}-degree coefficient envelope"
            ),
        )
    if work > MAX_ELEMENT_ADDITION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.element_addition_work_exceeds_envelope",
            message=(
                "function-field addition work exceeds the "
                f"{MAX_ELEMENT_ADDITION_WORK} unit envelope"
            ),
        )
    output_template = {"field": field.model_dump(mode="json"), "coordinates": []}
    encoded_bytes = len(encode_strict_json(output_template))
    for degree in output_degrees:
        polynomial = {
            "characteristic": prime,
            "coefficients": [prime - 1] * (degree + 1),
        }
        coordinate = {"numerator": polynomial, "denominator": polynomial}
        encoded_bytes += len(encode_strict_json(coordinate)) + 1
    if encoded_bytes > MAX_ELEMENT_VALUE_BYTES:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.element_addition_output_exceeds_envelope",
            message=(
                "the exact function-field sum exceeds the "
                f"{MAX_ELEMENT_VALUE_BYTES}-byte output envelope"
            ),
        )


def _admit_multiplication_resources(
    field: FiniteFunctionField,
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
) -> None:
    """Preflight the existing product envelope for canonical operands."""

    max_terms = 1
    total_degree = 0
    max_numerator_degree = 0
    max_denominator_degree = 0
    for element in (left, right):
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
        for coefficient in field.defining_polynomial
    )
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
    work = (2 * field.degree - 1) * max_terms * max(1, total_degree + field.degree)
    if work > MAX_MULTIPLICATION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.multiplication_work_exceeds_envelope",
            message=(
                "function-field multiplication work exceeds the "
                f"{MAX_MULTIPLICATION_WORK} unit envelope"
            ),
        )


def _admit_elements(
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
    *,
    operation: str = "multiplication",
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
    if operation == "addition":
        _admit_addition_resources(left_field, canonical_left, canonical_right)
    else:
        _admit_multiplication_resources(left_field, canonical_left, canonical_right)
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
    if field.degree == 1:
        # The rational field GF(p)(x) uses the sentinel defining polynomial 1;
        # its elements are single rational functions with no generator reduction.
        left_value = _internal_coordinates(left)[0]
        right_value = _internal_coordinates(right)[0]
        product_value = rf_mul(left_value, right_value, prime)
        product_coord = _from_internal_rational_function(product_value, prime)
        product = FiniteFunctionFieldElement.model_construct(
            field=field, coordinates=(product_coord,)
        )
        raw_terms: tuple[tuple[int, RF], ...] = (
            ((0, product_value),) if not rf_is_zero(product_value) else ()
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
            reduction_steps=(),
            product=product,
        )
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


def function_field_element_add(
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
) -> FiniteFunctionFieldElement:
    """Return the exact coordinatewise sum in one presented function field."""

    field, left, right = _admit_elements(left, right, operation="addition")
    prime = field.characteristic
    coordinates = tuple(
        _from_internal_rational_function(
            rf_add(
                _to_internal_rational_function(left_coordinate),
                _to_internal_rational_function(right_coordinate),
                prime,
            ),
            prime,
        )
        for left_coordinate, right_coordinate in zip(
            left.coordinates, right.coordinates, strict=True
        )
    )
    return FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=coordinates,
    )


def function_field_element_inverse(
    element: FiniteFunctionFieldElement,
) -> FiniteFunctionFieldElement:
    """Return the inverse via extended Euclid in the exact function field.

    A resultant-style degree bound admits the coefficient growth before the
    Euclidean algorithm.  The classical polynomial remainder sequence can
    have intermediate rational coefficients larger than the final inverse,
    so the admission uses a deliberately conservative determinant bound.
    """

    field, canonical = _preflight_inverse_operand(element)
    value = _internal_coordinates(canonical)
    if all(rf_is_zero(coordinate) for coordinate in value):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.zero_not_invertible",
            message="the zero element has no multiplicative inverse",
        )

    numerator_degree = max(
        len(coordinate[0]) - 1 for coordinate in value if coordinate[0]
    )
    denominator_degree = max(c[1].__len__() - 1 for c in value)
    field_numerator_degree = max(
        len(coefficient[0]) - 1 for coefficient in _field_kpoly(field) if coefficient[0]
    )
    field_denominator_degree = max(
        coefficient[1].__len__() - 1 for coefficient in _field_kpoly(field)
    )
    degree = field.degree
    coefficient_bound = degree**2 * (
        numerator_degree
        + denominator_degree
        + field_numerator_degree
        + field_denominator_degree
        + 1
    )
    if coefficient_bound > MAX_POLYNOMIAL_X_DEGREE:
        raise OperationResourceAdmissionError(
            location=("element", "coordinates"),
            code="function_field.inverse_coefficient_growth_exceeds_envelope",
            message=(
                "the conservative extended-Euclid coefficient bound exceeds "
                f"the degree-{MAX_POLYNOMIAL_X_DEGREE} envelope"
            ),
        )
    work = degree**3 * (coefficient_bound + 1) ** 2
    if work > MAX_INVERSION_WORK:
        raise OperationResourceAdmissionError(
            location=("element", "coordinates"),
            code="function_field.inverse_work_exceeds_envelope",
            message=f"function-field inversion exceeds the {MAX_INVERSION_WORK} unit envelope",
        )
    # Six coordinates, each with numerator and denominator of at most 13
    # coefficients in 0..256, remains below this exact JSON envelope.  The
    # input field is already bounded and retained verbatim in the result.
    encoded_bound = len(
        encode_strict_json(
            {
                "field": field.model_dump(mode="json"),
                "coordinates": [
                    {
                        "numerator": {
                            "characteristic": field.characteristic,
                            "coefficients": [field.characteristic - 1]
                            * (coefficient_bound + 1),
                        },
                        "denominator": {
                            "characteristic": field.characteristic,
                            "coefficients": [field.characteristic - 1]
                            * (coefficient_bound + 1),
                        },
                    }
                    for _ in range(degree)
                ],
            }
        )
    )
    if encoded_bound > MAX_ELEMENT_VALUE_BYTES:
        raise OperationResourceAdmissionError(
            location=("element", "coordinates"),
            code="function_field.inverse_output_exceeds_envelope",
            message="the exact inverse exceeds the function-field element output envelope",
        )
    _admit_field_algebra(field)
    return _inverse_canonical(field, value)


def _preflight_inverse_operand(
    element: FiniteFunctionFieldElement,
) -> tuple[FiniteFunctionField, FiniteFunctionFieldElement]:
    """Revalidate and bound one element without irreducibility work."""

    if not isinstance(element, FiniteFunctionFieldElement):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.element_type",
            message="element must be a finite function-field element value",
        )
    defining_polynomial = getattr(
        getattr(element, "field", None), "defining_polynomial", None
    )
    if type(defining_polynomial) is tuple and len(defining_polynomial) > (
        MAX_EXTENSION_DEGREE + 1
    ):
        raise OperationResourceAdmissionError(
            location=("element", "field", "defining_polynomial"),
            code="function_field.extension_degree_exceeds_envelope",
            message=(
                f"function fields admit extension degree at most {MAX_EXTENSION_DEGREE}"
            ),
        )
    try:
        validated = FiniteFunctionFieldElement.model_validate(element.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.invalid_element",
            message="element has malformed field or coordinate data",
        ) from exc
    field = _validated_field(validated.field)
    _admit_field_resources(field)
    return field, _canonical_element(validated, field)


def _element_inverse(
    element: FiniteFunctionFieldElement,
) -> FiniteFunctionFieldElement:
    """Private exact inverse via extended Euclid in GF(p)(x)[y]/(f)."""

    field, canonical, _ = _admit_elements(element, element)
    value = _internal_coordinates(canonical)
    if all(rf_is_zero(coordinate) for coordinate in value):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.zero_not_invertible",
            message="the zero element has no multiplicative inverse",
        )
    return _inverse_canonical(field, value)


def _inverse_canonical(
    field: FiniteFunctionField, value: tuple[RF, ...]
) -> FiniteFunctionFieldElement:
    """Euclidean inverse for an admitted, nonzero canonical element."""

    prime = field.characteristic
    if field.degree == 1:
        # Rational-field elements are single rational functions; the sentinel
        # defining polynomial 1 is not an extension modulus.
        if not value or rf_is_zero(value[0]):
            raise OperationDomainValidationError(
                location=("element",),
                code="function_field.zero_not_invertible",
                message="the zero element has no multiplicative inverse",
            )
        return FiniteFunctionFieldElement.model_construct(
            field=field,
            coordinates=(
                _from_internal_rational_function(rf_inv(value[0], prime), prime),
            ),
        )
    kpoly = _field_kpoly(field)
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


def _validated_field(field: object) -> FiniteFunctionField:
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


def _canonical_place(place: object) -> FunctionFieldPlace:
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


def function_field_rational_places_degree_bounded(
    field: FiniteFunctionField,
    maximum_degree: int,
) -> FunctionFieldPlaceEnumerationResult:
    """Enumerate every place of GF(p)(x) through the requested degree.

    Finite places correspond bijectively to monic irreducible polynomials in
    GF(p)[x]; the unique infinite place has degree one. The complete candidate
    space is admitted before irreducibility work or result construction.
    """

    if type(maximum_degree) is not int:
        raise OperationDomainValidationError(
            location=("maximum_degree",),
            code="function_field.place_enumeration_request_type",
            message="maximum degree must be a strict integer",
        )
    if not 1 <= maximum_degree <= MAX_RATIONAL_PLACE_DEGREE:
        raise OperationDomainValidationError(
            location=("maximum_degree",),
            code="function_field.invalid_place_enumeration_request",
            message="maximum degree is outside the admitted degree bound",
        )
    field = _admit_rational_place_field(field)
    prime = field.characteristic
    candidate_count = sum(prime**degree for degree in range(1, maximum_degree + 1))
    estimated_work = sum(
        prime**degree * degree**3 * max(1, prime.bit_length())
        for degree in range(1, maximum_degree + 1)
    )
    if candidate_count > MAX_RATIONAL_PLACE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="function_field.place_enumeration_candidates_exceed_envelope",
            message=(
                "complete rational-place enumeration requires testing more than "
                f"{MAX_RATIONAL_PLACE_CANDIDATES} monic candidates"
            ),
        )
    if estimated_work > MAX_RATIONAL_PLACE_WORK:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="function_field.place_enumeration_work_exceeds_envelope",
            message=(
                "rational-place irreducibility work exceeds the "
                f"{MAX_RATIONAL_PLACE_WORK} unit envelope"
            ),
        )
    if candidate_count + 1 > MAX_RATIONAL_PLACE_OUTPUT:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="function_field.place_enumeration_output_exceeds_envelope",
            message=(
                "the complete rational-place result could exceed "
                f"{MAX_RATIONAL_PLACE_OUTPUT} places"
            ),
        )

    places = [FunctionFieldPlace(field=field, kind="INFINITE", degree=1)]
    for degree in range(1, maximum_degree + 1):
        for lower_coefficients in product(range(prime), repeat=degree):
            coefficients = (*lower_coefficients, 1)
            if is_irreducible_over_gf(coefficients, prime):
                places.append(
                    FunctionFieldPlace(
                        field=field,
                        kind="FINITE",
                        prime_polynomial=PrimeFieldPolynomial(
                            characteristic=prime,
                            coefficients=coefficients,
                        ),
                        degree=degree,
                    )
                )
    places.sort(
        key=lambda place: (
            place.degree,
            place.kind,
            ()
            if place.prime_polynomial is None
            else place.prime_polynomial.coefficients,
        )
    )
    return FunctionFieldPlaceEnumerationResult(
        field=field,
        maximum_degree=maximum_degree,
        places=tuple(places),
    )


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
        if prime_polynomial is None:
            raise OperationDomainValidationError(
                location=("place", "prime_polynomial"),
                code="function_field.finite_place_polynomial",
                message="a finite place requires its prime polynomial",
            )
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
        admitted_polynomial = place.prime_polynomial
        if admitted_polynomial is None:
            raise OperationDomainValidationError(
                location=("place", "prime_polynomial"),
                code="function_field.finite_place_polynomial",
                message="a finite place requires its prime polynomial",
            )
        factors = _factor_polynomial(admitted_polynomial)
        if factors != ((admitted_polynomial, 1),):
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
    place_polynomial = place.prime_polynomial
    if place_polynomial is None:
        raise OperationDomainValidationError(
            location=("place", "prime_polynomial"),
            code="function_field.finite_place_polynomial",
            message="a finite place requires its prime polynomial",
        )
    divisor = list(place_polynomial.coefficients)

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


def function_field_place_residue(
    place: FunctionFieldPlace, element: FiniteFunctionFieldElement
) -> FunctionFieldResidueResult:
    """Reduce a rational function regular at one rational-function-field place.

    At a finite place ``phi(x)``, the residue field is the exact quotient
    GF(p)[z]/(phi(z)) with ``z`` the residue of ``x``. At infinity the residue
    field is GF(p), and a regular function has residue zero or its leading
    coefficient ratio.
    """

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

    prime = place.field.characteristic
    coordinate = element.coordinates[0]
    rational = rf_normalize(
        coordinate.numerator.coefficients,
        coordinate.denominator.coefficients,
        prime,
    )
    # A valuation routine cannot express the zero rational function, whose
    # valuation is infinite.  Zero is regular at every place with residue
    # zero, so detect it before the finite order-division loop.
    if not any(rational[0]):
        valuation = 0
    else:
        valuation = _rf_valuation(
            _from_internal_rational_function(rational, prime), place
        )
    if valuation < 0:
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.residue_requires_integral_element",
            message="a residue is defined only for elements regular at the place",
        )

    if place.kind == "INFINITE":
        presentation = FiniteFieldPresentation.model_construct(
            characteristic=prime, modulus_coefficients=(0, 1), generator="z"
        )
        if len(rational[0]) < len(rational[1]):
            residue_value = 0
        else:
            residue_value = rational[0][-1] * pow(rational[1][-1], -1, prime) % prime
        residue = FiniteFieldElement(
            presentation=presentation, coordinates=(residue_value,)
        )
    else:
        phi = place.prime_polynomial
        if phi is None:
            raise OperationDomainValidationError(
                location=("place", "prime_polynomial"),
                code="function_field.finite_place_polynomial",
                message="a finite place requires its prime polynomial",
            )
        degree = phi.degree
        # The existing finite-field carrier is bounded by order 65536.
        if prime**degree > 65_536:
            raise OperationResourceAdmissionError(
                location=("place",),
                code="function_field.residue_field_order_exceeds_envelope",
                message="the residue field exceeds the finite-field carrier order bound",
            )
        presentation = FiniteFieldPresentation.model_construct(
            characteristic=prime, modulus_coefficients=phi.coefficients, generator="z"
        )
        modulus = phi.coefficients
        numerator = poly_divmod(rational[0], modulus, prime)[1]
        denominator = poly_divmod(rational[1], modulus, prime)[1]
        if not denominator:
            raise ArithmeticError("regular function has zero residue denominator")
        inverse_denominator = poly_powmod(
            denominator, presentation.order - 2, modulus, prime
        )
        reduced = poly_divmod(
            poly_mul(numerator, inverse_denominator, prime), modulus, prime
        )[1]
        coordinates = (*reduced, *(0 for _ in range(degree - len(reduced))))
        residue = FiniteFieldElement(presentation=presentation, coordinates=coordinates)

    return FunctionFieldResidueResult(place=place, element=element, residue=residue)


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
    field: FiniteFunctionField,
    element: FiniteFunctionFieldElement,
) -> FunctionFieldPrincipalDivisorResult:
    field = _admit_rational_place_field(field)
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
    for index, term in enumerate(terms):
        multiplicity = getattr(term, "multiplicity", None)
        if (
            type(multiplicity) is int
            and multiplicity.bit_length() > MAX_DIVISOR_MULTIPLICITY_BITS
        ):
            raise OperationResourceAdmissionError(
                location=("divisor", "terms", index, "multiplicity"),
                code="function_field.divisor_multiplicity_exceeds_envelope",
                message=(
                    "divisor multiplicities may use at most "
                    f"{MAX_DIVISOR_MULTIPLICITY_BITS} bits"
                ),
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


def function_field_divisor_add(
    left: FunctionFieldDivisor, right: FunctionFieldDivisor
) -> FunctionFieldDivisor:
    """Add two finite divisors with the same exact field and place parents."""
    # Establish source identities before any prime-polynomial factorization.
    if not isinstance(left, FunctionFieldDivisor) or not isinstance(
        right, FunctionFieldDivisor
    ):
        raise OperationDomainValidationError(
            location=(),
            code="function_field.divisor_type",
            message="both operands must be function-field divisor values",
        )
    left_field = _validated_field(getattr(left, "field", None))
    right_field = _validated_field(getattr(right, "field", None))
    if left_field != right_field:
        raise OperationDomainValidationError(
            location=("right", "field"),
            code="function_field.parent_mismatch",
            message="divisors must belong to the same exact function field",
        )
    left = _admit_divisor(left)
    right = _admit_divisor(right)
    support: dict[str, tuple[FunctionFieldPlace, int]] = {}
    for term in (*left.terms, *right.terms):
        key = term.place.model_dump_json()
        previous = support.get(key)
        support[key] = (
            term.place,
            term.multiplicity + (previous[1] if previous else 0),
        )
    terms = tuple(
        FunctionFieldDivisorTerm(place=place, multiplicity=multiplicity)
        for place, multiplicity in sorted(
            support.values(), key=lambda item: item[0].model_dump_json()
        )
        if multiplicity
    )
    if len(terms) > 256:
        raise OperationResourceAdmissionError(
            location=("result", "terms"),
            code="function_field.divisor_support_exceeds_envelope",
            message="the combined divisor support exceeds 256 places",
        )
    if any(
        term.multiplicity.bit_length() > MAX_DIVISOR_MULTIPLICITY_BITS for term in terms
    ):
        raise OperationResourceAdmissionError(
            location=("result", "terms"),
            code="function_field.divisor_result_multiplicity_exceeds_envelope",
            message="the exact divisor sum exceeds the 4096-bit result envelope",
        )
    return FunctionFieldDivisor(field=left.field, terms=terms)


def function_field_divisor_negate(
    divisor: FunctionFieldDivisor,
) -> FunctionFieldDivisor:
    """Return the additive inverse of one finite divisor."""
    divisor = _admit_divisor(divisor)
    return FunctionFieldDivisor(
        field=divisor.field,
        terms=tuple(
            FunctionFieldDivisorTerm(place=t.place, multiplicity=-t.multiplicity)
            for t in divisor.terms
        ),
    )


def function_field_divisor_scale(
    divisor: FunctionFieldDivisor, scalar: int
) -> FunctionFieldDivisor:
    """Multiply divisor coefficients by a bounded exact integer."""
    if type(scalar) is not int:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="function_field.divisor_scalar_type",
            message="divisor scalar must be a strict integer",
        )
    if scalar.bit_length() > MAX_DIVISOR_MULTIPLICITY_BITS:
        raise OperationResourceAdmissionError(
            location=("scalar",),
            code="function_field.divisor_scalar_exceeds_envelope",
            message=f"divisor scalars may use at most {MAX_DIVISOR_MULTIPLICITY_BITS} bits",
        )
    divisor = _admit_divisor(divisor)
    terms = tuple(
        FunctionFieldDivisorTerm(place=t.place, multiplicity=t.multiplicity * scalar)
        for t in divisor.terms
        if t.multiplicity * scalar
    )
    if any(
        term.multiplicity.bit_length() > MAX_DIVISOR_MULTIPLICITY_BITS for term in terms
    ):
        raise OperationResourceAdmissionError(
            location=("result", "terms"),
            code="function_field.divisor_result_multiplicity_exceeds_envelope",
            message="the exact scaled divisor exceeds the 4096-bit result envelope",
        )
    return FunctionFieldDivisor(field=divisor.field, terms=terms)


def function_field_divisor_degree(
    divisor: FunctionFieldDivisor,
) -> FunctionFieldDivisorDegreeResult:
    divisor = _admit_divisor(divisor)
    return FunctionFieldDivisorDegreeResult(divisor=divisor, degree=divisor.degree)


def function_field_divisor_effective_parts(
    divisor: FunctionFieldDivisor,
) -> FunctionFieldDivisorEffectivePartsResult:
    """Return the unique coefficientwise positive and negative parts of D."""

    # Bound the only copied scalar data before field/place admission can reach
    # polynomial factorization. A caller can supply native model_construct
    # values, so neither Pydantic parsing nor wire-level limits are sufficient.
    if not isinstance(divisor, FunctionFieldDivisor):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.divisor_type",
            message="divisor must be a function-field divisor value",
        )
    terms = getattr(divisor, "terms", None)
    if type(terms) is not tuple or len(terms) > 256:
        raise OperationDomainValidationError(
            location=("divisor", "terms"),
            code="function_field.divisor_shape",
            message="divisor terms must be a bounded canonical tuple",
        )
    for index, term in enumerate(terms):
        if isinstance(term, FunctionFieldDivisorTerm):
            multiplicity = getattr(term, "multiplicity", None)
            if type(multiplicity) is int and multiplicity.bit_length() > (
                MAX_DIVISOR_MULTIPLICITY_BITS
            ):
                raise OperationResourceAdmissionError(
                    location=("divisor", "terms", index, "multiplicity"),
                    code="function_field.divisor_multiplicity_exceeds_envelope",
                    message=(
                        "divisor multiplicities may use at most "
                        f"{MAX_DIVISOR_MULTIPLICITY_BITS} bits"
                    ),
                )
    admitted = _admit_divisor(divisor)
    positive: list[FunctionFieldDivisorTerm] = []
    negative: list[FunctionFieldDivisorTerm] = []
    for term in admitted.terms:
        if term.multiplicity > 0:
            positive.append(term)
        else:
            negative.append(
                FunctionFieldDivisorTerm(
                    place=term.place, multiplicity=-term.multiplicity
                )
            )

    # The input support is unique. Sorting by the structural serialization
    # makes outputs independent of the caller's presentation order.
    def place_key(term: FunctionFieldDivisorTerm) -> str:
        return term.place.model_dump_json()

    positive.sort(key=place_key)
    negative.sort(key=place_key)
    positive_divisor = FunctionFieldDivisor(field=admitted.field, terms=tuple(positive))
    negative_divisor = FunctionFieldDivisor(field=admitted.field, terms=tuple(negative))
    return FunctionFieldDivisorEffectivePartsResult(
        divisor=admitted,
        positive_part=positive_divisor,
        negative_part=negative_divisor,
    )


def function_field_genus(field: FiniteFunctionField) -> FunctionFieldGenusResult:
    """Return genus for GF(p)(x) or a squarefree odd-characteristic hyperelliptic model."""

    field = _validated_field(field)
    # The shape and work envelope are admitted before recognizing either
    # supported model. A squarefree branch polynomial proves irreducibility
    # and separability for the accepted hyperelliptic family below.
    _admit_field_resources(field)
    rational_field = len(field.defining_polynomial) == 1 and (
        field.defining_polynomial[0].numerator.is_one()
        and field.defining_polynomial[0].denominator.is_one()
    )
    if rational_field:
        return FunctionFieldGenusResult(field=field, genus=0)

    polynomial = _hyperelliptic_branch_polynomial(field)
    if polynomial is None:
        raise OperationDomainValidationError(
            location=("field", "defining_polynomial"),
            code="function_field.genus_requires_supported_model",
            message=(
                "genus is supported for GF(p)(x), or for an odd-characteristic "
                "quadratic extension y^2=f(x) with squarefree polynomial f "
                "of degree 3 through the admitted polynomial bound"
            ),
        )
    return FunctionFieldGenusResult(field=field, genus=(len(polynomial) - 2) // 2)


def _hyperelliptic_branch_polynomial(
    field: FiniteFunctionField,
) -> tuple[int, ...] | None:
    """Recognize ``y^2=f(x)`` with squarefree nonconstant polynomial ``f``.

    For odd characteristic such an f has a simple zero over the algebraic
    closure, so it is not a square in the rational function field. This makes
    the quadratic extension geometrically integral. The smooth projective
    model is the degree-two cover of P1 branched at the roots of f and, when
    deg(f) is odd, infinity; Riemann-Hurwitz gives floor((deg(f)-1)/2).
    """

    prime = field.characteristic
    if prime == 2 or field.degree != 2:
        return None
    defining = _field_kpoly(_canonical_field(field))
    if len(defining) != 3 or defining[1] != ZERO_RF:
        return None
    constant = defining[0]
    if constant[1] != ONE_POLY or not constant[0]:
        return None
    branch = tuple((-coefficient) % prime for coefficient in constant[0])
    while branch and branch[-1] == 0:
        branch = branch[:-1]
    if not 3 <= len(branch) - 1 <= MAX_POLYNOMIAL_X_DEGREE:
        return None
    if poly_gcd(branch, poly_derivative(branch, prime), prime) != ONE_POLY:
        return None
    return branch


def _preflight_riemann_roch_input(
    divisor: FunctionFieldDivisor,
) -> tuple[FiniteFunctionField, tuple[FunctionFieldDivisorTerm, ...]]:
    if not isinstance(divisor, FunctionFieldDivisor):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.divisor_type",
            message="divisor must be a function-field divisor value",
        )
    field = _validated_field(getattr(divisor, "field", None))
    if not _is_prime(field.characteristic):
        raise OperationDomainValidationError(
            location=("divisor", "field", "characteristic"),
            code="function_field.characteristic_not_prime",
            message="the constant field characteristic must be prime",
        )
    rational_field = len(field.defining_polynomial) == 1 and (
        field.defining_polynomial[0].numerator.is_one()
        and field.defining_polynomial[0].denominator.is_one()
    )
    if not rational_field:
        raise OperationDomainValidationError(
            location=("divisor", "field", "defining_polynomial"),
            code="function_field.riemann_roch_requires_rational_field",
            message=(
                "Riemann-Roch spaces are currently supported only for "
                "divisors over the rational function field GF(p)(x)"
            ),
        )
    terms = getattr(divisor, "terms", None)
    if type(terms) is not tuple or len(terms) > 256:
        raise OperationDomainValidationError(
            location=("divisor", "terms"),
            code="function_field.divisor_shape",
            message="divisor terms must be a bounded canonical tuple",
        )
    return field, terms


def _preflight_riemann_roch_profile(
    field: FiniteFunctionField,
    terms: tuple[FunctionFieldDivisorTerm, ...],
) -> tuple[int, int, int, int]:
    finite_positive_degree = 0
    finite_negative_degree = 0
    infinity_multiplicity = 0
    positive_x_multiplicity = 0
    for index, term in enumerate(terms):
        location = ("divisor", "terms", index)
        if not isinstance(term, FunctionFieldDivisorTerm):
            raise OperationDomainValidationError(
                location=location,
                code="function_field.divisor_term_type",
                message="divisor terms must be typed place/multiplicity values",
            )
        multiplicity = getattr(term, "multiplicity", None)
        if type(multiplicity) is not int or multiplicity == 0:
            raise OperationDomainValidationError(
                location=(*location, "multiplicity"),
                code="function_field.divisor_multiplicity",
                message="divisor multiplicities must be nonzero strict integers",
            )
        if multiplicity.bit_length() > MAX_DIVISOR_MULTIPLICITY_BITS:
            raise OperationResourceAdmissionError(
                location=(*location, "multiplicity"),
                code="function_field.riemann_roch_multiplicity_exceeds_envelope",
                message=(
                    "Riemann-Roch divisor multiplicities may use at most "
                    f"{MAX_DIVISOR_MULTIPLICITY_BITS} bits"
                ),
            )
        place = _canonical_place(getattr(term, "place", None))
        if place.field != field:
            raise OperationDomainValidationError(
                location=(*location, "place", "field"),
                code="function_field.divisor_parent",
                message="every divisor place must belong to divisor.field",
            )
        if place.kind == "INFINITE":
            infinity_multiplicity = multiplicity
            continue
        prime_polynomial = place.prime_polynomial
        if prime_polynomial is None:
            raise OperationDomainValidationError(
                location=(*location, "place", "prime_polynomial"),
                code="function_field.finite_place_polynomial",
                message="a finite place requires its prime polynomial",
            )
        contribution = multiplicity * prime_polynomial.degree
        if contribution > 0:
            finite_positive_degree += contribution
            # The place contract permits any prime associate, so recognize the
            # x place by its monic representative, matching _admit_divisor.
            if _monic_polynomial(prime_polynomial).coefficients == (0, 1):
                positive_x_multiplicity = multiplicity
        else:
            finite_negative_degree -= contribution
    return (
        finite_positive_degree,
        finite_negative_degree,
        infinity_multiplicity,
        positive_x_multiplicity,
    )


def _preflight_riemann_roch_divisor(
    divisor: FunctionFieldDivisor,
) -> tuple[FiniteFunctionField, int, int, int, int, int]:
    """Bound scalar and basis growth before any finite-place factorization."""

    field, terms = _preflight_riemann_roch_input(divisor)
    (
        finite_positive_degree,
        finite_negative_degree,
        infinity_multiplicity,
        positive_x_multiplicity,
    ) = _preflight_riemann_roch_profile(field, terms)

    divisor_degree = (
        finite_positive_degree - finite_negative_degree + infinity_multiplicity
    )
    if divisor_degree < 0:
        return (
            field,
            divisor_degree,
            0,
            finite_positive_degree,
            finite_negative_degree,
            positive_x_multiplicity,
        )

    dimension = divisor_degree + 1
    if dimension > MAX_RIEMANN_ROCH_BASIS_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("divisor",),
            code="function_field.riemann_roch_basis_exceeds_envelope",
            message=(
                "the exact Riemann-Roch basis exceeds the admitted "
                f"dimension {MAX_RIEMANN_ROCH_BASIS_DIMENSION}"
            ),
        )
    maximum_numerator_degree = (
        finite_negative_degree
        + divisor_degree
        - min(divisor_degree, positive_x_multiplicity)
    )
    estimated_product_work = (finite_positive_degree + finite_negative_degree) * (
        MAX_POLYNOMIAL_X_DEGREE + 1
    ) ** 2
    if (
        finite_positive_degree > MAX_POLYNOMIAL_X_DEGREE
        or finite_negative_degree > MAX_POLYNOMIAL_X_DEGREE
        or maximum_numerator_degree > MAX_POLYNOMIAL_X_DEGREE
        or estimated_product_work > MAX_RIEMANN_ROCH_CONSTRUCTION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("divisor",),
            code="function_field.riemann_roch_output_exceeds_envelope",
            message=(
                "the canonical rational-function basis exceeds the "
                f"degree-{MAX_POLYNOMIAL_X_DEGREE} coefficient or work envelope"
            ),
        )
    return (
        field,
        divisor_degree,
        dimension,
        finite_positive_degree,
        finite_negative_degree,
        positive_x_multiplicity,
    )


def _multiply_prime_field_polynomials(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    product = [0] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            degree = left_degree + right_degree
            product[degree] = (
                product[degree] + left_coefficient * right_coefficient
            ) % prime
    while len(product) > 1 and product[-1] == 0:
        product.pop()
    return tuple(product)


def function_field_riemann_roch_space(
    divisor: FunctionFieldDivisor,
) -> FunctionFieldRiemannRochSpace:
    """Return a canonical exact basis of L(D) for the rational field GF(p)(x)."""

    (
        field,
        admitted_degree,
        admitted_dimension,
        admitted_positive_degree,
        admitted_negative_degree,
        admitted_positive_x_multiplicity,
    ) = _preflight_riemann_roch_divisor(divisor)
    divisor = _admit_divisor(divisor)
    degree = divisor.degree
    if degree != admitted_degree:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.invalid_divisor_degree",
            message="canonical divisor support changed its admitted degree",
        )
    if admitted_dimension == 0:
        return FunctionFieldRiemannRochSpace(divisor=divisor, dimension=0, basis=())

    prime = field.characteristic
    numerator_factor: tuple[int, ...] = (1,)
    denominator_factor: tuple[int, ...] = (1,)
    positive_x_multiplicity = 0
    ordered_terms = sorted(
        divisor.terms,
        key=lambda term: (
            term.place.prime_polynomial.coefficients
            if term.place.prime_polynomial is not None
            else ()
        ),
    )
    for term in ordered_terms:
        polynomial = term.place.prime_polynomial
        if polynomial is None:
            continue
        factor = polynomial.coefficients
        if term.multiplicity > 0:
            if factor == (0, 1):
                positive_x_multiplicity = term.multiplicity
            for _ in range(term.multiplicity):
                numerator_factor = _multiply_prime_field_polynomials(
                    numerator_factor, factor, prime
                )
        else:
            for _ in range(-term.multiplicity):
                denominator_factor = _multiply_prime_field_polynomials(
                    denominator_factor, factor, prime
                )
    if (
        len(numerator_factor) - 1 != admitted_positive_degree
        or len(denominator_factor) - 1 != admitted_negative_degree
        or positive_x_multiplicity != admitted_positive_x_multiplicity
    ):
        raise OperationDomainValidationError(
            location=("divisor",),
            code="function_field.invalid_divisor_profile",
            message="canonical finite support changed its admitted degree profile",
        )

    basis: list[FiniteFunctionFieldElement] = []
    for exponent in range(degree + 1):
        cancelled_x_power = min(exponent, positive_x_multiplicity)
        numerator = (0,) * (exponent - cancelled_x_power) + denominator_factor
        denominator = numerator_factor[cancelled_x_power:]
        if not denominator:
            denominator = (1,)
        rational_function = PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(
                characteristic=prime, coefficients=numerator
            ),
            denominator=PrimeFieldPolynomial(
                characteristic=prime, coefficients=denominator
            ),
        )
        basis.append(
            FiniteFunctionFieldElement(field=field, coordinates=(rational_function,))
        )
    return FunctionFieldRiemannRochSpace(
        divisor=divisor,
        dimension=admitted_dimension,
        basis=tuple(basis),
    )


__all__ = [
    "_element_inverse",
    "function_field_base_embedding",
    "function_field_base_embedding_apply",
    "function_field_divisor_degree",
    "function_field_divisor_effective_parts",
    "function_field_element_add",
    "function_field_element_inverse",
    "function_field_element_multiply",
    "function_field_genus",
    "function_field_place_valuation",
    "function_field_principal_divisor",
    "function_field_rational_places_degree_bounded",
]
