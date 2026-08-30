"""Exact local real-root certification for rational polynomial systems."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.matrices.values import rational_matrix_from_fractions
from jacobian.math.polynomials.intervals._kernel import natural_interval_extension
from jacobian.math.polynomials.intervals._models import (
    _require_enclosure_preflight,
)
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.maps.operations import jacobian_matrix
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.values import RationalPolynomial

from ._kernel import (
    ComponentExclusionKernelResult,
    KrawczykKernelData,
    KrawczykKernelResult,
    MidpointKernelData,
    RootBoxKernelBudgetError,
    SingularMidpointKernelResult,
    certify_root_box_kernel,
)
from ._models import (
    MAX_ROOT_BOX_AGGREGATE_TERMS,
    MAX_ROOT_BOX_COMPONENT_TERMS,
    MAX_ROOT_BOX_DIMENSION,
    MAX_ROOT_BOX_ENCLOSURE_DIGITS,
    MAX_ROOT_BOX_ENDPOINT_DIGITS,
    MAX_ROOT_BOX_INTERMEDIATE_DIGITS,
    MAX_ROOT_BOX_POINT_VALUE_DIGITS,
    MAX_ROOT_BOX_RESULT_BYTES,
    MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS,
    MAX_ROOT_BOX_SOURCE_BYTES,
    MAX_ROOT_BOX_TOTAL_DEGREE,
    PolynomialSystemRootBoxResult,
    RootBoxCertifiedUniqueNonsingular,
    RootBoxComponentExclusion,
    RootBoxConclusion,
    RootBoxInconclusiveKrawczykAttempt,
    RootBoxJacobianEnclosure,
    RootBoxKrawczykDisjointness,
    RootBoxKrawczykEvidence,
    RootBoxMidpointData,
    RootBoxNoRoot,
    RootBoxSingularMidpointAttempt,
    RootBoxUnknown,
)

_RESULT_RATIONAL_COMPONENTS = (
    4 * MAX_ROOT_BOX_DIMENSION * MAX_ROOT_BOX_DIMENSION + 4 * MAX_ROOT_BOX_DIMENSION
)
_RESULT_ENVELOPE_RESERVE_BYTES = 8_192


type _PreparedRootBox = ComponentExclusionKernelResult | MidpointKernelData


def _domain_error(message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("polynomial_map", "box"),
        code="polynomial.root_box_admission",
        message=message,
    )


def _total_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )


def _source_payload(
    polynomial_map: RationalPolynomialMap, box: RationalBox
) -> dict[str, object]:
    return {
        "polynomial_map": polynomial_map.model_dump(mode="json"),
        "box": box.model_dump(mode="json"),
    }


def _center_values(box: RationalBox) -> tuple[Fraction, ...]:
    return tuple(
        (interval.lower.as_fraction() + interval.upper.as_fraction()) / 2
        for interval in box.intervals
    )


def _point_box(box: RationalBox, point: tuple[Fraction, ...]) -> RationalBox:
    return RationalBox(
        domain=box.domain,
        variables=box.variables,
        intervals=tuple(
            ClosedRationalInterval(
                lower=CanonicalRational.from_fraction(value),
                upper=CanonicalRational.from_fraction(value),
            )
            for value in point
        ),
    )


def _admit_source_shape(
    polynomial_map: RationalPolynomialMap, box: RationalBox
) -> None:
    order = len(polynomial_map.input_variables)
    if order > MAX_ROOT_BOX_DIMENSION:
        raise _domain_error(
            f"root-box systems are limited to dimension {MAX_ROOT_BOX_DIMENSION}"
        )
    if len(polynomial_map.output_polynomials) != order:
        raise _domain_error("root-box certification requires a square system")
    if box.domain != "QQ" or box.variables != polynomial_map.input_variables:
        raise _domain_error(
            "root box must use the polynomial system's complete ordered axis and QQ parent"
        )
    aggregate_terms = sum(
        len(polynomial.polynomial.terms)
        for polynomial in polynomial_map.output_polynomials
    )
    if aggregate_terms > MAX_ROOT_BOX_AGGREGATE_TERMS:
        raise _domain_error(
            "root-box system exceeds the "
            f"{MAX_ROOT_BOX_AGGREGATE_TERMS}-term aggregate budget"
        )
    for polynomial in polynomial_map.output_polynomials:
        if len(polynomial.polynomial.terms) > MAX_ROOT_BOX_COMPONENT_TERMS:
            raise _domain_error(
                "root-box component exceeds the "
                f"{MAX_ROOT_BOX_COMPONENT_TERMS}-term budget"
            )
        if _total_degree(polynomial) > MAX_ROOT_BOX_TOTAL_DEGREE:
            raise _domain_error(
                f"root-box component exceeds total degree {MAX_ROOT_BOX_TOTAL_DEGREE}"
            )
    for variable, interval in zip(box.variables, box.intervals, strict=True):
        for endpoint in (interval.lower, interval.upper):
            require_bounded_rational(
                endpoint,
                max_digits=MAX_ROOT_BOX_ENDPOINT_DIGITS,
                label=f"root-box {variable} endpoint",
            )
    source_bytes = len(encode_strict_json(_source_payload(polynomial_map, box)))
    if source_bytes > MAX_ROOT_BOX_SOURCE_BYTES:
        raise _domain_error(
            "root-box retained source exceeds the "
            f"{MAX_ROOT_BOX_SOURCE_BYTES:,}-byte budget"
        )
    maximum_result_bytes = (
        source_bytes
        + _RESULT_RATIONAL_COMPONENTS * (2 * MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS + 64)
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    if maximum_result_bytes > MAX_ROOT_BOX_RESULT_BYTES:
        raise _domain_error(
            "root-box exact evidence would exceed the canonical output bound"
        )


def _admitted_enclosure(
    polynomial: RationalPolynomial,
    box: RationalBox,
    *,
    label: str,
    result_digits: int,
) -> tuple[Fraction, Fraction]:
    growth = _require_enclosure_preflight(polynomial, box)
    if max(growth.result_numerator_digits, growth.result_denominator_digits) > (
        result_digits
    ):
        raise _domain_error(
            f"{label} interval enclosure exceeds the "
            f"{result_digits:,}-digit result bound"
        )
    if growth.intermediate_digits > MAX_ROOT_BOX_INTERMEDIATE_DIGITS:
        raise _domain_error(
            f"{label} interval enclosure exceeds the "
            f"{MAX_ROOT_BOX_INTERMEDIATE_DIGITS:,}-digit intermediate bound"
        )
    enclosure = natural_interval_extension(polynomial, box)
    return enclosure.lower.as_fraction(), enclosure.upper.as_fraction()


def _bounded_component_exclusion(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> ComponentExclusionKernelResult | None:
    """Return any cheap exact component exclusion without making it mandatory."""

    for component_index, polynomial in enumerate(polynomial_map.output_polynomials):
        try:
            enclosure = _admitted_enclosure(
                polynomial,
                box,
                label="system component",
                result_digits=MAX_ROOT_BOX_ENCLOSURE_DIGITS,
            )
        except OperationDomainValidationError:
            # A component range is only a presolve. The Krawczyk attempt does
            # not need F(X), so an expensive range must not reject an otherwise
            # admissible midpoint/Jacobian computation.
            continue
        if enclosure[1] < 0 or enclosure[0] > 0:
            return ComponentExclusionKernelResult(
                status="NO_ROOT_COMPONENT",
                component_index=component_index,
                enclosure=enclosure,
            )
    return None


def _fraction_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _sum_height_bound(count: int, term_digits: int) -> int:
    # The extra digit also covers the checked-add guard before cancellation.
    return max(1, count * term_digits + len(str(count)) + 1)


def _require_stage_height(bound: int, *, label: str) -> None:
    if bound > MAX_ROOT_BOX_INTERMEDIATE_DIGITS:
        raise _domain_error(
            f"source-derived {label} exceeds the "
            f"{MAX_ROOT_BOX_INTERMEDIATE_DIGITS:,}-digit intermediate bound"
        )


def _require_krawczyk_height_envelope(
    data: MidpointKernelData,
    box: RationalBox,
) -> None:
    """Prove the complete exact Krawczyk arithmetic envelope before FLINT."""

    order = len(data.jacobian_at_center)
    entry_digits = max(
        _fraction_digits(value) for row in data.jacobian_at_center for value in row
    )
    # Clear denominators row by row. Each determinant term then has at most
    # order squared entry digits; multiplying a cofactor by its row denominator
    # has the same bound. The factorial term accounts for determinant sums.
    inverse_digits = (
        order * (order + 1) * entry_digits
        + len(format_canonical_integer(factorial(order)))
        + 1
    )
    if inverse_digits > MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS:
        raise _domain_error(
            "exact midpoint-Jacobian inverse exceeds the "
            f"{MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS:,}-digit result bound"
        )

    zero_value = all(value == 0 for value in data.value_at_center)
    constant_jacobian = all(
        enclosure == (data.jacobian_at_center[row][column],) * 2
        for row, entries in enumerate(data.jacobian_enclosure)
        for column, enclosure in enumerate(entries)
    )
    center_digits = max(_fraction_digits(value) for value in data.center)

    if zero_value:
        residual_digits = center_digits
    else:
        value_digits = max(_fraction_digits(value) for value in data.value_at_center)
        correction_product_digits = inverse_digits + value_digits
        _require_stage_height(
            correction_product_digits,
            label="preconditioner-value product",
        )
        correction_digits = _sum_height_bound(
            order,
            correction_product_digits,
        )
        _require_stage_height(
            correction_digits,
            label="preconditioner-value sum",
        )
        residual_digits = center_digits + correction_digits + 1
        _require_stage_height(residual_digits, label="Krawczyk residual center")

    if constant_jacobian:
        image_digits = residual_digits
    else:
        enclosure_digits = max(
            _fraction_digits(endpoint)
            for row in data.jacobian_enclosure
            for enclosure in row
            for endpoint in enclosure
        )
        matrix_product_term_digits = inverse_digits + enclosure_digits
        _require_stage_height(
            matrix_product_term_digits,
            label="preconditioner-Jacobian product",
        )
        matrix_product_digits = _sum_height_bound(
            order,
            matrix_product_term_digits,
        )
        _require_stage_height(
            matrix_product_digits,
            label="preconditioner-Jacobian sum",
        )
        remainder_matrix_digits = matrix_product_digits + 2
        _require_stage_height(
            remainder_matrix_digits,
            label="Krawczyk remainder matrix",
        )
        centered_box_digits = max(
            _fraction_digits(endpoint.as_fraction() - center)
            for interval, center in zip(box.intervals, data.center, strict=True)
            for endpoint in (interval.lower, interval.upper)
        )
        remainder_product_digits = remainder_matrix_digits + centered_box_digits
        _require_stage_height(
            remainder_product_digits,
            label="remainder-matrix box product",
        )
        remainder_digits = _sum_height_bound(
            order,
            remainder_product_digits,
        )
        _require_stage_height(remainder_digits, label="Krawczyk remainder sum")
        image_digits = residual_digits + remainder_digits + 1
        _require_stage_height(image_digits, label="Krawczyk image")

    if image_digits > MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS:
        raise _domain_error(
            "source-derived Krawczyk image exceeds the "
            f"{MAX_ROOT_BOX_RESULT_COMPONENT_DIGITS:,}-digit result bound"
        )


def _prepare_root_box(
    polynomial_map: RationalPolynomialMap, box: RationalBox
) -> _PreparedRootBox:
    try:
        _admit_source_shape(polynomial_map, box)
        component_exclusion = _bounded_component_exclusion(polynomial_map, box)
        if component_exclusion is not None:
            return component_exclusion

        jacobian = jacobian_matrix(polynomial_map)
        center = _center_values(box)
        center_box = _point_box(box, center)
        value_at_center = tuple(
            _admitted_enclosure(
                polynomial,
                center_box,
                label="system component midpoint value",
                result_digits=MAX_ROOT_BOX_POINT_VALUE_DIGITS,
            )[0]
            for polynomial in polynomial_map.output_polynomials
        )
        jacobian_at_center_flat = tuple(
            _admitted_enclosure(
                polynomial,
                center_box,
                label="Jacobian midpoint value",
                result_digits=MAX_ROOT_BOX_POINT_VALUE_DIGITS,
            )[0]
            for polynomial in jacobian.entries
        )
        jacobian_enclosure_flat = tuple(
            _admitted_enclosure(
                polynomial,
                box,
                label="Jacobian entry",
                result_digits=MAX_ROOT_BOX_ENCLOSURE_DIGITS,
            )
            for polynomial in jacobian.entries
        )
        order = len(polynomial_map.input_variables)
        jacobian_at_center = tuple(
            jacobian_at_center_flat[row * order : (row + 1) * order]
            for row in range(order)
        )
        prepared = MidpointKernelData(
            center=center,
            value_at_center=value_at_center,
            jacobian_at_center=jacobian_at_center,
            jacobian_enclosure=tuple(
                jacobian_enclosure_flat[row * order : (row + 1) * order]
                for row in range(order)
            ),
        )
        _require_krawczyk_height_envelope(prepared, box)
        return prepared
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise _domain_error(exc.message()) from exc
    except ValueError as exc:
        raise _domain_error(str(exc)) from exc


def _wire_interval(interval: tuple[Fraction, Fraction]) -> ClosedRationalInterval:
    return ClosedRationalInterval(
        lower=CanonicalRational.from_fraction(interval[0]),
        upper=CanonicalRational.from_fraction(interval[1]),
    )


def _wire_midpoint_data(
    variables: tuple[str, ...],
    data: MidpointKernelData,
) -> RootBoxMidpointData:
    return RootBoxMidpointData(
        center=VariablePoint(
            variables=variables,
            values=tuple(
                CanonicalRational.from_fraction(value) for value in data.center
            ),
        ),
        value_at_center=tuple(
            CanonicalRational.from_fraction(value) for value in data.value_at_center
        ),
        jacobian_at_center=rational_matrix_from_fractions(data.jacobian_at_center),
        jacobian_enclosure=RootBoxJacobianEnclosure(
            entries=tuple(
                tuple(_wire_interval(interval) for interval in row)
                for row in data.jacobian_enclosure
            )
        ),
    )


def _wire_krawczyk_evidence(
    variables: tuple[str, ...],
    data: KrawczykKernelData,
) -> RootBoxKrawczykEvidence:
    midpoint = _wire_midpoint_data(variables, data)
    return RootBoxKrawczykEvidence(
        center=midpoint.center,
        value_at_center=midpoint.value_at_center,
        jacobian_at_center=midpoint.jacobian_at_center,
        jacobian_enclosure=midpoint.jacobian_enclosure,
        preconditioner=rational_matrix_from_fractions(data.preconditioner),
        krawczyk_image=RationalBox(
            variables=variables,
            intervals=tuple(
                _wire_interval(interval) for interval in data.krawczyk_image
            ),
        ),
    )


def _run_request(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> PolynomialSystemRootBoxResult:
    prepared = _prepare_root_box(polynomial_map, box)
    try:
        kernel_result = certify_root_box_kernel(
            box,
            prepared,
        )
    except RootBoxKernelBudgetError as exc:
        raise OperationDomainValidationError(
            location=("polynomial_map", "box"),
            code="polynomial.root_box_intermediate_bound",
            message=str(exc),
        ) from exc

    variables = polynomial_map.input_variables
    conclusion: RootBoxConclusion
    if isinstance(kernel_result, ComponentExclusionKernelResult):
        conclusion = RootBoxNoRoot(
            evidence=RootBoxComponentExclusion(
                component_index=kernel_result.component_index,
                enclosure=_wire_interval(kernel_result.enclosure),
            )
        )
    elif isinstance(kernel_result, SingularMidpointKernelResult):
        conclusion = RootBoxUnknown(
            attempt=RootBoxSingularMidpointAttempt(
                data=_wire_midpoint_data(variables, kernel_result.data)
            )
        )
    elif isinstance(kernel_result, KrawczykKernelResult):
        evidence = _wire_krawczyk_evidence(variables, kernel_result.evidence)
        if kernel_result.status == "CERTIFIED_UNIQUE_NONSINGULAR":
            conclusion = RootBoxCertifiedUniqueNonsingular(evidence=evidence)
        elif kernel_result.status == "NO_ROOT_KRAWCZYK":
            conclusion = RootBoxNoRoot(
                evidence=RootBoxKrawczykDisjointness(evidence=evidence)
            )
        else:
            conclusion = RootBoxUnknown(
                attempt=RootBoxInconclusiveKrawczykAttempt(evidence=evidence)
            )
    else:
        raise AssertionError("unknown root-box kernel outcome")
    return PolynomialSystemRootBoxResult._from_kernel(
        polynomial_map=polynomial_map,
        box=box,
        conclusion=conclusion,
    )


def certify_real_root_box(
    polynomial_map: RationalPolynomialMap,
    box: RationalBox,
) -> PolynomialSystemRootBoxResult:
    """Certify a unique nonsingular real zero, exclusion, or non-conclusion."""

    return _run_request(
        polynomial_map,
        box,
    )


__all__ = ["certify_real_root_box"]
