"""Bounded exact Newton transforms at supplied simple rational edge roots."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, lcm
from typing import NoReturn

from pydantic import Field, StrictInt

from jacobian._exact import (
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian._models import StrictModel
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.newton_polygon import (
    MAX_LOCAL_POLYNOMIAL_ROWS,
    MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS,
    MAX_NEWTON_POLYGON_SCALAR_DIGITS,
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
    NewtonEdgeCharacteristicRequest,
    NewtonEdgeCharacteristicResult,
    newton_edge_characteristic_polynomial,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)

MAX_NEWTON_TRANSFORM_WORK = 250_000
# The Newton transform retains the source slots plus its exact output
# rows, so admission counts allocated cells and retained digits. A
# deployment's wire ceiling must not bound a native mathematical result.
MAX_NEWTON_TRANSFORM_OUTPUT_CELLS = 1_000_000


class NewtonTransformRequest(NewtonEdgeCharacteristicRequest):
    """Apply the selected edge transform at one supplied rational root."""

    initial_root: CanonicalRational = Field(
        description="The caller-selected simple rational root of the edge polynomial."
    )


class NewtonTransformResult(StrictModel):
    """Source-bound polynomial after one normalized rational-root transform."""

    characteristic: NewtonEdgeCharacteristicResult
    initial_root: CanonicalRational
    ramification_index: StrictInt = Field(ge=1)
    ordinate_power: StrictInt
    removed_valuation: StrictInt
    transformed_polynomial: LocalPolynomialInSeries
    constant_term_valuation_lower_bound: StrictInt = Field(ge=1)


@dataclass(frozen=True)
class _Admission:
    characteristic: NewtonEdgeCharacteristicResult
    root: Fraction
    ramification_index: int
    ordinate_power: int
    removed_valuation: int
    output_precisions: tuple[int, ...]
    output_slots: int
    coefficient_digits: int


@dataclass(frozen=True)
class _TransformGeometry:
    characteristic: NewtonEdgeCharacteristicResult
    root: Fraction
    ramification_index: int
    ordinate_power: int
    removed_valuation: int
    source_common_denominator: int


@dataclass(frozen=True)
class _SourceRow:
    y_degree: int
    series: TruncatedLaurentWindow


def _resource(code: str, message: str, location: tuple[str | int, ...]) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"local_series.newton_transform_{code}",
        message=message,
    )


def _domain(code: str, message: str, location: tuple[str | int, ...]) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"local_series.newton_transform_{code}",
        message=message,
    )


def _fraction_digits(value: Fraction) -> int:
    return max(
        decimal_digit_width(value.numerator),
        decimal_digit_width(value.denominator),
    )


def _retained_fraction_digits(value: Fraction) -> int:
    """Width of both components, because the result retains each of them."""
    return decimal_digit_width(value.numerator) + decimal_digit_width(value.denominator)


def _retained_cells(value: object) -> int:
    """Count admitted native model/tuple slots and scalar component widths.

    This walks the already bounded characteristic without making a serialized
    copy. A repeated scalar is charged at each retained location, including
    zero numerators, unit denominators, and the characteristic's two copies of
    each leading coefficient.
    """
    if isinstance(value, CanonicalRational):
        return 2 + _retained_fraction_digits(value.as_fraction())
    if isinstance(value, StrictModel):
        fields = type(value).model_fields
        return len(fields) + sum(
            _retained_cells(getattr(value, name)) for name in fields
        )
    if isinstance(value, tuple):
        return len(value) + sum(_retained_cells(item) for item in value)
    if isinstance(value, int):
        return decimal_digit_width(value)
    if isinstance(value, str):
        return len(value)
    if value is None:
        return 1
    raise AssertionError("admitted Newton values must have canonical scalar shapes")


def _admit_retained_output(
    request: NewtonTransformRequest,
    geometry: _TransformGeometry,
    output_precisions: tuple[int, ...],
    output_slots: int,
    work: int,
    coefficient_digits: int,
) -> None:
    # Every transformed row has the fixed parent (u, FINITE, 0). Count its
    # structural slots and coordinate widths before allocating coefficients.
    parent_cells = len("u") + len("FINITE") + 4  # zero rational: two slots + 0/1
    output_metadata = (
        len(LocalPolynomialInSeries.model_fields)
        + parent_cells
        + len(output_precisions)
        + sum(
            len(LocalPolynomialCoefficient.model_fields)
            + decimal_digit_width(degree)
            + len(TruncatedLaurentWindow.model_fields)
            + parent_cells
            + decimal_digit_width(min(0, precision))
            + decimal_digit_width(precision)
            for degree, precision in enumerate(output_precisions)
        )
    )
    # Each rational occupies a tuple slot, two component slots, and at least
    # two digits. Only slots reached by a retained nonzero product can grow;
    # each such slot has two components of at most coefficient_digits digits.
    wide_slots = min(work, output_slots)
    output_coefficient_cells = 5 * output_slots + 2 * wide_slots * (
        coefficient_digits - 1
    )
    output_cells = (
        len(NewtonTransformResult.model_fields)
        + _retained_cells(geometry.characteristic)
        + _retained_cells(request.initial_root)
        + decimal_digit_width(geometry.ramification_index)
        + decimal_digit_width(geometry.ordinate_power)
        + decimal_digit_width(geometry.removed_valuation)
        + decimal_digit_width(max(1, output_precisions[0]))
        + output_metadata
        + output_coefficient_cells
    )
    if output_cells > MAX_NEWTON_TRANSFORM_OUTPUT_CELLS:
        _resource(
            "output_bound",
            "Newton transform source and exact output exceed their admitted cell bound",
            ("polynomial",),
        )


def _admit_edge_root_powers(
    characteristic: NewtonEdgeCharacteristicResult,
    root: Fraction,
    common_denominator: int,
) -> None:
    """Bound exact root powers before evaluating the edge polynomial."""
    largest_exponent = max(
        (
            term.exponents[0]
            for term in characteristic.characteristic_polynomial.polynomial.terms
        ),
        default=0,
    )

    def power_digits(value: int) -> int:
        if abs(value) == 1 or largest_exponent == 0:
            return 1
        # log10(2) < 30103/100000 gives a cheap integer-only upper bound.
        scaled_bits = abs(value).bit_length() * largest_exponent * 30_103
        return (scaled_bits + 99_999) // 100_000

    root_power_digits = max(
        power_digits(root.numerator), power_digits(root.denominator)
    )
    term_count_digits = decimal_digit_width(
        len(characteristic.characteristic_polynomial.polynomial.terms)
    )
    projected_digits = (
        root_power_digits
        + MAX_NEWTON_POLYGON_SCALAR_DIGITS
        + decimal_digit_width(common_denominator)
        + term_count_digits
        + decimal_digit_width(largest_exponent)
    )
    if projected_digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        _resource(
            "root_power_bound",
            "edge-root evaluation exceeds the admitted exact coefficient digit limit",
            ("initial_root",),
        )


def _source_rows(source: LocalPolynomialInSeries) -> tuple[_SourceRow, ...]:
    rows = []
    for row in source.coefficients:
        if row.series is not None:
            rows.append(_SourceRow(y_degree=row.y_degree, series=row.series))
    return tuple(rows)


def _admit_source_common_denominator(
    characteristic: NewtonEdgeCharacteristicResult,
) -> int:
    """Bound only coefficients used to evaluate the selected edge polynomial."""
    common_denominator = 1
    for term in characteristic.terms:
        common_denominator = lcm(
            common_denominator, term.leading_coefficient.as_fraction().denominator
        )
        if (
            decimal_digit_width(common_denominator)
            > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
        ):
            _resource(
                "coefficient_bound",
                "edge polynomial denominator exceeds the coefficient limit",
                ("polynomial", "coefficients"),
            )
    return common_denominator


def _edge_polynomial_value(
    characteristic: NewtonEdgeCharacteristicResult, root: Fraction
) -> tuple[Fraction, Fraction]:
    value = Fraction(0)
    derivative = Fraction(0)
    for term in characteristic.characteristic_polynomial.polynomial.terms:
        exponent = term.exponents[0]
        coefficient = term.coefficient.as_fraction()
        value += coefficient * root**exponent
        if exponent:
            derivative += exponent * coefficient * root ** (exponent - 1)
    return value, derivative


def _canonical_center(source: LocalPolynomialInSeries) -> Fraction:
    """Establish the source centre's canonical representation before use.

    A native caller can bypass Pydantic with ``model_construct``, so the
    reducedness, positive denominator, and strict-integer component
    invariants are re-established here. Reading ``as_fraction()`` first would
    leak a raw ``ZeroDivisionError`` from a malformed centre.
    """
    center = source.center
    center_num = getattr(center, "num", None)
    center_den = getattr(center, "den", None)
    if type(center_num) is not int or type(center_den) is not int:
        _domain(
            "center_canonical",
            "the local polynomial center must have strict integer components",
            ("polynomial", "center"),
        )
    if center_den <= 0:
        _domain(
            "center_canonical",
            "the local polynomial center must have a positive denominator",
            ("polynomial", "center"),
        )
    try:
        value = center.as_fraction()
    except (TypeError, ValueError, ZeroDivisionError):
        _domain(
            "center_canonical",
            "the local polynomial center must be a valid canonical rational",
            ("polynomial", "center"),
        )
    if (center_num, center_den) != (value.numerator, value.denominator):
        _domain(
            "center_canonical",
            "the local polynomial center must be reduced",
            ("polynomial", "center"),
        )
    return value


def _admit_geometry(request: NewtonTransformRequest) -> _TransformGeometry:
    if not isinstance(request, NewtonTransformRequest):
        _domain("request_type", "request must select an edge and rational root", ())
    if type(request.edge_index) is not int or request.edge_index < 0:
        _domain(
            "edge_index",
            "edge_index must be a nonnegative strict integer",
            ("edge_index",),
        )
    source = request.polynomial
    if not isinstance(source, LocalPolynomialInSeries):
        _domain(
            "polynomial_type",
            "polynomial must be a local series polynomial",
            ("polynomial",),
        )
    if source.place != "FINITE" or _canonical_center(source) != 0:
        _domain(
            "origin_only",
            "Newton transforms currently require a finite local parameter centered at zero",
            ("polynomial",),
        )
    if not isinstance(request.initial_root, CanonicalRational):
        _domain(
            "root_type", "initial_root must be a canonical rational", ("initial_root",)
        )
    # Native callers may bypass Pydantic with model_construct; validate the
    # canonical representation before any helper converts it to Fraction.
    root_value = request.initial_root
    canonical_root = None
    if (
        type(root_value.num) is int
        and type(root_value.den) is int
        and root_value.den > 0
    ):
        canonical_root = Fraction(root_value.num, root_value.den)
    if (
        canonical_root is None
        or canonical_root.numerator != root_value.num
        or canonical_root.denominator != root_value.den
    ):
        _domain(
            "root_canonical",
            "initial_root must have canonical integer numerator and positive denominator",
            ("initial_root",),
        )
    try:
        require_bounded_rational(
            root_value,
            max_digits=MAX_NEWTON_POLYGON_SCALAR_DIGITS,
            label="Newton initial root",
        )
    except ValueError as error:
        _resource("root_coefficient_bound", str(error), ("initial_root",))
    root = root_value.as_fraction()
    if root == 0:
        _domain(
            "zero_root",
            "the selected Newton edge root must be nonzero",
            ("initial_root",),
        )

    characteristic = newton_edge_characteristic_polynomial(
        NewtonEdgeCharacteristicRequest(
            polynomial=source,
            edge_index=request.edge_index,
        )
    )
    source_common_denominator = _admit_source_common_denominator(characteristic)
    _admit_edge_root_powers(characteristic, root, source_common_denominator)
    polynomial_value, polynomial_derivative = _edge_polynomial_value(
        characteristic, root
    )
    if polynomial_value != 0:
        _domain(
            "root_not_on_edge",
            "initial_root must be an exact root of the selected edge polynomial",
            ("initial_root",),
        )
    if polynomial_derivative == 0:
        _domain(
            "multiple_root",
            "the supplied rational edge root must be simple for this transform",
            ("initial_root",),
        )

    slope = characteristic.edge.slope.as_fraction()
    exponent = -slope
    if exponent < 0:
        _domain(
            "negative_ordinate_power",
            "the selected edge gives a negative ordinate power at the origin",
            ("characteristic", "edge", "slope"),
        )
    ordinate_power = exponent.numerator
    ramification_index = exponent.denominator
    if (
        abs(ordinate_power) > MAX_LOCAL_SERIES_EXPONENT
        or ramification_index > MAX_LOCAL_SERIES_EXPONENT
    ):
        _resource(
            "ramification_bound",
            "edge slope exceeds the local exponent and ramification limits",
            ("characteristic", "edge", "slope"),
        )

    edge = characteristic.edge
    removed_valuation = (
        ramification_index * edge.left.valuation + ordinate_power * edge.left.y_degree
    )
    if (
        ramification_index * edge.right.valuation + ordinate_power * edge.right.y_degree
        != removed_valuation
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    return _TransformGeometry(
        characteristic=characteristic,
        root=root,
        ramification_index=ramification_index,
        ordinate_power=ordinate_power,
        removed_valuation=removed_valuation,
        source_common_denominator=source_common_denominator,
    )


def _admit_output_windows(
    source_rows: tuple[_SourceRow, ...],
    largest_degree: int,
    geometry: _TransformGeometry,
) -> tuple[tuple[int, ...], int]:
    output_precisions = tuple(
        min(
            geometry.ramification_index * row.series.precision
            + geometry.ordinate_power * row.y_degree
            - geometry.removed_valuation
            for row in source_rows
            if row.y_degree >= output_degree
        )
        for output_degree in range(largest_degree + 1)
    )
    output_widths = tuple(max(0, precision) for precision in output_precisions)
    if any(
        precision < -MAX_LOCAL_SERIES_EXPONENT
        or precision > MAX_LOCAL_SERIES_EXPONENT
        or width > MAX_LOCAL_SERIES_TERMS
        for precision, width in zip(output_precisions, output_widths, strict=True)
    ):
        _resource(
            "precision_bound",
            "transformed coefficient precision exceeds the local-series window bound",
            ("polynomial", "coefficients"),
        )
    output_slots = sum(output_widths)
    if output_slots > MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS:
        _resource(
            "output_slots_bound",
            f"transformed polynomial exceeds {MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS} retained slots",
            ("polynomial", "coefficients"),
        )
    return output_precisions, output_slots


def _admit_coefficient_digits(
    source_rows: tuple[_SourceRow, ...],
    source_slots: int,
    root: Fraction,
    common_denominator: int,
    geometry: _TransformGeometry,
    output_precisions: tuple[int, ...],
) -> int:
    max_lifted_numerator_digits = 1
    largest_root_power = 0
    for row in source_rows:
        for offset, coefficient in enumerate(row.series.coefficients):
            value = coefficient.as_fraction()
            if not value:
                continue
            target_exponent = (
                geometry.ramification_index * (row.series.valuation_lower + offset)
                + geometry.ordinate_power * row.y_degree
                - geometry.removed_valuation
            )
            retained_degrees = tuple(
                degree
                for degree in range(row.y_degree + 1)
                if target_exponent < output_precisions[degree]
            )
            if not retained_degrees:
                continue
            max_lifted_numerator_digits = max(
                max_lifted_numerator_digits,
                decimal_digit_width(value.numerator)
                + decimal_digit_width(common_denominator // value.denominator),
            )
            largest_root_power = max(
                largest_root_power,
                *(row.y_degree - degree for degree in retained_degrees),
            )
    root_digits = _fraction_digits(root)
    term_digits = (
        max_lifted_numerator_digits
        + largest_root_power * root_digits
        + largest_root_power
        + 1
    )
    denominator_digits = decimal_digit_width(
        common_denominator
    ) + largest_root_power * (decimal_digit_width(root.denominator))
    coefficient_digits = max(
        term_digits + len(str(source_slots)) + 1,
        denominator_digits,
    )
    if coefficient_digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        _resource(
            "coefficient_bound",
            "predicted transformed rational coefficients exceed the local-series digit limit",
            ("initial_root",),
        )
    return coefficient_digits


def _admit_retained_common_denominator(
    source_rows: tuple[_SourceRow, ...],
    geometry: _TransformGeometry,
    output_precisions: tuple[int, ...],
) -> int:
    """Bound denominators of source coefficients that can reach retained slots."""
    common_denominator = 1
    for row in source_rows:
        for offset, coefficient in enumerate(row.series.coefficients):
            value = coefficient.as_fraction()
            if not value:
                continue
            exponent = (
                geometry.ramification_index * (row.series.valuation_lower + offset)
                + geometry.ordinate_power * row.y_degree
                - geometry.removed_valuation
            )
            if any(
                exponent < output_precisions[degree]
                for degree in range(row.y_degree + 1)
            ):
                common_denominator = lcm(common_denominator, value.denominator)
                if (
                    decimal_digit_width(common_denominator)
                    > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
                ):
                    _resource(
                        "coefficient_bound",
                        "retained source denominator exceeds the output coefficient limit",
                        ("polynomial", "coefficients"),
                    )
    return common_denominator


def _admit(request: NewtonTransformRequest) -> _Admission:
    geometry = _admit_geometry(request)

    source = geometry.characteristic.source
    source_rows = _source_rows(source)
    if not source_rows:
        _domain(
            "zero_polynomial",
            "the source polynomial has no nonzero coefficient",
            ("polynomial",),
        )
    source_slots = sum(len(row.series.coefficients) for row in source_rows)
    largest_degree = max(row.y_degree for row in source_rows)
    output_row_count = largest_degree + 1
    if output_row_count > MAX_LOCAL_POLYNOMIAL_ROWS:
        _resource(
            "output_rows_bound",
            f"Newton transform exceeds {MAX_LOCAL_POLYNOMIAL_ROWS} dense output rows",
            ("polynomial", "coefficients"),
        )

    coefficient_valuations: dict[int, int] = {}
    for row in source_rows:
        valuation = next(
            (
                row.series.valuation_lower + offset
                for offset, coefficient in enumerate(row.series.coefficients)
                if bool(coefficient.as_fraction())
            ),
            None,
        )
        if valuation is None:
            _domain(
                "unknown_coefficient_valuation",
                "nonzero coefficient prefixes must contain a known nonzero term",
                ("polynomial", "coefficients", row.y_degree),
            )
        coefficient_valuations[row.y_degree] = valuation
    minimum_weight = min(
        geometry.ramification_index * valuation + geometry.ordinate_power * degree
        for degree, valuation in coefficient_valuations.items()
    )
    if minimum_weight != geometry.removed_valuation:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    output_precisions, output_slots = _admit_output_windows(
        source_rows, largest_degree, geometry
    )
    retained_denominator = _admit_retained_common_denominator(
        source_rows, geometry, output_precisions
    )
    work = 0
    for row in source_rows:
        for offset, coefficient in enumerate(row.series.coefficients):
            if not coefficient.as_fraction():
                continue
            target_exponent = (
                geometry.ramification_index * (row.series.valuation_lower + offset)
                + geometry.ordinate_power * row.y_degree
                - geometry.removed_valuation
            )
            if target_exponent < 0:
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            retained_products = sum(
                target_exponent < output_precisions[degree]
                for degree in range(row.y_degree + 1)
            )
            if retained_products:
                work += retained_products
                if work > MAX_NEWTON_TRANSFORM_WORK:
                    _resource(
                        "work_bound",
                        f"Newton transform exceeds {MAX_NEWTON_TRANSFORM_WORK} retained source-term products",
                        ("polynomial", "coefficients"),
                    )
    coefficient_digits = _admit_coefficient_digits(
        source_rows,
        source_slots,
        geometry.root,
        retained_denominator,
        geometry,
        output_precisions,
    )

    _admit_retained_output(
        request,
        geometry,
        output_precisions,
        output_slots,
        work,
        coefficient_digits,
    )

    return _Admission(
        characteristic=geometry.characteristic,
        root=geometry.root,
        ramification_index=geometry.ramification_index,
        ordinate_power=geometry.ordinate_power,
        removed_valuation=geometry.removed_valuation,
        output_precisions=output_precisions,
        output_slots=output_slots,
        coefficient_digits=coefficient_digits,
    )


def newton_transform(request: NewtonTransformRequest) -> NewtonTransformResult:
    """Return ``u^-m F(u^e, u^p(c + z))`` for one simple rational edge root.

    The selected edge fixes ``q = p/e = -slope`` in lowest terms and ``m``
    is the least weight ``e*valuation(a_j) + p*j``. Input series tails remain
    unknown; the transformed coefficient windows stop before the first
    transported unknown input coefficient.
    """
    admitted = _admit(request)
    source = admitted.characteristic.source
    source_rows = _source_rows(source)
    largest_degree = max(row.y_degree for row in source_rows)
    accumulators: list[dict[int, Fraction]] = [{} for _ in range(largest_degree + 1)]
    root_powers = [Fraction(1)]
    for _ in range(largest_degree):
        root_powers.append(root_powers[-1] * admitted.root)

    for row in source_rows:
        degree = row.y_degree
        for offset, canonical_coefficient in enumerate(row.series.coefficients):
            coefficient = canonical_coefficient.as_fraction()
            if not coefficient:
                continue
            source_exponent = row.series.valuation_lower + offset
            target_exponent = (
                admitted.ramification_index * source_exponent
                + admitted.ordinate_power * degree
                - admitted.removed_valuation
            )
            if target_exponent < 0:
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            for transformed_degree in range(degree + 1):
                if target_exponent >= admitted.output_precisions[transformed_degree]:
                    continue
                contribution = (
                    coefficient
                    * comb(degree, transformed_degree)
                    * root_powers[degree - transformed_degree]
                )
                terms = accumulators[transformed_degree]
                terms[target_exponent] = (
                    terms.get(target_exponent, Fraction(0)) + contribution
                )

    transformed_rows: list[LocalPolynomialCoefficient] = []
    for degree, precision in enumerate(admitted.output_precisions):
        lower = min(0, precision)
        if precision < 0:
            lower = precision
        coefficients = tuple(
            CanonicalRational.from_fraction(
                accumulators[degree].get(exponent, Fraction(0))
            )
            for exponent in range(lower, precision)
        )
        transformed_rows.append(
            LocalPolynomialCoefficient(
                y_degree=degree,
                series=TruncatedLaurentWindow(
                    variable="u",
                    place="FINITE",
                    center=CanonicalRational(num=0, den=1),
                    valuation_lower=lower,
                    precision=precision,
                    coefficients=coefficients,
                ),
            )
        )
    transformed = LocalPolynomialInSeries(
        variable="u",
        place="FINITE",
        center=CanonicalRational(num=0, den=1),
        coefficients=tuple(transformed_rows),
    )
    constant_term = transformed.coefficients[0].series
    if (
        constant_term is None
        or constant_term.valuation_lower > 0
        or constant_term.precision < 1
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    if constant_term.coefficients[0].as_fraction() != 0:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    linear_term = transformed.coefficients[1].series
    if (
        linear_term is None
        or linear_term.valuation_lower > 0
        or linear_term.precision < 1
        or linear_term.coefficients[0].as_fraction() == 0
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    constant_valuation_lower_bound = next(
        (
            constant_term.valuation_lower + offset
            for offset, coefficient in enumerate(constant_term.coefficients)
            if coefficient.as_fraction()
        ),
        constant_term.precision,
    )
    return NewtonTransformResult(
        characteristic=admitted.characteristic,
        initial_root=request.initial_root,
        ramification_index=admitted.ramification_index,
        ordinate_power=admitted.ordinate_power,
        removed_valuation=admitted.removed_valuation,
        transformed_polynomial=transformed,
        constant_term_valuation_lower_bound=constant_valuation_lower_bound,
    )


__all__ = [
    "NewtonTransformRequest",
    "NewtonTransformResult",
    "newton_transform",
]
