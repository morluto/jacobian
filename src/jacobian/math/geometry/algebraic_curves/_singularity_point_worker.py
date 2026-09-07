"""One-shot exact point-construction kernel for projective singular loci."""

from __future__ import annotations

import sys
from fractions import Fraction
from math import comb
from typing import Any, Literal, Self

import sympy
from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry.algebraic_curves._singularity_models import (
    MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE,
    MAX_PROJECTIVE_SINGULAR_POINTS,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomialIdeal,
)

_MAX_POINT_COORDINATE_DIGITS = 256
_MAX_SHAPE_ATTEMPTS = comb(MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE, 2) + 1


class ProjectiveSingularityPointSeed(StrictModel):
    """One residue-field point before its exact embeddings are enumerated."""

    presentation: SimpleNumberFieldPresentation
    coordinates: tuple[
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
    ]
    chart_index: StrictInt = Field(ge=0, le=2)

    @model_validator(mode="after")
    def bind_field_and_chart(self) -> Self:
        if any(
            coordinate.presentation != self.presentation
            for coordinate in self.coordinates
        ):
            raise ValueError("point-seed coordinates do not share one presentation")
        coordinate_is_zero = tuple(
            all(
                coefficient.as_fraction() == 0
                for coefficient in coordinate.coefficients_ascending
            )
            for coordinate in self.coordinates
        )
        coordinate_is_one = tuple(
            coordinate.coefficients_ascending[0].as_fraction() == 1
            and all(
                coefficient.as_fraction() == 0
                for coefficient in coordinate.coefficients_ascending[1:]
            )
            for coordinate in self.coordinates
        )
        if (
            any(not coordinate_is_zero[index] for index in range(self.chart_index))
            or not coordinate_is_one[self.chart_index]
        ):
            raise ValueError("point seed is not normalized in its declared chart")
        return self


class ProjectiveSingularityPointWorkerRequest(StrictModel):
    """Canonical exact input for one complete point-construction transaction."""

    variables: tuple[PolynomialVariable, PolynomialVariable, PolynomialVariable]
    chart_zero_components: tuple[RationalPolynomialIdeal, ...] = Field(
        max_length=MAX_PROJECTIVE_SINGULAR_POINTS
    )
    chart_one_components: tuple[RationalPolynomialIdeal, ...] = Field(
        max_length=MAX_PROJECTIVE_SINGULAR_POINTS
    )
    chart_two_present: StrictBool

    @model_validator(mode="after")
    def bind_component_axes(self) -> Self:
        if any(
            component.variables != self.variables[1:]
            for component in self.chart_zero_components
        ):
            raise ValueError("first-chart component has the wrong axis")
        if any(
            component.variables != self.variables[2:]
            for component in self.chart_one_components
        ):
            raise ValueError("second-chart component has the wrong axis")
        if (
            len(self.chart_zero_components) + len(self.chart_one_components)
            > MAX_PROJECTIVE_SINGULAR_POINTS
        ):
            raise ValueError("chart components exceed the geometric point bound")
        return self


class ProjectiveSingularityPointWorkerComplete(StrictModel):
    """Complete residue-field seeds for the disjoint projective chart cover."""

    kind: Literal["complete"]
    seeds: tuple[ProjectiveSingularityPointSeed, ...] = Field(
        min_length=1,
        max_length=MAX_PROJECTIVE_SINGULAR_POINTS,
    )

    @model_validator(mode="after")
    def require_complete_canonical_family(self) -> Self:
        if sum(seed.presentation.degree for seed in self.seeds) > (
            MAX_PROJECTIVE_SINGULAR_POINTS
        ):
            raise ValueError("embedded point count exceeds the geometric point bound")
        keys = tuple(seed.model_dump_json() for seed in self.seeds)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("point seeds are not unique and canonically ordered")
        return self


def _shape_data(
    component: RationalPolynomialIdeal,
) -> tuple[sympy.Symbol, sympy.Poly, tuple[sympy.Poly, sympy.Poly]]:
    """Return a deterministic rational univariate representation of a chart prime."""

    coordinate_symbols = symbols_for_variables(component.variables)
    if len(coordinate_symbols) != 2:
        raise ValueError("shape construction requires a two-axis chart")
    parameter = sympy.Symbol("__jacobian_shape_parameter")
    expressions = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in component.generators
    ]

    for coefficient in range(_MAX_SHAPE_ATTEMPTS):
        basis = sympy.groebner(
            [
                *expressions,
                parameter - coordinate_symbols[0] - coefficient * coordinate_symbols[1],
            ],
            *coordinate_symbols,
            parameter,
            order="lex",
            domain=sympy.QQ,
        )
        univariate = [
            polynomial
            for polynomial in basis.polys
            if polynomial.as_expr().free_symbols <= {parameter}
            and polynomial.degree(parameter) > 0
        ]
        if len(univariate) != 1:
            continue
        eliminant = sympy.Poly(
            univariate[0].monic().as_expr(), parameter, domain=sympy.QQ
        )
        if not 1 <= int(eliminant.degree()) <= MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE:
            raise ValueError("a chart residue field exceeds the degree-four bound")

        coordinate_polynomials: list[sympy.Poly] = []
        for coordinate_index, coordinate in enumerate(coordinate_symbols):
            other = coordinate_symbols[1 - coordinate_index]
            selected: sympy.Poly | None = None
            for polynomial in basis.polys:
                expression = polynomial.as_expr()
                if polynomial.degree(coordinate) != 1 or polynomial.degree(other) != 0:
                    continue
                as_coordinate = sympy.Poly(expression, coordinate)
                leading = as_coordinate.coeff_monomial(coordinate)
                remainder = sympy.expand(expression - leading * coordinate)
                if leading.free_symbols or remainder.free_symbols - {parameter}:
                    continue
                selected = sympy.Poly(
                    sympy.expand(-remainder / leading), parameter, domain=sympy.QQ
                )
                break
            if selected is None:
                break
            coordinate_polynomials.append(selected)
        if len(coordinate_polynomials) == 2:
            return (
                parameter,
                eliminant,
                (coordinate_polynomials[0], coordinate_polynomials[1]),
            )
    raise ValueError("no separating chart coordinate fit the seven-attempt bound")


def _univariate_component_polynomial(
    component: RationalPolynomialIdeal,
) -> tuple[sympy.Symbol, sympy.Poly]:
    (variable,) = symbols_for_variables(component.variables)
    nonzero: list[sympy.Poly] = []
    for generator in component.generators:
        converted = rational_polynomial_to_sympy(generator)
        if not converted.is_zero:
            nonzero.append(converted)
    if not nonzero:
        raise ValueError("a finite chart component returned the zero ideal")
    polynomial = nonzero[0]
    for generator in nonzero[1:]:
        polynomial = sympy.gcd(polynomial, generator)
    polynomial = sympy.Poly(polynomial.monic().as_expr(), variable, domain=sympy.QQ)
    if not 1 <= int(polynomial.degree()) <= MAX_PROJECTIVE_SINGULAR_FIELD_DEGREE:
        raise ValueError("a chart residue field exceeds the degree-four bound")
    return variable, polynomial


def _primitive_integer_factor(polynomial: sympy.Poly) -> tuple[int, ...]:
    _denominator, integer_polynomial = polynomial.clear_denoms(convert=True)
    _content, primitive = integer_polynomial.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    coefficients = tuple(int(coefficient) for coefficient in primitive.all_coeffs())
    if any(
        len(format_canonical_integer(abs(coefficient))) > _MAX_POINT_COORDINATE_DIGITS
        for coefficient in coefficients
    ):
        raise ValueError("a residue-field polynomial exceeds the carrier digit bound")
    return tuple(coefficients)


def _canonical_rational(value: Any) -> CanonicalRational:
    rational = sympy.Rational(value)
    numerator = int(rational.p)
    denominator = int(rational.q)
    if (
        len(format_canonical_integer(abs(numerator))) > _MAX_POINT_COORDINATE_DIGITS
        or len(format_canonical_integer(denominator)) > _MAX_POINT_COORDINATE_DIGITS
    ):
        raise ValueError("a point coordinate exceeds the carrier digit bound")
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def _field_element(
    presentation: SimpleNumberFieldPresentation,
    polynomial: sympy.Poly,
) -> SimpleNumberFieldElement:
    parameter = polynomial.gens[0]
    if polynomial.degree(parameter) >= presentation.degree:
        raise ValueError("a point coordinate is not reduced in its residue field")
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            _canonical_rational(polynomial.nth(degree))
            for degree in range(presentation.degree)
        ),
    )


def _rational_field_element(
    presentation: SimpleNumberFieldPresentation,
    value: Any,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=(_canonical_rational(value),),
    )


def _seed_for_factor(
    *,
    factor: sympy.Poly,
    parameter: sympy.Symbol,
    coordinate_polynomials: tuple[sympy.Poly, sympy.Poly, sympy.Poly],
    chart_index: int,
) -> ProjectiveSingularityPointSeed:
    if int(factor.degree()) == 1:
        presentation = SimpleNumberFieldPresentation(coefficients_descending=(1, 0))
        root = -factor.nth(0) / factor.nth(1)
        coordinate_values = tuple(
            _rational_field_element(presentation, polynomial.eval(root))
            for polynomial in coordinate_polynomials
        )
    else:
        presentation = SimpleNumberFieldPresentation(
            coefficients_descending=_primitive_integer_factor(factor)
        )
        coordinate_values = tuple(
            _field_element(
                presentation,
                sympy.Poly(
                    polynomial.rem(factor).as_expr(), parameter, domain=sympy.QQ
                ),
            )
            for polynomial in coordinate_polynomials
        )
    coordinates = (
        coordinate_values[0],
        coordinate_values[1],
        coordinate_values[2],
    )
    return ProjectiveSingularityPointSeed(
        presentation=presentation,
        coordinates=coordinates,
        chart_index=chart_index,
    )


def _factor_seeds(
    *,
    eliminant: sympy.Poly,
    parameter: sympy.Symbol,
    coordinate_polynomials: tuple[sympy.Poly, sympy.Poly, sympy.Poly],
    chart_index: int,
) -> tuple[ProjectiveSingularityPointSeed, ...]:
    _coefficient, factors = eliminant.factor_list()
    if any(multiplicity != 1 for _factor, multiplicity in factors):
        raise ValueError("a minimal-prime eliminant retained multiplicity")
    return tuple(
        _seed_for_factor(
            factor=sympy.Poly(factor.monic().as_expr(), parameter, domain=sympy.QQ),
            parameter=parameter,
            coordinate_polynomials=coordinate_polynomials,
            chart_index=chart_index,
        )
        for factor, _multiplicity in factors
    )


def _two_axis_seeds(
    component: RationalPolynomialIdeal,
) -> tuple[ProjectiveSingularityPointSeed, ...]:
    parameter, eliminant, (first, second) = _shape_data(component)
    one = sympy.Poly(1, parameter, domain=sympy.QQ)
    return _factor_seeds(
        eliminant=eliminant,
        parameter=parameter,
        coordinate_polynomials=(one, first, second),
        chart_index=0,
    )


def _one_axis_seeds(
    component: RationalPolynomialIdeal,
) -> tuple[ProjectiveSingularityPointSeed, ...]:
    parameter, eliminant = _univariate_component_polynomial(component)
    zero = sympy.Poly(0, parameter, domain=sympy.QQ)
    one = sympy.Poly(1, parameter, domain=sympy.QQ)
    coordinate = sympy.Poly(parameter, parameter, domain=sympy.QQ)
    return _factor_seeds(
        eliminant=eliminant,
        parameter=parameter,
        coordinate_polynomials=(zero, one, coordinate),
        chart_index=1,
    )


def _chart_two_seed() -> ProjectiveSingularityPointSeed:
    parameter = sympy.Symbol("__jacobian_rational_parameter")
    factor = sympy.Poly(parameter, parameter, domain=sympy.QQ)
    zero = sympy.Poly(0, parameter, domain=sympy.QQ)
    one = sympy.Poly(1, parameter, domain=sympy.QQ)
    return _seed_for_factor(
        factor=factor,
        parameter=parameter,
        coordinate_polynomials=(zero, zero, one),
        chart_index=2,
    )


def compute_point_worker_response(
    request: ProjectiveSingularityPointWorkerRequest,
) -> ProjectiveSingularityPointWorkerComplete:
    """Construct every residue-field seed in one isolated exact transaction."""

    seeds: list[ProjectiveSingularityPointSeed] = []
    for component in request.chart_zero_components:
        seeds.extend(_two_axis_seeds(component))
    for component in request.chart_one_components:
        seeds.extend(_one_axis_seeds(component))
    if request.chart_two_present:
        seeds.append(_chart_two_seed())
    ordered = tuple(sorted(seeds, key=lambda seed: seed.model_dump_json()))
    return ProjectiveSingularityPointWorkerComplete(kind="complete", seeds=ordered)


def main() -> int:
    request = ProjectiveSingularityPointWorkerRequest.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    response = compute_point_worker_response(request)
    sys.stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProjectiveSingularityPointSeed",
    "ProjectiveSingularityPointWorkerComplete",
    "ProjectiveSingularityPointWorkerRequest",
    "compute_point_worker_response",
]
