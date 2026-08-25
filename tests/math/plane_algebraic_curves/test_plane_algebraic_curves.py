"""Tests for plane algebraic curve operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy

import pytest
import sympy
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.plane_algebraic_curves import _conic
from jacobian.math.plane_algebraic_curves._conic import MAX_CONIC_INPUT_DIGITS
from jacobian.math.plane_algebraic_curves._models import (
    AffineChartRequest,
    AffineCurveRequest,
    ProjectiveClosureRequest,
    RationalConicParametrizationRequest,
    RationalConicParametrizationResult,
)
from jacobian.math.plane_algebraic_curves._operations import (
    compute_affine_chart,
    compute_affine_curve_check,
    compute_projective_closure,
    compute_rational_conic_parametrization,
)
from jacobian.math.plane_algebraic_curves._tools import TOOLS
from jacobian.math.polynomials._conversions import (
    rational_function_to_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


@contextmanager
def _raises_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == f"plane_algebraic_curve.{code}"


def _rational(numerator: str | int, denominator: str | int = 1) -> CanonicalRational:
    return CanonicalRational(num=str(numerator), den=str(denominator))


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(coefficient, 1),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _rational_polynomial(
    variables: tuple[str, ...],
    *terms: tuple[tuple[str, str], tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(*coefficient),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _parametrize(
    polynomial: RationalPolynomial,
    point: tuple[CanonicalRational, CanonicalRational],
    parameter: str = "t",
) -> RationalConicParametrizationResult:
    return compute_rational_conic_parametrization(
        RationalConicParametrizationRequest(
            polynomial=polynomial,
            point=_point(polynomial.variables, point),
            parameter=parameter,
        )
    )


def _point(
    variables: tuple[str, ...],
    values: tuple[CanonicalRational, CanonicalRational],
) -> VariablePoint:
    return VariablePoint(variables=variables, values=values)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebraic_geometry.affine_plane_curve.check",
        "algebraic_geometry.conic.rational_parametrization.compute",
        "algebraic_geometry.plane_curve.projective_closure.compute",
        "algebraic_geometry.projective_curve.affine_chart.compute",
    }


def test_rational_conic_parametrization_has_canonical_known_answer() -> None:
    source = _polynomial(
        ("x", "y"),
        (1, (2, 0)),
        (1, (1, 1)),
        (-1, (0, 2)),
        (-1, (0, 0)),
    )
    point = (_rational(1), _rational(0))
    result = _parametrize(source, point)
    t, x, y = sympy.symbols("t x y")

    assert rational_function_to_sympy(result.coordinates[0]) == (t**2 + 1) / (
        t**2 + t - 1
    )
    assert rational_function_to_sympy(result.coordinates[1]) == (2 * t + 1) / (
        t**2 + t - 1
    )
    assert rational_function_to_sympy(result.inverse_parameter) == (
        -x / 2 + y + sympy.Rational(1, 2)
    ) / (x + y / 2 - 1)
    assert (
        rational_polynomial_to_sympy(result.finite_parameter_denominator).as_expr()
        == t**2 + t - 1
    )
    assert result.exceptional_point == _point(source.variables, point)
    assert result.exceptional_parameter == "PROJECTIVE_INFINITY"
    assert result.normalization == "GRADIENT_ORTHOGONAL_LINE_PENCIL"


@pytest.mark.parametrize(
    ("source", "point"),
    [
        (
            _polynomial(
                ("x", "y"),
                (1, (2, 0)),
                (1, (0, 2)),
                (-1, (0, 0)),
            ),
            (_rational(1), _rational(0)),
        ),
        (
            _polynomial(("x", "y"), (1, (1, 1)), (-1, (0, 0))),
            (_rational(1), _rational(1)),
        ),
        (
            _polynomial(("x", "y"), (-1, (2, 0)), (1, (0, 1))),
            (_rational(0), _rational(0)),
        ),
        (
            _polynomial(
                ("u", "v"),
                (2, (2, 0)),
                (-3, (1, 1)),
                (-1, (0, 2)),
                (-2, (0, 0)),
            ),
            (_rational(1), _rational(0)),
        ),
    ],
)
def test_parametrization_replays_substitution_and_inverse_identities(
    source: RationalPolynomial,
    point: tuple[CanonicalRational, CanonicalRational],
) -> None:
    result = _parametrize(source, point)
    coordinate_expressions = tuple(
        rational_function_to_sympy(coordinate) for coordinate in result.coordinates
    )
    variables = symbols_for_variables(source.variables)
    parameter = sympy.Symbol(result.parameter)
    substitutions = dict(zip(variables, coordinate_expressions, strict=True))

    assert (
        sympy.cancel(
            rational_polynomial_to_sympy(source)
            .as_expr()
            .subs(
                substitutions,
                simultaneous=True,
            )
        )
        == 0
    )
    assert (
        sympy.cancel(
            rational_function_to_sympy(result.inverse_parameter).subs(
                substitutions,
                simultaneous=True,
            )
            - parameter
        )
        == 0
    )
    inverse = rational_function_to_sympy(result.inverse_parameter)
    source_polynomial = rational_polynomial_to_sympy(source)
    for coordinate, variable in zip(coordinate_expressions, variables, strict=True):
        numerator, _denominator = sympy.fraction(
            sympy.cancel(coordinate.subs(parameter, inverse) - variable)
        )
        _quotient, remainder = sympy.Poly(numerator, *variables, domain=sympy.QQ).div(
            source_polynomial
        )
        assert remainder.is_zero
    assert tuple(
        sympy.limit(coordinate, parameter, sympy.oo)
        for coordinate in coordinate_expressions
    ) == tuple(sympy.Rational(*coordinate.as_integer_ratio()) for coordinate in point)


def test_parabola_retains_common_denominator_locus_after_coordinate_cancellation() -> (
    None
):
    result = _parametrize(
        _polynomial(("x", "y"), (-1, (2, 0)), (1, (0, 1))),
        (_rational(0), _rational(0)),
    )
    t = sympy.Symbol("t")

    assert rational_function_to_sympy(result.coordinates[0]) == -1 / t
    assert rational_function_to_sympy(result.coordinates[1]) == 1 / t**2
    assert (
        rational_polynomial_to_sympy(result.finite_parameter_denominator).as_expr()
        == t**2
    )
    source = rational_polynomial_to_sympy(result.source_polynomial)
    projective_x = sympy.Poly(
        sympy.cancel(rational_function_to_sympy(result.coordinates[0]) * t**2),
        t,
        domain=sympy.QQ,
    )
    projective_y = sympy.Poly(
        sympy.cancel(rational_function_to_sympy(result.coordinates[1]) * t**2),
        t,
        domain=sympy.QQ,
    )
    projective_z = sympy.Poly(t**2, t, domain=sympy.QQ)
    assert projective_x.gcd(projective_y).gcd(projective_z).degree() == 0
    z = sympy.Symbol("z")
    assert (
        sympy.expand(
            source.homogenize(z)
            .as_expr()
            .subs(
                {
                    source.gens[0]: projective_x.as_expr(),
                    source.gens[1]: projective_y.as_expr(),
                    z: projective_z.as_expr(),
                },
                simultaneous=True,
            )
        )
        == 0
    )


def test_hyperbola_denominator_locus_is_not_inferred_from_one_coordinate() -> None:
    result = _parametrize(
        _polynomial(("x", "y"), (1, (1, 1)), (-1, (0, 0))),
        (_rational(1), _rational(1)),
    )
    t = sympy.Symbol("t")

    assert rational_function_to_sympy(result.coordinates[0]) == (t - 1) / (t + 1)
    assert rational_function_to_sympy(result.coordinates[1]) == (t + 1) / (t - 1)
    assert (
        rational_polynomial_to_sympy(result.finite_parameter_denominator).as_expr()
        == t**2 - 1
    )


def test_nonzero_equation_rescaling_preserves_normalized_parametrization() -> None:
    source = _polynomial(
        ("x", "y"),
        (1, (2, 0)),
        (1, (1, 1)),
        (-1, (0, 2)),
        (-1, (0, 0)),
    )
    rescaled = _polynomial(
        ("x", "y"),
        (-7, (2, 0)),
        (-7, (1, 1)),
        (7, (0, 2)),
        (7, (0, 0)),
    )
    point = (_rational(1), _rational(0))

    original = _parametrize(source, point)
    scaled = _parametrize(rescaled, point)
    assert scaled.coordinates == original.coordinates
    assert scaled.inverse_parameter == original.inverse_parameter
    assert scaled.finite_parameter_denominator == original.finite_parameter_denominator


def test_request_rejects_point_outside_conic() -> None:
    with _raises_code("point_not_on_conic"):
        RationalConicParametrizationRequest(
            polynomial=_polynomial(
                ("x", "y"),
                (1, (2, 0)),
                (1, (0, 2)),
                (-1, (0, 0)),
            ),
            point=_point(("x", "y"), (_rational(0), _rational(0))),
        )


@pytest.mark.parametrize(
    ("source", "point"),
    [
        (
            _polynomial(("x", "y"), (1, (1, 1))),
            (_rational(0), _rational(0)),
        ),
        (
            _polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2))),
            (_rational(0), _rational(0)),
        ),
        (
            _polynomial(("x", "y"), (1, (2, 0)), (-1, (0, 0))),
            (_rational(1), _rational(0)),
        ),
    ],
)
def test_request_rejects_singular_or_reducible_quadratics(
    source: RationalPolynomial,
    point: tuple[CanonicalRational, CanonicalRational],
) -> None:
    with _raises_code("conic_not_smooth_irreducible"):
        RationalConicParametrizationRequest(
            polynomial=source,
            point=_point(source.variables, point),
        )


def test_request_rejects_wrong_degree_axis_and_parameter_collision() -> None:
    point = (_rational(0), _rational(0))
    with _raises_code("conic_degree_invalid"):
        RationalConicParametrizationRequest(
            polynomial=_polynomial(("x", "y"), (1, (1, 0))),
            point=_point(("x", "y"), point),
        )
    with _raises_code("conic_axis_invalid"):
        RationalConicParametrizationRequest(
            polynomial=_polynomial(("x",), (1, (2,))),
            point=_point(("x", "y"), point),
        )
    with _raises_code("parameter_axis_collision"):
        RationalConicParametrizationRequest(
            polynomial=_polynomial(
                ("x", "y"),
                (1, (2, 0)),
                (1, (0, 2)),
                (-1, (0, 0)),
            ),
            point=_point(("x", "y"), (_rational(1), _rational(0))),
            parameter="x",
        )


@pytest.mark.parametrize("variables", [("x", "z"), ("y", "x")])
def test_request_rejects_mismatched_or_reordered_point_axis(
    variables: tuple[str, str],
) -> None:
    source = _polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0)))
    with _raises_code("point_axis_mismatch"):
        RationalConicParametrizationRequest(
            polynomial=source,
            point=_point(variables, (_rational(1), _rational(0))),
        )


def test_input_coefficient_boundary_is_accepted_then_rejected() -> None:
    accepted_denominator = "1" + "0" * (MAX_CONIC_INPUT_DIGITS - 1)
    accepted = _rational_polynomial(
        ("x", "y"),
        (("1", accepted_denominator), (2, 0)),
        (("1", accepted_denominator), (0, 2)),
        (("-1", accepted_denominator), (0, 0)),
    )
    result = _parametrize(accepted, (_rational(1), _rational(0)))
    assert result.coordinates

    rejected_denominator = "1" + "0" * MAX_CONIC_INPUT_DIGITS
    rejected = _rational_polynomial(
        ("x", "y"),
        (("1", rejected_denominator), (2, 0)),
        (("1", rejected_denominator), (0, 2)),
        (("-1", rejected_denominator), (0, 0)),
    )
    with _raises_code("coefficient_height_exceeded"):
        RationalConicParametrizationRequest(
            polynomial=rejected,
            point=_point(("x", "y"), (_rational(1), _rational(0))),
        )


def test_normalized_result_height_boundary_is_accepted_then_rejected() -> None:
    source = _polynomial(("x", "y"), (-1, (2, 0)), (1, (0, 1)))
    accepted_x = 10**31
    accepted = _parametrize(
        source,
        (_rational(accepted_x), _rational(accepted_x**2)),
    )
    assert accepted.coordinates

    rejected_x = 10**32
    with _raises_code("coefficient_height_exceeded"):
        RationalConicParametrizationRequest(
            polynomial=source,
            point=_point(
                ("x", "y"),
                (_rational(rejected_x), _rational(rejected_x**2)),
            ),
        )


def test_request_admission_completes_before_any_backend_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ForbiddenSympy:
        def __getattr__(self, name: str) -> object:
            raise AssertionError("request admission must not enter SymPy")

    monkeypatch.setattr(_conic, "sympy", _ForbiddenSympy())
    source = _polynomial(("x", "y"), (-1, (2, 0)), (1, (0, 1)))

    accepted_x = 10**31
    request = RationalConicParametrizationRequest(
        polynomial=source,
        point=_point(("x", "y"), (_rational(accepted_x), _rational(accepted_x**2))),
    )
    assert request.parameter == "t"

    rejected_x = 10**32
    with _raises_code("coefficient_height_exceeded"):
        RationalConicParametrizationRequest(
            polynomial=source,
            point=_point(
                ("x", "y"),
                (_rational(rejected_x), _rational(rejected_x**2)),
            ),
        )


def test_exact_preflight_admits_rational_point_on_hyperbola() -> None:
    source = _polynomial(("x", "y"), (1, (1, 1)), (-1, (0, 0)))
    point = (_rational(1, 3), _rational(3))
    result = _parametrize(source, point)
    coordinate_expressions = tuple(
        rational_function_to_sympy(coordinate) for coordinate in result.coordinates
    )
    substitutions = dict(
        zip(
            symbols_for_variables(result.source_polynomial.variables),
            coordinate_expressions,
            strict=True,
        )
    )

    assert (
        sympy.cancel(
            rational_polynomial_to_sympy(source)
            .as_expr()
            .subs(substitutions, simultaneous=True)
        )
        == 0
    )


def test_result_rejects_independently_forged_source_point_and_coordinates() -> None:
    source = _polynomial(
        ("x", "y"),
        (1, (2, 0)),
        (1, (1, 1)),
        (-1, (0, 2)),
        (-1, (0, 0)),
    )
    result = _parametrize(source, (_rational(1), _rational(0)))
    payload = result.model_dump(mode="json")

    forged_coordinates = deepcopy(payload)
    forged_coordinates["coordinates"][0] = payload["coordinates"][1]
    with _raises_code("parametrization_not_canonical"):
        RationalConicParametrizationResult.model_validate(forged_coordinates)

    forged_source = deepcopy(payload)
    forged_source["source_polynomial"] = _polynomial(
        ("x", "y"),
        (1, (2, 0)),
        (1, (0, 2)),
        (-1, (0, 0)),
    ).model_dump(mode="json")
    with _raises_code("parametrization_not_canonical"):
        RationalConicParametrizationResult.model_validate(forged_source)

    forged_point = deepcopy(payload)
    forged_point["exceptional_point"] = _point(
        source.variables, (_rational(-1), _rational(0))
    ).model_dump(mode="json")
    with _raises_code("parametrization_not_canonical"):
        RationalConicParametrizationResult.model_validate(forged_point)

    forged_inverse = deepcopy(payload)
    forged_inverse["inverse_parameter"] = payload["coordinates"][0]
    with _raises_code("parametrization_not_canonical"):
        RationalConicParametrizationResult.model_validate(forged_inverse)

    forged_denominator = deepcopy(payload)
    denominator_terms = forged_denominator["finite_parameter_denominator"][
        "polynomial"
    ]["terms"]
    denominator_terms[-1]["coefficient"]["num"] = "-2"
    with _raises_code("parametrization_not_canonical"):
        RationalConicParametrizationResult.model_validate(forged_denominator)


def test_parametrization_schema_guides_cross_field_contract() -> None:
    schema = RationalConicParametrizationRequest.model_json_schema()
    assert "smooth" in schema["properties"]["polynomial"]["description"]
    assert "ordered" in schema["properties"]["point"]["description"]
    assert schema["properties"]["point"]["examples"]
    assert schema["properties"]["parameter"]["examples"] == ["t"]

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "algebraic_geometry.conic.rational_parametrization.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    output = tool.run(request)
    assert tool.result_type.model_validate(output.model_dump(mode="json")) == output


def test_affine_curve_check_circle() -> None:
    request = AffineCurveRequest(
        polynomial=_polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0)))
    )
    result = compute_affine_curve_check(request)
    assert result.is_valid is True
    assert result.degree == 2


def test_projective_closure_circle_is_canonical_polynomial() -> None:
    source = _polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0)))
    result = compute_projective_closure(ProjectiveClosureRequest(polynomial=source))
    assert result.polynomial == _polynomial(
        ("x", "y", "z"),
        (1, (2, 0, 0)),
        (1, (0, 2, 0)),
        (-1, (0, 0, 2)),
    )


def test_affine_chart_circle_is_directly_composable() -> None:
    projective = _polynomial(
        ("x", "y", "z"),
        (1, (2, 0, 0)),
        (1, (0, 2, 0)),
        (-1, (0, 0, 2)),
    )
    result = compute_affine_chart(
        AffineChartRequest(polynomial=projective, chart_variable="z")
    )
    assert result.polynomial == _polynomial(
        ("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0))
    )
    AffineCurveRequest(polynomial=result.polynomial)


def test_homogenize_dehomogenize_round_trip() -> None:
    affine = _polynomial(
        ("x", "y"),
        (1, (3, 0)),
        (-2, (1, 1)),
        (1, (0, 1)),
        (-7, (0, 0)),
    )
    closure = compute_projective_closure(ProjectiveClosureRequest(polynomial=affine))
    chart = compute_affine_chart(
        AffineChartRequest(polynomial=closure.polynomial, chart_variable="z")
    )
    assert chart.polynomial == affine


def test_expression_strings_are_not_a_public_polynomial_contract() -> None:
    for payload in (
        "sin(x) + y",
        "x +* y",
        "x + t",
        "__import__('os').getcwd()",
    ):
        with pytest.raises(ValidationError) as caught:
            AffineCurveRequest.model_validate({"polynomial": payload})
        assert caught.value.errors()[0]["type"] == "model_type"


def test_duplicate_and_invalid_variable_names_are_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        _polynomial(("x", "x"), (1, (1, 0)))
    assert caught.value.errors()[0]["type"] == "polynomial.duplicate_variables"
    with pytest.raises(ValidationError) as caught:
        _polynomial(("", "y"), (1, (1, 0)))
    assert caught.value.errors()[0]["type"] == "string_pattern_mismatch"


def test_variable_named_z_rejected_in_projective_closure() -> None:
    with _raises_code("homogenizing_coordinate_reserved"):
        ProjectiveClosureRequest(
            polynomial=_polynomial(("x", "z"), (1, (2, 0)), (-1, (0, 1)))
        )


@pytest.mark.parametrize("constant", [0, 5])
def test_constant_polynomial_is_not_a_valid_curve(constant: int) -> None:
    terms = () if constant == 0 else ((constant, (0, 0)),)
    result = compute_affine_curve_check(
        AffineCurveRequest(polynomial=_polynomial(("x", "y"), *terms))
    )
    assert result.is_valid is False
    assert result.degree == 0


def test_chart_requires_three_variables_and_a_homogeneous_polynomial() -> None:
    with _raises_code("chart_axis_invalid"):
        AffineChartRequest(
            polynomial=_polynomial(("x", "y"), (1, (2, 0))),
            chart_variable="x",
        )
    with _raises_code("polynomial_not_homogeneous"):
        AffineChartRequest(
            polynomial=_polynomial(("x", "y", "z"), (1, (2, 0, 0)), (1, (0, 1, 0))),
            chart_variable="z",
        )


def test_chart_variable_must_be_on_the_projective_axis() -> None:
    with _raises_code("chart_variable_axis_mismatch"):
        AffineChartRequest(
            polynomial=_polynomial(
                ("x", "y", "z"),
                (1, (2, 0, 0)),
                (1, (0, 2, 0)),
                (-1, (0, 0, 2)),
            ),
            chart_variable="w",
        )
