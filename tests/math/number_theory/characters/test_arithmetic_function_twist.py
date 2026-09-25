from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool
from jacobian.math.number_theory.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletConvolutionResult,
)
from jacobian.math.number_theory.arithmetic_functions._tools import (
    compute_dirichlet_convolution,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterArithmeticFunctionTwistRequest,
)
from jacobian.math.number_theory.characters._tools import TOOLS
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character_arithmetic_function_twist,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _function(values: tuple[int, ...]) -> DirichletConvolutionResult:
    return DirichletConvolutionResult(
        values=tuple(_rational(value) for value in values), length=len(values)
    )


def _coefficients(result: object) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients_ascending)
        for value in result.values
    )


def test_quartic_character_twists_arithmetic_function_prefix_exactly() -> None:
    # Independent table: for the generator 2 mod 5, chi(2)=i, so the
    # character values on 1,...,7 are 1,i,-i,-1,0,1,i.
    character = DirichletCharacter(group=character_group(5), coordinates=(1,))
    request = DirichletCharacterArithmeticFunctionTwistRequest(
        function=_function((2, 3, 5, 7, 11, 13, 17)), character=character
    )

    result = dirichlet_character_arithmetic_function_twist(request)

    assert result.index_origin == 1
    assert result.field.order == 4
    assert _coefficients(result) == (
        (Fraction(2), Fraction(0)),
        (Fraction(0), Fraction(3)),
        (Fraction(0), Fraction(-5)),
        (Fraction(-7), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(13), Fraction(0)),
        (Fraction(0), Fraction(17)),
    )


def test_modulus_one_trivial_character_preserves_rational_prefix() -> None:
    character = DirichletCharacter(group=character_group(1), coordinates=())
    result = dirichlet_character_arithmetic_function_twist(
        DirichletCharacterArithmeticFunctionTwistRequest(
            function=_function((0, -3, 8)), character=character
        )
    )

    assert result.index_origin == 1
    assert result.field.order == 1
    assert _coefficients(result) == (
        (Fraction(0),),
        (Fraction(-3),),
        (Fraction(8),),
    )


def test_dirichlet_convolution_output_composes_with_character_twist() -> None:
    convolution = compute_dirichlet_convolution(
        DirichletConvolutionRequest(
            f=tuple(_rational(value) for value in (1, 2, 3, 4, 5, 6)),
            g=tuple(_rational(value) for value in (2, 1, 0, 3, 2, 1)),
        )
    )
    character = DirichletCharacter(group=character_group(3), coordinates=(1,))
    twisted = dirichlet_character_arithmetic_function_twist(
        DirichletCharacterArithmeticFunctionTwistRequest(
            function=convolution, character=character
        )
    )

    # The convolution prefix is (2, 5, 6, 13, 12, 16); chi_3 is +1,-1,0,
    # hence the arithmetic-function twist is (2,-5,0,13,-12,0).
    assert _coefficients(twisted) == tuple(
        (Fraction(value),) for value in (2, -5, 0, 13, -12, 0)
    )


def test_all_zero_prefix_stays_exact_zero_in_character_field() -> None:
    character = DirichletCharacter(group=character_group(5), coordinates=(1,))
    result = dirichlet_character_arithmetic_function_twist(
        DirichletCharacterArithmeticFunctionTwistRequest(
            function=_function((0, 0, 0, 0, 0)), character=character
        )
    )

    assert result.field.order == 4
    assert len(result.values) == 5
    assert all(
        not any(coefficient.num for coefficient in value.coefficients_ascending)
        for value in result.values
    )


def test_catalog_publishes_and_executes_arithmetic_function_twist() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "arithmetic_function.dirichlet_character_twist.compute"
    )
    assert isinstance(tool, MathTool)
    request = DirichletCharacterArithmeticFunctionTwistRequest(
        function=_function((2, 3, 4)),
        character=DirichletCharacter(group=character_group(3), coordinates=(1,)),
    )

    result = tool.run(request)

    assert result.index_origin == 1
    assert _coefficients(result) == (
        (Fraction(2),),
        (Fraction(-3),),
        (Fraction(0),),
    )
