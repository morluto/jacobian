"""Finite-field Jacobian syzygy checks (#964): Graf example, char-p binding."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields import (
    Axis,
    FiniteFieldElement,
    FiniteFieldPresentation,
    check_jacobian_syzygy,
    finite_field,
)
from jacobian.math.finite_fields._algebraic_sets import (
    AlgebraicMonomial,
    AlgebraicPolynomial,
)
from jacobian.math.finite_fields._jacobian_syzygy_models import (
    MAX_JACOBIAN_SYZYGY_ROWS,
    JacobianSyzygyCheckRequest,
    JacobianSyzygyCheckResult,
)
from jacobian.math.finite_fields._tools import TOOLS

OPERATION_ID = "finite_field.jacobian_syzygy.check"


def _tool():
    return next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)


def _poly(
    presentation: FiniteFieldPresentation,
    axis: Axis,
    terms: dict[tuple[int, ...], int],
) -> AlgebraicPolynomial:
    canonical = tuple(
        AlgebraicMonomial._from_kernel(
            coefficient=FiniteFieldElement(
                presentation=presentation, coordinates=(coefficient,)
            ),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(terms.items(), reverse=True)
    )
    if not canonical:
        canonical = (
            AlgebraicMonomial._from_kernel(
                coefficient=FiniteFieldElement(
                    presentation=presentation, coordinates=(0,)
                ),
                exponents=(0,) * len(axis.labels),
            ),
        )
    return AlgebraicPolynomial._from_kernel(
        presentation=presentation, variable_axis=axis, terms=canonical
    )


def _term_map(polynomial: AlgebraicPolynomial) -> dict[tuple[int, ...], int]:
    return {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }


def _graf() -> tuple[FiniteFieldPresentation, Axis, AlgebraicPolynomial]:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x", "y", "z"))
    f = _poly(presentation, axis, {(0, 0, 2): 1, (3, 0, 0): 1, (0, 5, 0): 1})
    return presentation, axis, f


def _graf_rows(
    presentation: FiniteFieldPresentation, axis: Axis
) -> tuple[tuple[AlgebraicPolynomial, ...], ...]:
    zero = _poly(presentation, axis, {})
    one_z = _poly(presentation, axis, {(0, 0, 1): 1})
    y4 = _poly(presentation, axis, {(0, 4, 0): 1})
    x2 = _poly(presentation, axis, {(2, 0, 0): 1})
    return ((zero, zero, one_z), (y4, x2, zero))


def test_graf_frozen_example_verifies_with_derived_char_two_jacobian() -> None:
    presentation, axis, f = _graf()
    rows = _graf_rows(presentation, axis)

    result = check_jacobian_syzygy(
        polynomial=f, rows=rows, ideal_generators=(f,), reduction_variable="z"
    )

    assert result.status == "VERIFIED"
    assert result.first_failure is None
    assert [_term_map(partial) for partial in result.jacobian] == [
        {(2, 0, 0): 1},
        {(0, 4, 0): 1},
        {},
    ]
    assert all(entry.verified for entry in result.ledger)
    assert tuple(entry.row_index for entry in result.ledger) == (0, 1)
    # Source binding: the retained rows are exactly the supplied candidates.
    assert result.rows == rows
    assert result.polynomial is f


def test_declared_tool_example_runs_through_the_catalog_projection() -> None:
    tool = _tool()
    payload = json.dumps(tool.examples[0].input)
    request = JacobianSyzygyCheckRequest.model_validate_json(payload)
    result = tool.run(request)
    assert result.status == "VERIFIED"
    assert result.ledger[1].remainder.terms[0].coefficient.is_zero

    catalog_tool = Catalog.open().operation(OPERATION_ID)
    assert catalog_tool is not None
    catalog_result = catalog_tool.run(request)
    assert catalog_result.model_dump_json() == result.model_dump_json()


def test_unit_row_is_rejected_with_exact_x_squared_remainder() -> None:
    presentation, axis, f = _graf()
    one = _poly(presentation, axis, {(0, 0, 0): 1})
    zero = _poly(presentation, axis, {})

    result = check_jacobian_syzygy(
        polynomial=f,
        rows=((one, zero, zero),),
        ideal_generators=(f,),
        reduction_variable="z",
    )

    assert result.status == "REJECTED"
    assert result.first_failure == 0
    assert not result.ledger[0].verified
    assert _term_map(result.ledger[0].remainder) == {(2, 0, 0): 1}


def test_characteristic_zero_gradient_claim_cannot_slip_through() -> None:
    # The char-0 gradient (3x^2, 5y^4, 2z) of z^2+x^3+y^5, reduced to canonical
    # F2 residues, is (x^2, y^4, 0); the operation derives its own Jacobian and
    # this row is not a syzygy of it: x^2*x^2 + y^4*y^4 = x^4 + y^8.
    presentation, axis, f = _graf()
    x2 = _poly(presentation, axis, {(2, 0, 0): 1})
    y4 = _poly(presentation, axis, {(0, 4, 0): 1})
    zero = _poly(presentation, axis, {})

    result = check_jacobian_syzygy(
        polynomial=f,
        rows=((x2, y4, zero),),
        ideal_generators=(f,),
        reduction_variable="z",
    )

    assert result.status == "REJECTED"
    assert _term_map(result.ledger[0].remainder) == {(4, 0, 0): 1, (0, 8, 0): 1}


def test_formal_differentiation_is_characteristic_bound() -> None:
    axis = Axis(name="vars", labels=("x",))
    char_two = finite_field(2, (0, 1))
    char_three = finite_field(3, (0, 1))
    square_two = _poly(char_two, axis, {(2,): 1})
    square_three = _poly(char_three, axis, {(2,): 1})
    unit_two = _poly(char_two, axis, {(0,): 1})
    unit_three = _poly(char_three, axis, {(0,): 1})

    # d(x^2)/dx = 0 in characteristic 2, so every row is a syzygy.
    vanishing = check_jacobian_syzygy(polynomial=square_two, rows=((unit_two,),))
    assert vanishing.status == "VERIFIED"
    assert _term_map(vanishing.jacobian[0]) == {}

    # d(x^2)/dx = 2x in characteristic 3, and 1*2x does not reduce to zero.
    nonzero = check_jacobian_syzygy(polynomial=square_three, rows=((unit_three,),))
    assert nonzero.status == "REJECTED"
    assert _term_map(nonzero.jacobian[0]) == {(1,): 2}
    assert _term_map(nonzero.ledger[0].remainder) == {(1,): 2}


def test_empty_ideal_checks_syzygies_in_the_plain_polynomial_ring() -> None:
    presentation = finite_field(5, (0, 1))
    axis = Axis(name="vars", labels=("x", "y"))
    f = _poly(presentation, axis, {(2, 0): 1, (0, 2): 1})
    # (y, -x) is a syzygy of (2x, 2y): 2xy - 2xy = 0 over GF(5).
    row = (
        _poly(presentation, axis, {(0, 1): 1}),
        _poly(presentation, axis, {(1, 0): 4}),
    )

    result = check_jacobian_syzygy(polynomial=f, rows=(row,))

    assert result.status == "VERIFIED"
    assert result.ideal_generators == ()
    assert result.reduction_variable is None
    assert result.ledger[0].reduction_steps == 0


def test_zero_row_verifies_trivially() -> None:
    presentation, axis, f = _graf()
    zero = _poly(presentation, axis, {})

    result = check_jacobian_syzygy(
        polynomial=f,
        rows=((zero, zero, zero),),
        ideal_generators=(f,),
        reduction_variable="z",
    )

    assert result.status == "VERIFIED"
    assert result.ledger[0].reduction_steps == 0


def test_quotient_reduction_uses_the_principal_relation_exactly() -> None:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x", "z"))
    f = _poly(presentation, axis, {(1, 0): 1, (0, 2): 1})
    # z^2 + x is its own generator here; d/dx = 1, d/dz = 0 in char 2.
    generator = _poly(presentation, axis, {(0, 2): 1, (1, 0): 1})
    z2 = _poly(presentation, axis, {(0, 2): 1})
    x_plus_z2 = _poly(presentation, axis, {(1, 0): 1, (0, 2): 1})
    zero = _poly(presentation, axis, {})

    # z^2 * 1 = z^2 ≡ x mod (z^2 + x): rejected with remainder x after one step.
    rejected = check_jacobian_syzygy(
        polynomial=f,
        rows=((z2, zero),),
        ideal_generators=(generator,),
        reduction_variable="z",
    )
    assert rejected.status == "REJECTED"
    assert _term_map(rejected.ledger[0].remainder) == {(1, 0): 1}
    assert rejected.ledger[0].reduction_steps == 1

    # (z^2 + x) * 1 ≡ 0: verified through the same single reduction step.
    verified = check_jacobian_syzygy(
        polynomial=f,
        rows=((x_plus_z2, zero),),
        ideal_generators=(generator,),
        reduction_variable="z",
    )
    assert verified.status == "VERIFIED"
    assert verified.ledger[0].reduction_steps == 1


def test_defining_invariant_replays_through_independent_reduction() -> None:
    sympy = pytest.importorskip("sympy")
    presentation, axis, f = _graf()
    rows = _graf_rows(presentation, axis)
    result = check_jacobian_syzygy(
        polynomial=f, rows=rows, ideal_generators=(f,), reduction_variable="z"
    )
    assert result.status == "VERIFIED"

    def to_sympy(polynomial: AlgebraicPolynomial) -> object:
        generators = sympy.symbols(list(axis.labels))
        prime = presentation.characteristic
        expression = sympy.S.Zero
        for term in polynomial.terms:
            monomial = sympy.S.One
            for generator, exponent in zip(generators, term.exponents, strict=True):
                monomial *= generator**exponent
            expression += int(term.coefficient.coordinates[0]) * monomial
        return sympy.Poly(expression, *generators, modulus=prime)

    basis = sympy.groebner(
        [to_sympy(f)], *sympy.symbols(list(axis.labels)), order="lex", modulus=2
    )
    jacobian = [to_sympy(partial) for partial in result.jacobian]
    for row in result.rows:
        combination = sympy.S.Zero
        for entry, partial in zip(row, jacobian, strict=True):
            combination += to_sympy(entry).as_expr() * partial.as_expr()
        _quotients, remainder = basis.reduce(
            sympy.Poly(combination, *sympy.symbols(list(axis.labels)), modulus=2)
        )
        assert remainder.is_zero


def test_non_monic_principal_generator_is_a_typed_admission_rejection() -> None:
    presentation = finite_field(3, (0, 1))
    axis = Axis(name="vars", labels=("x", "z"))
    f = _poly(presentation, axis, {(1, 0): 1, (0, 2): 1})
    generator = _poly(presentation, axis, {(1, 1): 2, (0, 0): 1})
    zero = _poly(presentation, axis, {})

    with pytest.raises(OperationDomainValidationError) as excinfo:
        check_jacobian_syzygy(
            polynomial=f,
            rows=((zero, zero),),
            ideal_generators=(generator,),
            reduction_variable="z",
        )
    assert (
        excinfo.value.errors()[0]["type"]
        == "finite_field.jacobian_syzygy_monic_generator"
    )


def test_extension_field_presentation_is_rejected() -> None:
    presentation = finite_field(2, (1, 1, 1))
    axis = Axis(name="vars", labels=("x",))
    one = FiniteFieldElement(presentation=presentation, coordinates=(1, 0))
    f = AlgebraicPolynomial._from_kernel(
        presentation=presentation,
        variable_axis=axis,
        terms=(AlgebraicMonomial._from_kernel(coefficient=one, exponents=(2,)),),
    )

    with pytest.raises(OperationDomainValidationError) as excinfo:
        check_jacobian_syzygy(polynomial=f, rows=((f,),))
    assert (
        excinfo.value.errors()[0]["type"]
        == "finite_field.jacobian_syzygy_prime_field_only"
    )


def test_composite_characteristic_is_rejected() -> None:
    presentation = FiniteFieldPresentation(
        characteristic=4, modulus_coefficients=(0, 1)
    )
    axis = Axis(name="vars", labels=("x",))
    f = _poly(presentation, axis, {(2,): 1})

    with pytest.raises(OperationDomainValidationError) as excinfo:
        check_jacobian_syzygy(polynomial=f, rows=((f,),))
    assert (
        excinfo.value.errors()[0]["type"] == "finite_field.characteristic_prime_integer"
    )


def test_row_with_foreign_presentation_is_rejected() -> None:
    presentation, axis, f = _graf()
    foreign = finite_field(3, (0, 1))
    intruder = _poly(foreign, axis, {(0, 0, 1): 1})
    zero = _poly(presentation, axis, {})

    with pytest.raises(OperationDomainValidationError) as excinfo:
        check_jacobian_syzygy(
            polynomial=f,
            rows=((zero, zero, intruder),),
            ideal_generators=(f,),
            reduction_variable="z",
        )
    assert (
        excinfo.value.errors()[0]["type"]
        == "finite_field.jacobian_syzygy_shared_parent"
    )


def test_short_row_is_rejected() -> None:
    presentation, axis, f = _graf()
    zero = _poly(presentation, axis, {})

    with pytest.raises(OperationDomainValidationError) as excinfo:
        check_jacobian_syzygy(polynomial=f, rows=((zero, zero),))
    assert (
        excinfo.value.errors()[0]["type"] == "finite_field.jacobian_syzygy_row_length"
    )


def test_variable_count_envelope_rejects_before_any_reduction() -> None:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x1", "x2", "x3", "x4", "x5"))
    f = _poly(presentation, axis, {(1, 0, 0, 0, 0): 1})
    zero = _poly(presentation, axis, {})

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        check_jacobian_syzygy(polynomial=f, rows=((zero,) * 5,))
    assert (
        excinfo.value.errors()[0]["type"]
        == "finite_field.jacobian_syzygy_variable_bound"
    )


def test_term_envelope_rejects_before_any_reduction() -> None:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x",))
    terms = {(exponent,): 1 for exponent in range(65)}
    f = _poly(presentation, axis, terms)

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        check_jacobian_syzygy(polynomial=f, rows=((f,),))
    assert (
        excinfo.value.errors()[0]["type"] == "finite_field.jacobian_syzygy_term_bound"
    )


def test_row_count_envelope_rejects_native_and_request_boundaries() -> None:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("x",))
    zero = _poly(presentation, axis, {})
    rows = ((zero,),) * (MAX_JACOBIAN_SYZYGY_ROWS + 1)

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        check_jacobian_syzygy(polynomial=zero, rows=rows)
    assert excinfo.value.errors()[0]["type"] == "finite_field.jacobian_syzygy_row_bound"

    with pytest.raises(ValidationError):
        JacobianSyzygyCheckRequest(polynomial=zero, rows=rows)


def test_reduction_work_envelope_rejects_before_expansion() -> None:
    presentation = finite_field(2, (0, 1))
    axis = Axis(name="vars", labels=("w", "x", "y", "z"))
    dense = {(i, j, 0, 0): 1 for i in range(8) for j in range(8)}
    f = _poly(presentation, axis, dense)
    generator_terms: dict[tuple[int, ...], int] = {(0, 0, 0, 1): 1}
    generator_terms.update(
        {(i, j, k, 0): 1 for i in range(3) for j in range(3) for k in range(4)}
    )
    generator = _poly(presentation, axis, generator_terms)
    row = tuple(_poly(presentation, axis, dense) for _ in range(4))

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        check_jacobian_syzygy(
            polynomial=f,
            rows=(row,),
            ideal_generators=(generator,),
            reduction_variable="z",
        )
    assert (
        excinfo.value.errors()[0]["type"] == "finite_field.jacobian_syzygy_work_bound"
    )


def test_result_serialization_round_trips_and_composes() -> None:
    presentation, axis, f = _graf()
    rows = _graf_rows(presentation, axis)
    result = check_jacobian_syzygy(
        polynomial=f, rows=rows, ideal_generators=(f,), reduction_variable="z"
    )

    decoded = JacobianSyzygyCheckResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.status == "VERIFIED"
    assert decoded.rows == result.rows
    assert decoded.jacobian == result.jacobian


def test_request_decodes_from_strict_json_and_composes() -> None:
    tool = _tool()
    request = JacobianSyzygyCheckRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    payload = json.loads(request.model_dump_json())
    assert payload["reduction_variable"] == "z"
    assert payload["polynomial"]["presentation"]["characteristic"] == "2"
    result = check_jacobian_syzygy(
        polynomial=request.polynomial,
        rows=request.rows,
        ideal_generators=request.ideal_generators,
        reduction_variable=request.reduction_variable,
    )
    assert result.status == "VERIFIED"


def test_forged_ledger_verdict_fails_structural_decoding() -> None:
    presentation, axis, f = _graf()
    rows = _graf_rows(presentation, axis)
    result = check_jacobian_syzygy(
        polynomial=f, rows=rows, ideal_generators=(f,), reduction_variable="z"
    )

    forged = json.loads(result.model_dump_json())
    forged["ledger"][0]["verified"] = False
    with pytest.raises(ValidationError):
        JacobianSyzygyCheckResult.model_validate_json(json.dumps(forged))

    forged_status = json.loads(result.model_dump_json())
    forged_status["status"] = "REJECTED"
    with pytest.raises(ValidationError):
        JacobianSyzygyCheckResult.model_validate_json(json.dumps(forged_status))
