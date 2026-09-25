"""Exact finite algebra values for degree-truncated free-algebra quotients."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    TruncatedFreeAlgebraQuotient,
    truncated_quotient_algebra,
)
from jacobian.math.free_algebras._models import canonical_word_key


def _polynomial(
    alphabet: tuple[str, ...], values: dict[tuple[str, ...], Fraction | int]
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
            word=word,
        )
        for word, coefficient in sorted(
            values.items(),
            key=lambda entry: canonical_word_key(alphabet, entry[0]),
            reverse=True,
        )
        if coefficient
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _ideal(
    alphabet: tuple[str, ...], generators: tuple[FreeAlgebraPolynomial, ...]
) -> FreeAlgebraIdeal:
    return FreeAlgebraIdeal(alphabet=alphabet, generators=generators, side="two-sided")


def _coordinates(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _multiply_coordinates(
    algebra: TruncatedFreeAlgebraQuotient,
    left: dict[tuple[str, ...], Fraction],
    right: dict[tuple[str, ...], Fraction],
) -> dict[tuple[str, ...], Fraction]:
    positions = {word: index for index, word in enumerate(algebra.basis_words)}
    result: dict[tuple[str, ...], Fraction] = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            value = algebra.multiplication[positions[left_word]][positions[right_word]]
            for term in value.terms:
                result[term.word] = result.get(term.word, Fraction(0)) + (
                    left_coefficient
                    * right_coefficient
                    * term.coefficient.as_fraction()
                )
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def test_commutator_quotient_is_a_truncated_commutative_polynomial_algebra() -> None:
    alphabet = ("x", "y")
    relation = _polynomial(alphabet, {("x", "y"): 1, ("y", "x"): -1})
    algebra = truncated_quotient_algebra(_ideal(alphabet, (relation,)), 3)

    expected_basis = tuple(
        ("x",) * x_degree + ("y",) * (degree - x_degree)
        for degree in range(4)
        for x_degree in range(degree, -1, -1)
    )
    assert algebra.basis_words == expected_basis
    assert algebra.unit == _polynomial(alphabet, {(): 1})

    positions = {word: index for index, word in enumerate(algebra.basis_words)}
    for left in algebra.basis_words:
        for right in algebra.basis_words:
            product = algebra.multiplication[positions[left]][positions[right]]
            if len(left) + len(right) > 3:
                assert not product.terms
            else:
                expected = ("x",) * (left.count("x") + right.count("x")) + ("y",) * (
                    left.count("y") + right.count("y")
                )
                assert _coordinates(product) == {expected: Fraction(1)}

    for left in algebra.basis_words:
        left_coordinates = {left: Fraction(1)}
        assert (
            _multiply_coordinates(algebra, _coordinates(algebra.unit), left_coordinates)
            == left_coordinates
        )
        for middle in algebra.basis_words:
            middle_coordinates = {middle: Fraction(1)}
            for right in algebra.basis_words:
                right_coordinates = {right: Fraction(1)}
                left_associated = _multiply_coordinates(
                    algebra,
                    _multiply_coordinates(
                        algebra, left_coordinates, middle_coordinates
                    ),
                    right_coordinates,
                )
                right_associated = _multiply_coordinates(
                    algebra,
                    left_coordinates,
                    _multiply_coordinates(
                        algebra, middle_coordinates, right_coordinates
                    ),
                )
                assert left_associated == right_associated

    restored = type(algebra).model_validate_json(algebra.model_dump_json())
    assert restored == algebra


def test_rational_relation_coefficients_survive_in_the_table() -> None:
    alphabet = ("x", "y")
    relation = _polynomial(
        alphabet,
        {("y", "x"): 1, ("x", "y"): Fraction(-1, 2)},
    )
    algebra = truncated_quotient_algebra(_ideal(alphabet, (relation,)), 2)
    positions = {word: index for index, word in enumerate(algebra.basis_words)}

    product = algebra.multiplication[positions[("y",)]][positions[("x",)]]
    assert _coordinates(product) == {("x", "y"): Fraction(1, 2)}


def test_free_and_zero_quotients_keep_the_correct_unit_and_cutoff() -> None:
    free = truncated_quotient_algebra(_ideal(("x",), ()), 2)
    positions = {word: index for index, word in enumerate(free.basis_words)}
    assert free.basis_words == ((), ("x",), ("x", "x"))
    assert _coordinates(free.multiplication[positions[("x",)]][positions[("x",)]]) == {
        ("x", "x"): Fraction(1)
    }
    assert not free.multiplication[positions[("x", "x")]][positions[("x",)]].terms

    zero = truncated_quotient_algebra(
        _ideal(("x",), (_polynomial(("x",), {(): 1}),)), 2
    )
    assert zero.basis_words == ()
    assert zero.multiplication == ()
    assert not zero.unit.terms


def test_decoded_carrier_shape_is_not_a_semantic_certificate() -> None:
    alphabet = ("x",)
    relation = _polynomial(alphabet, {("x",): 1})
    algebra = truncated_quotient_algebra(_ideal(alphabet, (relation,)), 1)
    assert algebra.basis_words == ((),)

    # JSON carries the source relation and completed presentation, but wire
    # validation deliberately checks shape rather than replaying completion or
    # multiplication. A consumer supplied such a value must check any table
    # relation on which its own result depends.
    forged = algebra.model_dump(mode="json")
    forged["basis_words"] = [[], ["x"]]
    zero = {"alphabet": ["x"], "terms": []}
    forged["multiplication"] = [[zero, zero], [zero, zero]]
    decoded = TruncatedFreeAlgebraQuotient.model_validate_json(json.dumps(forged))
    assert decoded.basis_words == ((), ("x",))
    assert decoded.multiplication[0][0].terms == ()
    assert decoded.unit.terms[0].word == ()


def test_nonhomogeneous_ideal_is_rejected() -> None:
    alphabet = ("x",)
    nonhomogeneous = _polynomial(alphabet, {(): 1, ("x",): 1})
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        truncated_quotient_algebra(_ideal(alphabet, (nonhomogeneous,)), 2)


def test_output_bound_precedes_multiplication_table_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.free_algebras.operations as operations

    def no_table(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("multiplication table construction began before admission")

    monkeypatch.setattr(operations, "_encode", no_table)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        truncated_quotient_algebra(_ideal(("x",), ()), 32)
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.truncated_quotient_output_bytes"
    )


def test_table_term_bound_precedes_multiplication_table_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.free_algebras.operations as operations

    def no_table(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("multiplication table construction began before admission")

    monkeypatch.setattr(operations, "_truncated_multiplication_table", no_table)
    monomial = _polynomial(("x", "y"), {("x", "x", "x"): 1})
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        truncated_quotient_algebra(_ideal(("x", "y"), (monomial,)), 6)
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.truncated_quotient_table_terms"
    )


def test_chained_reduction_coefficient_growth_is_admitted_before_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.free_algebras.operations as operations

    alphabet = ("x", "y")
    relation = _polynomial(alphabet, {("y", "x"): 1, ("x", "y"): -(10**16)})

    def no_table(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("table construction began before coefficient admission")

    monkeypatch.setattr(operations, "_truncated_multiplication_table", no_table)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        truncated_quotient_algebra(_ideal(alphabet, (relation,)), 4)
    assert (
        exc_info.value.errors()[0]["type"]
        == "free_algebra.gs_coefficient_growth_budget"
    )


def test_reducer_coefficient_ratio_is_admitted_before_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.free_algebras.operations as operations

    alphabet = ("x", "y")
    # Each crossing multiplies the coefficient by the 16-digit ratio between
    # the 10^8 tail and the 1/10^8 leading coefficient, even though every
    # individual component is only 9 digits wide.
    relation = _polynomial(
        alphabet,
        {("y", "x"): Fraction(1, 10**8), ("x", "y"): -(10**8)},
    )

    def no_table(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("table construction began before coefficient admission")

    monkeypatch.setattr(operations, "_truncated_multiplication_table", no_table)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        truncated_quotient_algebra(_ideal(alphabet, (relation,)), 4)
    assert (
        exc_info.value.errors()[0]["type"]
        == "free_algebra.gs_coefficient_growth_budget"
    )


def test_admitted_reducer_ratio_still_builds_the_table() -> None:
    alphabet = ("x", "y")
    # A ratio of 4 per crossing stays inside the coefficient envelope, so the
    # strengthened admission must not reject it.
    relation = _polynomial(
        alphabet,
        {("y", "x"): Fraction(1, 2), ("x", "y"): -2},
    )
    algebra = truncated_quotient_algebra(_ideal(alphabet, (relation,)), 3)
    positions = {word: index for index, word in enumerate(algebra.basis_words)}

    product = algebra.multiplication[positions[("y",)]][positions[("x",)]]
    assert _coordinates(product) == {("x", "y"): Fraction(4)}


def test_catalog_example_round_trips_and_is_discoverable() -> None:
    operation_id = "free_algebra.two_sided_quotient.truncated_algebra.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, TruncatedFreeAlgebraQuotient)
    assert len(result.basis_words) == 6
