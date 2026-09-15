"""Exact class and unit groups (#3717), backed by the PARI worker."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields._models import (
    MAX_CLASS_GROUP_DEGREE,
    NumberFieldClassGroupRequest,
    NumberFieldClassGroupResult,
    NumberFieldUnitGroupRequest,
    NumberFieldUnitGroupResult,
)
from jacobian.math.number_theory.number_fields.operations import (
    class_group,
    discriminant,
    embeddings,
    unit_group,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)

pytestmark = pytest.mark.requires_backend("pari")


def _field(*coefficients: int) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(coefficients_descending=tuple(coefficients))


def test_imaginary_quadratic_class_two() -> None:
    """QQ(sqrt(-5)) has class number 2 with cyclic structure C2."""
    result = class_group(_field(1, 0, 5))
    assert result.class_number == 2
    assert result.abelian_invariants == (2,)
    assert result.field_discriminant == -20
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (0, 1)
    assert len(result.ideal_representatives) == 1
    # The retained HNF basis vectors generate an ideal of norm two.
    basis = result.ideal_representatives[0].entries
    assert basis == ((2, 0), (1, 1))


def test_imaginary_quadratic_class_three() -> None:
    result = class_group(_field(1, 0, 23))
    assert result.class_number == 3
    assert result.abelian_invariants == (3,)


def test_real_quadratic_trivial_class_group() -> None:
    result = class_group(_field(1, 0, -2))
    assert result.class_number == 1
    assert result.abelian_invariants == ()
    assert result.ideal_representatives == ()
    assert result.field_discriminant == 8
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (2, 0)


def test_real_quadratic_fundamental_unit() -> None:
    """QQ(sqrt(2)) has rank 1 and fundamental unit up to sign 1 + sqrt(2)."""
    result = unit_group(_field(1, 0, -2))
    assert result.rank == 1
    assert result.torsion_order == 2
    assert len(result.fundamental_units) == 1
    coefficients = tuple(
        component.as_fraction()
        for component in result.fundamental_units[0].coefficients_ascending
    )
    assert coefficients in ((Fraction(1), Fraction(1)), (Fraction(-1), Fraction(-1)))


def test_imaginary_quadratic_unit_rank_zero() -> None:
    result = unit_group(_field(1, 0, 5))
    assert result.rank == 0
    assert result.torsion_order == 2


def test_cubic_signature_and_torsion() -> None:
    """x^3 - x - 1 has one real and one complex pair, with a single unit."""
    result = unit_group(_field(1, 0, -1, -1))
    assert (result.real_embedding_count, result.complex_embedding_pair_count) == (1, 1)
    assert result.rank == 1
    assert result.torsion_order == 2


def test_reducible_polynomial_rejected() -> None:
    """x^2 - 1 is reducible, so it does not define a field."""
    with pytest.raises(OperationDomainValidationError, match="irreducible"):
        class_group(_field(1, 0, -1))


def test_degree_bound_rejected_natively() -> None:
    coefficients = (1,) + (0,) * MAX_CLASS_GROUP_DEGREE + (1,)
    with pytest.raises(OperationDomainValidationError):
        class_group(_field(*coefficients))


def test_request_validators_gate_degree() -> None:
    from pydantic import ValidationError

    coefficients = (1,) + (0,) * MAX_CLASS_GROUP_DEGREE + (1,)
    with pytest.raises(ValidationError):
        NumberFieldClassGroupRequest(
            field=SimpleNumberFieldPresentation(coefficients_descending=coefficients)
        )


def test_serialized_results_round_trip() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldClassGroupResult,
        NumberFieldUnitGroupResult,
    )

    class_result = class_group(_field(1, 0, 5))
    restored = NumberFieldClassGroupResult.model_validate_json(
        encode_strict_json(class_result.model_dump(mode="json")), strict=True
    )
    assert restored.class_number == 2

    unit_result = unit_group(_field(1, 0, -2))
    restored_unit = NumberFieldUnitGroupResult.model_validate_json(
        encode_strict_json(unit_result.model_dump(mode="json")), strict=True
    )
    assert restored_unit.rank == 1
    assert restored_unit.fundamental_units == unit_result.fundamental_units


def test_forged_rank_signature_rejected() -> None:
    from pydantic import ValidationError

    from jacobian.canonical import encode_strict_json
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldUnitGroupResult,
    )

    result = unit_group(_field(1, 0, -2))
    payload = result.model_dump(mode="json")
    payload["rank"] = 0
    payload["fundamental_units"] = []
    with pytest.raises(ValidationError):
        NumberFieldUnitGroupResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


def test_unit_request_validates_field_type() -> None:
    with pytest.raises(
        OperationDomainValidationError, match="SimpleNumberFieldPresentation"
    ):
        unit_group("not a field")  # type: ignore[arg-type]


def test_class_group_request_admits_native_field() -> None:
    request = NumberFieldClassGroupRequest(field=_field(1, 0, 5))
    assert request.field.degree == 2
    unit_request = NumberFieldUnitGroupRequest(field=_field(1, 0, 5))
    assert unit_request.field.degree == 2


def _ascending_fractions(element: SimpleNumberFieldElement) -> list[Fraction]:
    return [component.as_fraction() for component in element.coefficients_ascending]


def _quotient_mul(
    field: SimpleNumberFieldPresentation,
    left: list[Fraction],
    right: list[Fraction],
) -> list[Fraction]:
    degree = field.degree
    leading = Fraction(int(field.coefficients_descending[0]))
    descending = [Fraction(int(c)) for c in field.coefficients_descending]
    product = [Fraction(0)] * (2 * degree - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    for k in range(2 * degree - 2, degree - 1, -1):
        factor = product[k] / leading
        if factor:
            for i in range(1, degree + 1):
                product[k - i] -= factor * descending[i]
    return product[:degree]


def _quotient_pow(
    field: SimpleNumberFieldPresentation,
    base: list[Fraction],
    exponent: int,
) -> list[Fraction]:
    result = [Fraction(1)] + [Fraction(0)] * (field.degree - 1)
    while exponent:
        if exponent & 1:
            result = _quotient_mul(field, result, base)
        base = _quotient_mul(field, base, base)
        exponent >>= 1
    return result


def _field_norm(
    field: SimpleNumberFieldPresentation, element: SimpleNumberFieldElement
) -> Fraction:
    """Independent norm oracle: Norm(A(alpha)) via the f/A resultant."""
    from sympy import Poly, Rational, Symbol

    x = Symbol("x")
    defining = Poly([int(c) for c in field.coefficients_descending], x, domain="ZZ")
    coefficients = _ascending_fractions(element)
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    if len(coefficients) == 1:
        return coefficients[0] ** field.degree
    image = Poly(
        [Rational(c.numerator, c.denominator) for c in reversed(coefficients)],
        x,
        domain="QQ",
    )
    degree = image.degree()
    assert isinstance(degree, int)
    resultant = defining.resultant(image)
    # Norm(A(alpha)) = prod A(alpha_i) = (-1)^(n*m) Res(f, A) / lc(f)^m.
    return Fraction(int(resultant * (-1) ** (field.degree * degree))) / (
        Fraction(int(field.coefficients_descending[0])) ** degree
    )


def _verify_unit_group_claim(
    field: SimpleNumberFieldPresentation, claim: NumberFieldUnitGroupResult
) -> bool:
    if claim.field != field:
        return False
    if claim.rank != len(claim.fundamental_units):
        return False
    profile = embeddings(field)
    if (
        claim.real_embedding_count != profile.signature.real_embedding_count
        or claim.complex_embedding_pair_count
        != profile.signature.complex_conjugate_pair_count
    ):
        return False
    if claim.field_discriminant != discriminant(field):
        return False
    units = (*claim.fundamental_units, claim.torsion_generator)
    if any(u.presentation != field for u in units):
        return False
    if any(_field_norm(field, u) not in (Fraction(1), Fraction(-1)) for u in units):
        return False
    torsion = _ascending_fractions(claim.torsion_generator)
    return _quotient_pow(field, torsion, claim.torsion_order) == [Fraction(1)] + [
        Fraction(0)
    ] * (field.degree - 1)


def _verify_class_group_claim(
    field: SimpleNumberFieldPresentation, claim: NumberFieldClassGroupResult
) -> bool:
    from math import prod

    if claim.field != field:
        return False
    if prod(claim.abelian_invariants, start=1) != claim.class_number:
        return False
    if any(
        right % left != 0
        for left, right in zip(
            claim.abelian_invariants, claim.abelian_invariants[1:], strict=False
        )
    ):
        return False
    nontrivial = [d for d in claim.abelian_invariants if d > 1]
    if len(claim.ideal_representatives) != len(nontrivial):
        return False
    for matrix in claim.ideal_representatives:
        if matrix.row_count != field.degree or matrix.column_count != field.degree:
            return False
    profile = embeddings(field)
    if (
        claim.real_embedding_count != profile.signature.real_embedding_count
        or claim.complex_embedding_pair_count
        != profile.signature.complex_conjugate_pair_count
    ):
        return False
    return bool(claim.field_discriminant == discriminant(field))


def test_unit_claims_satisfy_norm_order_and_signature_oracles() -> None:
    field = _field(1, 0, -2)
    result = unit_group(field)
    assert _verify_unit_group_claim(field, result)
    assert _field_norm(field, result.fundamental_units[0]) == -1
    # The fundamental unit is defined up to sign; both associates are units.
    negated = result.fundamental_units[0].model_copy(
        update={
            "coefficients_ascending": tuple(
                type(c)(num=-c.num, den=c.den)
                for c in result.fundamental_units[0].coefficients_ascending
            )
        }
    )
    assert _field_norm(field, negated) == -1

    cubic = _field(1, 0, -1, -1)
    cubic_result = unit_group(cubic)
    assert _verify_unit_group_claim(cubic, cubic_result)

    gaussian = _field(1, 0, 1)
    gaussian_result = unit_group(gaussian)
    assert gaussian_result.torsion_order == 4
    assert _verify_unit_group_claim(gaussian, gaussian_result)


def test_forged_unit_claims_fail_the_independent_oracle() -> None:
    field = _field(1, 0, -2)
    result = unit_group(field)
    assert not _verify_unit_group_claim(
        field, result.model_copy(update={"rank": result.rank + 1})
    )
    non_unit = result.fundamental_units[0].model_copy(
        update={
            "coefficients_ascending": tuple(
                type(c)(num=c.num + 1, den=c.den)
                for c in result.fundamental_units[0].coefficients_ascending
            )
        }
    )
    forged_units = (non_unit,) * len(result.fundamental_units)
    assert not _verify_unit_group_claim(
        field, result.model_copy(update={"fundamental_units": forged_units})
    )
    assert not _verify_unit_group_claim(_field(1, 0, 5), result)


def test_class_group_claims_satisfy_invariant_and_composition_oracles() -> None:
    field = _field(1, 0, 5)
    result = class_group(field)
    assert _verify_class_group_claim(field, result)
    assert result.abelian_invariants == (2,)
    matrix = result.ideal_representatives[0]
    determinant = matrix.entries[0][0] * matrix.entries[1][1] - (
        matrix.entries[0][1] * matrix.entries[1][0]
    )
    assert abs(determinant) == 2

    trivial = _field(1, 0, -2)
    trivial_result = class_group(trivial)
    assert _verify_class_group_claim(trivial, trivial_result)


def test_forged_class_group_claims_fail_the_independent_oracle() -> None:
    field = _field(1, 0, 5)
    result = class_group(field)
    assert not _verify_class_group_claim(
        field, result.model_copy(update={"class_number": 3})
    )
    assert not _verify_class_group_claim(
        field, result.model_copy(update={"abelian_invariants": (2, 3)})
    )
    assert not _verify_class_group_claim(
        field, result.model_copy(update={"ideal_representatives": ()})
    )
    assert not _verify_class_group_claim(_field(1, 0, -2), result)


def test_missing_pari_backend_is_a_typed_refusal_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cypari", None)
    with pytest.raises(OperationResourceAdmissionError) as error:
        class_group(_field(1, 0, 5))
    assert error.value.errors()[0]["type"] == "number_field.pari_backend_unavailable"
    with pytest.raises(OperationResourceAdmissionError):
        unit_group(_field(1, 0, -2))
