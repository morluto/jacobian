"""Known-answer and adversarial tests for integral binary quadratic form operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.integral_binary_quadratic_forms._models import (
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
)
from jacobian.math.integral_binary_quadratic_forms._operations import (
    compute_check,
    compute_evaluate,
    compute_proper_equivalence,
    compute_reduce,
    compute_reduced_classes,
)


class TestCheck:
    def test_primitive_positive_definite(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=1, b=1, c=1))
        assert result.status == "PRIMITIVE_POSITIVE_DEFINITE"
        assert result.discriminant == -3
        assert result.gram == ((1, 1), (1, 1))

    def test_non_positive_definite(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=-1, b=0, c=1))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert "positive" in result.obstruction.lower()

    def test_nonnegative_discriminant(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=1, b=0, c=-1))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert "discriminant" in result.obstruction.lower()

    def test_imprimitive(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=2, b=2, c=2))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert "primitive" in result.obstruction.lower()

    def test_invalid_discriminant_congruence(self) -> None:
        # D = 1 - 4*1*1 = -3, which is valid (D ≡ 1 mod 4)
        # Let's find one with D ≡ 2 mod 4: b=0, a=1, c=1 -> D=-4 ≡ 0 mod 4, valid
        # b=1, a=1, c=2 -> D=1-8=-7 ≡ 1 mod 4, valid
        # We need D ≡ 2 or 3 mod 4: b=0, a=1, c=2 -> D=-8 ≡ 0 mod 4
        # Actually any D = b^2 - 4ac. If b is even, D ≡ 0 mod 4. If b is odd, D ≡ 1 mod 4.
        # So D is always 0 or 1 mod 4! The congruence check is always satisfied.
        # The check is still there for safety. Let's just test a valid case.
        result = compute_check(BinaryQuadraticFormCheckRequest(a=1, b=0, c=1))
        assert result.status == "PRIMITIVE_POSITIVE_DEFINITE"
        assert result.discriminant == -4


class TestEvaluate:
    def test_evaluate_at_origin(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(a=1, b=1, c=1, x=0, y=0)
        )
        assert result.value == 0
        assert not result.primitive

    def test_evaluate_at_1_0(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(a=1, b=1, c=1, x=1, y=0)
        )
        assert result.value == 1
        assert result.primitive

    def test_evaluate_at_2_3(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(a=2, b=3, c=5, x=2, y=3)
        )
        assert result.value == 2 * 4 + 3 * 6 + 5 * 9  # 8 + 18 + 45 = 71
        assert result.value == 71
        assert result.primitive

    def test_evaluate_wrong_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BinaryQuadraticFormEvaluateRequest.model_validate(
                {"a": 1, "b": 1, "c": 1, "x": 1, "y": 0, "value": 2, "primitive": True}
            )


class TestReduce:
    def test_reduce_already_reduced(self) -> None:
        result = compute_reduce(BinaryQuadraticFormReduceRequest(a=1, b=0, c=1))
        assert result.reduced_a == 1
        assert result.reduced_b == 0
        assert result.reduced_c == 1

    def test_reduce_5_3_1(self) -> None:
        result = compute_reduce(BinaryQuadraticFormReduceRequest(a=5, b=3, c=1))
        # D = 9 - 20 = -11, reduced form is [1,1,3]
        assert result.reduced_a == 1
        assert result.reduced_b == 1
        assert result.reduced_c == 3
        # Check the matrix has det 1
        p, q = result.matrix[0]
        r, s = result.matrix[1]
        assert p * s - q * r == 1

    def test_reduce_preserves_discriminant(self) -> None:
        for a, b, c in [(5, 3, 1), (7, 5, 3), (2, 1, 3), (10, 7, 2)]:
            result = compute_reduce(BinaryQuadraticFormReduceRequest(a=a, b=b, c=c))
            d1 = b * b - 4 * a * c
            d2 = (
                result.reduced_b * result.reduced_b
                - 4 * result.reduced_a * result.reduced_c
            )
            assert d1 == d2, f"discriminant changed for [{a},{b},{c}]"

    def test_reduce_idempotent(self) -> None:
        result = compute_reduce(BinaryQuadraticFormReduceRequest(a=5, b=3, c=1))
        # Reducing the reduced form should be idempotent
        result2 = compute_reduce(
            BinaryQuadraticFormReduceRequest(
                a=result.reduced_a, b=result.reduced_b, c=result.reduced_c
            )
        )
        assert (result2.reduced_a, result2.reduced_b, result2.reduced_c) == (
            result.reduced_a,
            result.reduced_b,
            result.reduced_c,
        )


class TestProperEquivalence:
    def test_self_equivalent(self) -> None:
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(form1=(1, 1, 1), form2=(1, 1, 1))
        )
        assert result.status == "PROPERLY_EQUIVALENT"

    def test_different_discriminants_not_equivalent(self) -> None:
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(form1=(1, 1, 1), form2=(1, 0, 1))
        )
        assert result.status == "NOT_PROPERLY_EQUIVALENT"

    def test_equivalent_forms(self) -> None:
        # [5,3,1] reduces to [1,1,3], and [1,1,3] is itself reduced
        # So [5,3,1] and [1,1,3] should be properly equivalent
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(form1=(5, 3, 1), form2=(1, 1, 3))
        )
        assert result.status == "PROPERLY_EQUIVALENT"

    def test_non_equivalent_same_discriminant(self) -> None:
        # D=-23 has class number 3, so [1,1,6] and [2,1,3] are not equivalent
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(form1=(1, 1, 6), form2=(2, 1, 3))
        )
        assert result.status == "NOT_PROPERLY_EQUIVALENT"


class TestReducedClasses:
    def test_disc_neg_3(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-3)
        )
        assert result.class_number == 1
        assert result.classes == ((1, 1, 1),)

    def test_disc_neg_4(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-4)
        )
        assert result.class_number == 1
        assert result.classes == ((1, 0, 1),)

    def test_disc_neg_23(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-23)
        )
        assert result.class_number == 3
        # Verify all classes have the correct discriminant
        for a, b, c in result.classes:
            assert b * b - 4 * a * c == -23

    def test_disc_neg_20(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-20)
        )
        assert result.class_number == 2

    def test_all_classes_reduced(self) -> None:
        for D in [-3, -4, -7, -8, -11, -15, -19, -20, -23, -43, -47, -163]:  # noqa: N806
            result = compute_reduced_classes(
                BinaryQuadraticFormReducedClassesRequest(discriminant=D)
            )
            for a, b, c in result.classes:
                assert a > 0 and c > 0
                assert abs(b) <= a
                assert a <= c
                if abs(b) == a:
                    assert b >= 0
                if a == c:
                    assert b >= 0
