"""Known-answer and adversarial tests for integral binary quadratic form operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.quadratic_forms.binary._models import (
    MAX_REPRESENTATION_TARGET,
    MAX_REPRESENTATION_Y_CANDIDATES,
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormClassComposeRequest,
    BinaryQuadraticFormClassCompositionResult,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
    BinaryQuadraticFormRepresentationsRequest,
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperBinaryQuadraticFormClass,
    _representation_y_bound,
)
from jacobian.math.number_theory.quadratic_forms.binary._tools import (
    TOOLS,
    compute_check,
    compute_class_compose,
    compute_evaluate,
    compute_proper_equivalence,
    compute_reduce,
    compute_reduced_classes,
    compute_representations,
)
from jacobian.math.number_theory.quadratic_forms.binary.operations import (
    compose_classes,
    reduced_form,
)


def _positive_form(
    a: int, b: int, c: int
) -> PrimitivePositiveDefiniteBinaryQuadraticForm:
    return PrimitivePositiveDefiniteBinaryQuadraticForm(a=a, b=b, c=c)


def _proper_class(a: int, b: int, c: int) -> ProperBinaryQuadraticFormClass:
    return ProperBinaryQuadraticFormClass(representative=_positive_form(a, b, c))


def _assert_error_type(
    exc_info: pytest.ExceptionInfo[ValidationError | OperationDomainValidationError],
    code: str,
) -> None:
    assert any(error["type"] == code for error in exc_info.value.errors())


class TestCheck:
    def test_primitive_positive_definite(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=1, b=1, c=1))
        assert result.status == "PRIMITIVE_POSITIVE_DEFINITE"
        assert result.form == _positive_form(1, 1, 1)
        assert result.form.discriminant == -3
        assert "gram" not in result.model_dump(mode="json")

    def test_non_positive_definite(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=-1, b=0, c=1))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert result.obstruction is not None
        assert "positive" in result.obstruction.lower()
        assert result.form is None

    def test_nonnegative_discriminant(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=1, b=0, c=-1))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert result.obstruction is not None
        assert "discriminant" in result.obstruction.lower()

    def test_imprimitive(self) -> None:
        result = compute_check(BinaryQuadraticFormCheckRequest(a=2, b=2, c=2))
        assert result.status == "NOT_IN_INITIAL_DOMAIN"
        assert result.obstruction is not None
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
        assert result.form == _positive_form(1, 0, 1)


class TestEvaluate:
    def test_evaluate_at_origin(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(form=_positive_form(1, 1, 1), x=0, y=0)
        )
        assert result.value == 0
        assert not result.primitive

    def test_evaluate_at_1_0(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(form=_positive_form(1, 1, 1), x=1, y=0)
        )
        assert result.value == 1
        assert result.primitive

    def test_evaluate_at_2_3(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(form=_positive_form(2, 3, 5), x=2, y=3)
        )
        assert result.value == 2 * 4 + 3 * 6 + 5 * 9  # 8 + 18 + 45 = 71
        assert result.value == 71
        assert result.primitive

    def test_evaluate_rejects_raw_coefficient_fork(self) -> None:
        with pytest.raises(ValidationError):
            BinaryQuadraticFormEvaluateRequest.model_validate(
                {"a": 1, "b": 1, "c": 1, "x": 1, "y": 0, "value": 2, "primitive": True}
            )

    def test_evaluate_rejects_values_beyond_the_interoperable_integer_range(
        self,
    ) -> None:
        # Q(10^8, 0) = 10^16 > 2^53 - 1 for [1,0,1]: the request must be
        # rejected at admission instead of failing transport canonicalization.
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_evaluate(
                BinaryQuadraticFormEvaluateRequest(
                    form=_positive_form(1, 0, 1), x=100_000_000, y=0
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.evaluated_value_range"
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_evaluate(
                BinaryQuadraticFormEvaluateRequest(
                    form=_positive_form(1, 0, 1), x=-100_000_000, y=0
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.evaluated_value_range"
        )

    def test_evaluate_admits_the_largest_transportable_square(self) -> None:
        # 94_906_265 = floor_sqrt(2^53 - 1), so Q(x,0) = x^2 fits exactly.
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(
                form=_positive_form(1, 0, 1), x=94_906_265, y=0
            )
        )
        assert result.value == 9_007_199_136_250_225
        assert result.value <= 2**53 - 1

    def test_evaluate_boundary_one_above_the_interoperable_bound_is_rejected(
        self,
    ) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_evaluate(
                BinaryQuadraticFormEvaluateRequest(
                    form=_positive_form(1, 0, 1), x=94_906_266, y=0
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.evaluated_value_range"
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_evaluate(
                BinaryQuadraticFormEvaluateRequest(
                    form=_positive_form(1, 0, 1), x=-94_906_266, y=0
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.evaluated_value_range"
        )

    def test_evaluate_admission_tracks_the_exact_value_not_the_worst_case(
        self,
    ) -> None:
        # [1,-2,2] at (floor_sqrt(2^53-1), 1): the worst-case envelope
        # a*x^2 + abs(b*x*y) + c*y^2 exceeds the bound, but the exact value
        # 9_007_198_946_437_697 does not, so the request stays admitted.
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(
                form=_positive_form(1, -2, 2), x=94_906_265, y=1
            )
        )
        assert result.value == 9_007_198_946_437_697

    def test_maximal_admitted_evaluation_canonicalizes_for_transport(self) -> None:
        result = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(
                form=_positive_form(1, 0, 1), x=94_906_265, y=0
            )
        )
        canonicalize_json(result.model_dump(mode="json"))


class TestReduce:
    def test_semantically_invalid_form_is_a_typed_domain_rejection(self) -> None:
        request = BinaryQuadraticFormReduceRequest(form=_positive_form(-1, 3, 1))
        with pytest.raises(OperationDomainValidationError, match="positive definite"):
            compute_reduce(request)

    def test_reduce_already_reduced(self) -> None:
        result = compute_reduce(
            BinaryQuadraticFormReduceRequest(form=_positive_form(1, 0, 1))
        )
        assert result.reduced_form == _positive_form(1, 0, 1)

    def test_reduce_5_3_1(self) -> None:
        result = compute_reduce(
            BinaryQuadraticFormReduceRequest(form=_positive_form(5, 3, 1))
        )
        # D = 9 - 20 = -11, reduced form is [1,1,3]
        assert result.reduced_form == _positive_form(1, 1, 3)
        # Check the matrix has det 1
        p, q = result.matrix[0]
        r, s = result.matrix[1]
        assert p * s - q * r == 1

    def test_reduce_preserves_discriminant(self) -> None:
        for a, b, c in [(5, 3, 1), (7, 5, 3), (2, 1, 3), (10, 7, 2)]:
            result = compute_reduce(
                BinaryQuadraticFormReduceRequest(form=_positive_form(a, b, c))
            )
            d1 = b * b - 4 * a * c
            d2 = (
                result.reduced_form.b * result.reduced_form.b
                - 4 * result.reduced_form.a * result.reduced_form.c
            )
            assert d1 == d2, f"discriminant changed for [{a},{b},{c}]"

    def test_reduce_idempotent(self) -> None:
        result = compute_reduce(
            BinaryQuadraticFormReduceRequest(form=_positive_form(5, 3, 1))
        )
        # Reducing the reduced form should be idempotent
        result2 = compute_reduce(
            BinaryQuadraticFormReduceRequest(form=result.reduced_form)
        )
        assert result2.reduced_form == result.reduced_form


class TestProperEquivalence:
    def test_semantically_invalid_form_is_a_typed_domain_rejection(self) -> None:
        request = BinaryQuadraticFormProperEquivRequest(
            first=_positive_form(-1, 1, 1),
            second=_positive_form(1, 1, 1),
        )
        with pytest.raises(OperationDomainValidationError, match="positive definite"):
            compute_proper_equivalence(request)

    def test_self_equivalent(self) -> None:
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(
                first=_positive_form(1, 1, 1), second=_positive_form(1, 1, 1)
            )
        )
        assert result.status == "PROPERLY_EQUIVALENT"

    def test_different_discriminants_not_equivalent(self) -> None:
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(
                first=_positive_form(1, 1, 1), second=_positive_form(1, 0, 1)
            )
        )
        assert result.status == "NOT_PROPERLY_EQUIVALENT"

    def test_equivalent_forms(self) -> None:
        # [5,3,1] reduces to [1,1,3], and [1,1,3] is itself reduced
        # So [5,3,1] and [1,1,3] should be properly equivalent
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(
                first=_positive_form(5, 3, 1), second=_positive_form(1, 1, 3)
            )
        )
        assert result.status == "PROPERLY_EQUIVALENT"

    def test_non_equivalent_same_discriminant(self) -> None:
        # D=-23 has class number 3, so [1,1,6] and [2,1,3] are not equivalent
        result = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(
                first=_positive_form(1, 1, 6), second=_positive_form(2, 1, 3)
            )
        )
        assert result.status == "NOT_PROPERLY_EQUIVALENT"


class TestReducedClasses:
    def test_schema_exposes_the_reduced_class_scan_admission_condition(self) -> None:
        discriminant_schema = (
            BinaryQuadraticFormReducedClassesRequest.model_json_schema()["properties"][
                "discriminant"
            ]
        )
        assert "A*(A+2)" in discriminant_schema["description"]
        assert "floor_sqrt((-D)//3)+1" in discriminant_schema["description"]

    def test_example_states_the_scan_envelope_precondition(self) -> None:
        (tool,) = (
            candidate
            for candidate in TOOLS
            if candidate.operation_id
            == "number_theory.binary_quadratic_form.reduced_classes.compute"
        )
        description = str(tool.examples[0].description)
        assert "A*(A+2)" in description
        assert "is 8" in description

    def test_disc_neg_3(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-3)
        )
        assert result.class_number == 1
        assert result.classes == (_proper_class(1, 1, 1),)

    def test_disc_neg_4(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-4)
        )
        assert result.class_number == 1
        assert result.classes == (_proper_class(1, 0, 1),)

    def test_disc_neg_23(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-23)
        )
        assert result.class_number == 3
        # Verify all classes have the correct discriminant
        for form_class in result.classes:
            assert form_class.discriminant == -23

    def test_disc_neg_20(self) -> None:
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-20)
        )
        assert result.class_number == 2

    def test_exact_reduced_class_search_boundary_is_complete(self) -> None:
        # Here A=99, so the exact nested scan has A(A+2)=9,999 candidates.
        result = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-28_812)
        )
        assert result.class_number == len(result.classes)

    def test_reduced_class_search_just_over_budget_is_rejected(self) -> None:
        # Here A=100, so the exact nested scan would have 10,200 candidates.
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_reduced_classes(
                BinaryQuadraticFormReducedClassesRequest(discriminant=-29_404)
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.reduced_class_candidate_budget"
        )

    def test_non_discriminant_is_rejected_before_enumeration(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_reduced_classes(
                BinaryQuadraticFormReducedClassesRequest(discriminant=-5)
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.invalid_discriminant_congruence"
        )

    def test_all_classes_reduced(self) -> None:
        for D in [-3, -4, -7, -8, -11, -15, -19, -20, -23, -43, -47, -163]:  # noqa: N806
            result = compute_reduced_classes(
                BinaryQuadraticFormReducedClassesRequest(discriminant=D)
            )
            for form_class in result.classes:
                form = form_class.representative
                assert form.a > 0 and form.c > 0
                assert abs(form.b) <= form.a
                assert form.a <= form.c
                if abs(form.b) == form.a:
                    assert form.b >= 0
                if form.a == form.c:
                    assert form.b >= 0


class TestProperClassComposition:
    @staticmethod
    def _evaluate(
        form: PrimitivePositiveDefiniteBinaryQuadraticForm, x: int, y: int
    ) -> int:
        return form.a * x * x + form.b * x * y + form.c * y * y

    def _assert_direct_identity(
        self,
        first: ProperBinaryQuadraticFormClass,
        second: ProperBinaryQuadraticFormClass,
    ) -> None:
        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(first=first, second=second)
        )
        x_coefficients = result.direct_composition_map.x_coefficients
        y_coefficients = result.direct_composition_map.y_coefficients
        for x1 in range(-2, 3):
            for y1 in range(-2, 3):
                for x2 in range(-2, 3):
                    for y2 in range(-2, 3):
                        monomials = (x1 * x2, x1 * y2, y1 * x2, y1 * y2)
                        x = sum(
                            coefficient * monomial
                            for coefficient, monomial in zip(
                                x_coefficients, monomials, strict=True
                            )
                        )
                        y = sum(
                            coefficient * monomial
                            for coefficient, monomial in zip(
                                y_coefficients, monomials, strict=True
                            )
                        )
                        assert self._evaluate(result.composed_form, x, y) == (
                            self._evaluate(first.representative, x1, y1)
                            * self._evaluate(second.representative, x2, y2)
                        )
        p, q = result.reduction_matrix[0]
        r, s = result.reduction_matrix[1]
        assert p * s - q * r == 1
        source = result.composed_form
        transformed = _positive_form(
            source.a * p * p + source.b * p * r + source.c * r * r,
            2 * source.a * p * q + source.b * (p * s + q * r) + 2 * source.c * r * s,
            source.a * q * q + source.b * q * s + source.c * s * s,
        )
        assert transformed == result.product.representative
        assert result.product.representative.discriminant == first.discriminant

    def test_discriminant_neg_23_generator_squares_to_inverse(self) -> None:
        generator = _proper_class(2, -1, 3)

        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(first=generator, second=generator)
        )

        assert result.composed_form == _positive_form(4, 3, 2)
        assert result.product == _proper_class(2, 1, 3)
        self._assert_direct_identity(generator, generator)

    def test_common_divisor_case_squares_to_principal_class(self) -> None:
        nonprincipal = _proper_class(2, 2, 3)

        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(
                first=nonprincipal, second=nonprincipal
            )
        )

        assert result.composed_form == _positive_form(1, 0, 5)
        assert result.product == _proper_class(1, 0, 5)
        self._assert_direct_identity(nonprincipal, nonprincipal)

    @pytest.mark.parametrize("discriminant", [-20, -23, -31, -47, -56, -87])
    def test_complete_small_class_sets_form_an_associative_commutative_group(
        self, discriminant: int
    ) -> None:
        classes = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=discriminant)
        ).classes
        principal = _proper_class(
            1,
            discriminant % 2,
            ((discriminant % 2) - discriminant) // 4,
        )

        def product(
            first: ProperBinaryQuadraticFormClass,
            second: ProperBinaryQuadraticFormClass,
        ) -> ProperBinaryQuadraticFormClass:
            return compute_class_compose(
                BinaryQuadraticFormClassComposeRequest(first=first, second=second)
            ).product

        for first in classes:
            assert product(principal, first) == first
            assert product(first, principal) == first
            inverse_representative = reduced_form(
                _positive_form(
                    first.representative.a,
                    -first.representative.b,
                    first.representative.c,
                )
            )
            inverse = ProperBinaryQuadraticFormClass(
                representative=inverse_representative
            )
            assert product(first, inverse) == principal
            for second in classes:
                assert product(first, second) == product(second, first)
                assert product(first, second) in classes
                for third in classes:
                    assert product(product(first, second), third) == product(
                        first, product(second, third)
                    )

    def test_reduced_class_output_serializes_into_composition_unchanged(self) -> None:
        classes = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-23)
        )
        request = BinaryQuadraticFormClassComposeRequest.model_validate(
            {
                "first": classes.classes[1].model_dump(mode="json"),
                "second": classes.classes[2].model_dump(mode="json"),
            }
        )

        result = compute_class_compose(request)

        assert result.product == classes.classes[0]

    def test_composition_result_round_trips_with_all_source_bindings(self) -> None:
        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(
                first=_proper_class(2, -1, 3),
                second=_proper_class(2, -1, 3),
            )
        )

        assert (
            BinaryQuadraticFormClassCompositionResult.model_validate_json(
                result.model_dump_json()
            )
            == result
        )

    def test_composition_result_rejects_a_different_direct_form_discriminant(
        self,
    ) -> None:
        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(
                first=_proper_class(2, -1, 3),
                second=_proper_class(2, -1, 3),
            )
        ).model_dump(mode="json")
        result["composed_form"] = {"a": 1, "b": 0, "c": 1}

        with pytest.raises(
            ValidationError, match="bind the composed and product forms"
        ):
            BinaryQuadraticFormClassCompositionResult.model_validate(result)

    def test_catalog_example_exercises_nontrivial_proper_class_product(self) -> None:
        (tool,) = (
            candidate
            for candidate in TOOLS
            if candidate.operation_id
            == "number_theory.binary_quadratic_form.class_compose.compute"
        )
        request = tool.request_type.model_validate(tool.examples[0].input)

        result = tool.run(request)

        assert result.product == _proper_class(2, 1, 3)

    def test_nonreduced_representative_is_not_a_proper_class_value(self) -> None:
        carrier = ProperBinaryQuadraticFormClass(representative=_positive_form(5, 3, 1))
        assert carrier.representative == _positive_form(5, 3, 1)
        with pytest.raises(
            (OperationDomainValidationError, ValueError), match="reduced"
        ):
            compose_classes(carrier, carrier)

    def test_different_discriminants_are_rejected_before_composition(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_class_compose(
                BinaryQuadraticFormClassComposeRequest(
                    first=_proper_class(1, 1, 1),
                    second=_proper_class(1, 0, 1),
                )
            )
        _assert_error_type(
            exc_info,
            "integral_binary_quadratic_form.class_discriminant_mismatch",
        )

    def test_composition_admits_exact_reduced_class_scan_boundary(self) -> None:
        principal = _proper_class(1, 0, 7_203)

        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(first=principal, second=principal)
        )

        assert result.product == principal

    def test_composition_admits_discriminant_above_class_scan_budget(self) -> None:
        """Composition is O(1) + O(log|D|) reduction, not class enumeration."""
        principal = _proper_class(1, 0, 7_351)
        result = compute_class_compose(
            BinaryQuadraticFormClassComposeRequest(first=principal, second=principal)
        )
        assert result.product == principal

    def test_composition_rejects_mismatched_discriminants(self) -> None:
        first = _proper_class(1, 0, 7_351)
        second = _proper_class(1, 0, 7)
        with pytest.raises(ValueError, match="same discriminant"):
            compute_class_compose(
                BinaryQuadraticFormClassComposeRequest(first=first, second=second)
            )

    def test_composition_rejects_oversized_output_coefficients(self) -> None:
        """Composition of large discriminant forms can exceed the coefficient bound."""
        first = _proper_class(600_014, 0, 999_993)
        second = _proper_class(666_662, 0, 900_021)
        with pytest.raises(
            OperationDomainValidationError, match="composition product exceeds"
        ):
            compute_class_compose(
                BinaryQuadraticFormClassComposeRequest(first=first, second=second)
            )


class TestRepresentations:
    def test_schema_exposes_the_complete_y_scan_admission_condition(self) -> None:
        target_schema = BinaryQuadraticFormRepresentationsRequest.model_json_schema()[
            "properties"
        ]["target"]
        assert "2*floor_sqrt(4*a*n/(-D))+1" in target_schema["description"]

    def test_schema_exposes_the_modular_empty_admission_exception(self) -> None:
        target_schema = BinaryQuadraticFormRepresentationsRequest.model_json_schema()[
            "properties"
        ]["target"]
        assert "(a,b,c)=(1,0,1)" in target_schema["description"]
        assert "n mod 4 is 3" in target_schema["description"]

    def test_x_squared_plus_y_squared_of_five_has_eight_ordered_signed_pairs(
        self,
    ) -> None:
        result = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(
                form=_positive_form(1, 0, 1), target=5
            )
        )
        assert result.count == 8
        assert result.primitive_count == 8
        assert tuple((row.x, row.y) for row in result.representations) == (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )

    def test_primitive_count_is_distinct_from_raw_count(self) -> None:
        result = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(
                form=_positive_form(1, 0, 1), target=25
            )
        )
        assert result.count == 12
        assert result.primitive_count == 8

    def test_no_representation_is_complete_not_a_bound_limited_claim(self) -> None:
        result = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(
                form=_positive_form(1, 0, 1), target=3
            )
        )
        assert result.representations == ()
        assert result.count == result.primitive_count == 0

    def test_mod_four_obstruction_skips_an_inapplicable_general_scan(self) -> None:
        """A huge ``3 mod 4`` sum-of-two-squares target is proved empty first."""
        target = MAX_REPRESENTATION_TARGET - 1
        form = _positive_form(1, 0, 1)
        assert target % 4 == 3
        assert 2 * _representation_y_bound(form, target) + 1 > (
            MAX_REPRESENTATION_Y_CANDIDATES
        )

        result = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(form=form, target=target)
        )

        assert result.representations == ()
        assert result.count == result.primitive_count == 0
        assert type(result).model_validate(result.model_dump(mode="json")) == result

        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_representations(
                BinaryQuadraticFormRepresentationsRequest(
                    form=form, target=MAX_REPRESENTATION_TARGET
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.representation_candidate_budget"
        )

    def test_zero_has_the_single_nonprimitive_origin(self) -> None:
        result = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(
                form=_positive_form(1, 1, 1), target=0
            )
        )
        assert tuple(
            (row.x, row.y, row.primitive) for row in result.representations
        ) == ((0, 0, False),)

    def test_admission_rejects_unbounded_y_search_before_enumeration(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_representations(
                BinaryQuadraticFormRepresentationsRequest(
                    form=_positive_form(1_000_000, 0, 1),
                    target=1_000_000_000_000,
                )
            )
        _assert_error_type(
            exc_info, "integral_binary_quadratic_form.representation_candidate_budget"
        )


class TestCanonicalFormComposition:
    def test_kernel_results_compose_as_canonical_values(self) -> None:
        form = _positive_form(5, 3, 1)
        checked = compute_check(BinaryQuadraticFormCheckRequest(a=5, b=3, c=1))
        evaluated = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest(form=form, x=1, y=0)
        )
        reduced = compute_reduce(BinaryQuadraticFormReduceRequest(form=form))
        equivalent = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest(
                first=form, second=reduced.reduced_form
            )
        )
        representation_set = compute_representations(
            BinaryQuadraticFormRepresentationsRequest(form=form, target=5)
        )
        classes = compute_reduced_classes(
            BinaryQuadraticFormReducedClassesRequest(discriminant=-11)
        )

        assert checked.form == form
        assert evaluated.value == 5
        assert (
            reduced.reduced_form.a,
            reduced.reduced_form.b,
            reduced.reduced_form.c,
        ) == (
            1,
            1,
            3,
        )
        assert equivalent.status == "PROPERLY_EQUIVALENT"
        assert representation_set.count == len(representation_set.representations)
        assert classes.class_number == len(classes.classes)

    def test_checked_form_serializes_into_every_form_consumer(self) -> None:
        checked = compute_check(BinaryQuadraticFormCheckRequest(a=5, b=3, c=1))
        assert checked.form is not None
        serialized_form = checked.model_dump(mode="json")["form"]

        evaluated = compute_evaluate(
            BinaryQuadraticFormEvaluateRequest.model_validate(
                {"form": serialized_form, "x": 1, "y": 0}
            )
        )
        assert evaluated.form == checked.form
        assert evaluated.value == 5

        reduced = compute_reduce(
            BinaryQuadraticFormReduceRequest.model_validate({"form": serialized_form})
        )
        assert reduced.form == checked.form
        assert reduced.reduced_form == _positive_form(1, 1, 3)

        representations = compute_representations(
            BinaryQuadraticFormRepresentationsRequest.model_validate(
                {"form": serialized_form, "target": 5}
            )
        )
        assert representations.form == checked.form
        assert representations.count == 4
        witness = representations.representations[0]
        assert (
            compute_evaluate(
                BinaryQuadraticFormEvaluateRequest(
                    form=checked.form, x=witness.x, y=witness.y
                )
            ).value
            == representations.target
        )

        equivalence = compute_proper_equivalence(
            BinaryQuadraticFormProperEquivRequest.model_validate(
                {
                    "first": serialized_form,
                    "second": reduced.model_dump(mode="json")["reduced_form"],
                }
            )
        )
        assert equivalence.status == "PROPERLY_EQUIVALENT"
