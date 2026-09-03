"""Exact simple number fields and complete Archimedean embedding profiles."""

from __future__ import annotations

import cProfile
import time
from fractions import Fraction
from threading import Event
from types import CodeType

import pytest
from pydantic import ValidationError
from tests.math.number_theory.number_fields._embedding_invariants import (
    require_real_interval_selects_root,
    require_rectangle_selects_root,
)

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory import number_fields
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
    RationalComplexIsolatingRectangle,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields import (
    NumberFieldEmbeddingProfile,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    embeddings,
)
from jacobian.math.number_theory.number_fields._embedding_protocol import (
    NumberFieldEmbeddingWorkerComplete,
    NumberFieldEmbeddingWorkerRequest,
)
from jacobian.math.number_theory.number_fields._embeddings_worker import (
    compute_embeddings_worker_response,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldEmbeddingsRequest,
)
from jacobian.math.number_theory.number_fields._tools import TOOLS
from jacobian.math.number_theory.number_fields.operations import (
    NumberFieldEmbeddingAdmissionError,
    _admit_number_field_embeddings,
)
from jacobian.math.number_theory.number_fields.values import (
    ComplexNumberFieldEmbeddingRecord,
    RealNumberFieldEmbedding,
    RealNumberFieldEmbeddingRecord,
)
from jacobian.process import bounded_process_cancellation


def _field(*coefficients: str) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(
        coefficients_descending=coefficients,
    )


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _worker_request(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldEmbeddingWorkerRequest:
    admission = _admit_number_field_embeddings(field)
    return NumberFieldEmbeddingWorkerRequest(
        field=field,
        root_isolation_bits=admission.root_isolation_bits,
        evidence_grid_bits=admission.evidence_grid_bits,
    )


def test_rational_field_degree_one_is_the_cheapest_complete_profile() -> None:
    field = _field("1", "0")

    result = embeddings(field)

    assert field.degree == 1
    assert result.signature.real_embedding_count == 1
    assert result.signature.complex_conjugate_pair_count == 0
    assert result.defining_polynomial_discriminant == "1"
    assert result.complex_conjugate_pairs == ()
    assert isinstance(result.records[0], RealNumberFieldEmbeddingRecord)
    assert result.records[0].isolating_interval.lower.as_fraction() == 0
    assert result.records[0].isolating_interval.interval_type == "SINGLETON"


def test_sqrt_two_has_two_ordered_real_embeddings_and_polynomial_discriminant() -> None:
    result = embeddings(_field("1", "0", "-2"))

    assert result.signature.real_embedding_count == 2
    assert result.signature.complex_conjugate_pair_count == 0
    assert result.defining_polynomial_discriminant == "8"
    assert all(
        isinstance(record, RealNumberFieldEmbeddingRecord) for record in result.records
    )
    left, right = result.records
    assert isinstance(left, RealNumberFieldEmbeddingRecord)
    assert isinstance(right, RealNumberFieldEmbeddingRecord)
    assert (
        left.isolating_interval.upper.as_fraction()
        <= right.isolating_interval.lower.as_fraction()
    )


def test_gaussian_field_distinguishes_i_from_negative_i_exactly() -> None:
    result = embeddings(_field("1", "0", "1"))

    assert result.signature.real_embedding_count == 0
    assert result.signature.complex_conjugate_pair_count == 1
    assert result.defining_polynomial_discriminant == "-4"
    assert len(result.complex_conjugate_pairs) == 1
    pair = result.complex_conjugate_pairs[0]
    assert (pair.negative_embedding_index, pair.positive_embedding_index) == (0, 1)
    negative, positive = result.records
    assert isinstance(negative, ComplexNumberFieldEmbeddingRecord)
    assert isinstance(positive, ComplexNumberFieldEmbeddingRecord)
    assert negative.embedding.root.root_index == 0
    assert positive.embedding.root.root_index == 1
    assert negative.half_plane == "NEGATIVE_IMAGINARY"
    assert positive.half_plane == "POSITIVE_IMAGINARY"
    assert negative.isolating_rectangle.conjugate() == positive.isolating_rectangle


def test_signature_one_one_cubic_is_complete() -> None:
    result = embeddings(_field("1", "0", "-1", "1"))

    assert result.signature.real_embedding_count == 1
    assert result.signature.complex_conjugate_pair_count == 1
    assert result.defining_polynomial_discriminant == "-23"
    assert [record.kind for record in result.records] == [
        "REAL",
        "COMPLEX",
        "COMPLEX",
    ]
    assert result.complex_conjugate_pairs[0].positive_embedding_index == 2


def test_pair_order_uses_positive_representatives_not_full_root_lexicography() -> None:
    # The full lexicographic order of the four roots of this irreducible
    # polynomial is not conjugate-pair adjacent.  The public convention orders
    # the positive representatives by imaginary part, then emits each pair as
    # negative followed by positive.
    result = embeddings(_field("1", "0", "5", "0", "5"))

    assert result.signature.complex_conjugate_pair_count == 2
    first_positive = result.records[1]
    second_positive = result.records[3]
    assert isinstance(first_positive, ComplexNumberFieldEmbeddingRecord)
    assert isinstance(second_positive, ComplexNumberFieldEmbeddingRecord)
    assert (
        first_positive.isolating_rectangle.imaginary_upper.as_fraction()
        < second_positive.isolating_rectangle.imaginary_lower.as_fraction()
    )
    assert tuple(
        (pair.negative_embedding_index, pair.positive_embedding_index)
        for pair in result.complex_conjugate_pairs
    ) == ((0, 1), (2, 3))


def test_embedding_identity_is_independent_of_valid_isolation_evidence() -> None:
    result = embeddings(_field("1", "0", "1"))
    original = result.records[0]
    assert isinstance(original, ComplexNumberFieldEmbeddingRecord)
    wider_evidence = RationalComplexIsolatingRectangle(
        real_lower=_rational(Fraction(-1, 2)),
        real_upper=_rational(Fraction(1, 2)),
        imaginary_lower=_rational(Fraction(-3, 2)),
        imaginary_upper=_rational(Fraction(-1, 2)),
    )
    assert (
        require_rectangle_selects_root(original.embedding.root, wider_evidence)
        == "NEGATIVE_IMAGINARY"
    )

    alternate_record = ComplexNumberFieldEmbeddingRecord(
        kind="COMPLEX",
        embedding=original.embedding,
        isolating_rectangle=wider_evidence,
        half_plane="NEGATIVE_IMAGINARY",
    )

    assert alternate_record.embedding == original.embedding
    assert alternate_record.isolating_rectangle != original.isolating_rectangle
    assert "isolating_rectangle" not in original.embedding.model_dump()


def test_nonprimitive_presentations_and_malformed_elements_are_rejected() -> None:
    with pytest.raises(ValidationError, match="primitive"):
        _field("2", "0", "2")
    with pytest.raises(ValidationError, match="positive leading"):
        _field("-1", "0", "-1")
    with pytest.raises(ValidationError, match="exactly one coefficient"):
        SimpleNumberFieldElement(
            presentation=_field("1", "0", "1"),
            coefficients_ascending=(_rational(1),),
        )


def test_malformed_overlapping_wrong_root_and_wrong_sign_evidence_are_rejected() -> (
    None
):
    result = embeddings(_field("1", "0", "1"))
    negative, positive = result.records
    assert isinstance(negative, ComplexNumberFieldEmbeddingRecord)
    assert isinstance(positive, ComplexNumberFieldEmbeddingRecord)

    with pytest.raises(ValueError, match="selected indexed root"):
        require_rectangle_selects_root(
            negative.embedding.root,
            positive.isolating_rectangle,
        )

    overlapping = RationalComplexIsolatingRectangle(
        real_lower=_rational(-1),
        real_upper=_rational(1),
        imaginary_lower=_rational(-2),
        imaginary_upper=_rational(2),
    )
    with pytest.raises(ValueError, match="exactly one root"):
        require_rectangle_selects_root(negative.embedding.root, overlapping)

    wrong_sign = negative.model_dump(mode="json")
    wrong_sign["half_plane"] = "POSITIVE_IMAGINARY"
    with pytest.raises(ValidationError, match="half-plane"):
        ComplexNumberFieldEmbeddingRecord.model_validate(wrong_sign)

    boundary_root = RationalComplexIsolatingRectangle(
        real_lower=_rational(0),
        real_upper=_rational(Fraction(1, 2)),
        imaginary_lower=_rational(-1),
        imaginary_upper=_rational(Fraction(-1, 2)),
    )
    with pytest.raises(ValueError, match="selected indexed root"):
        require_rectangle_selects_root(negative.embedding.root, boundary_root)

    oversized = negative.model_dump(mode="json")
    oversized["isolating_rectangle"]["real_lower"]["num"] = "1" * 4_097
    with pytest.raises(ValidationError, match="4,096-digit bound"):
        ComplexNumberFieldEmbeddingRecord.model_validate(oversized)

    oversized_component = CanonicalRational(num="-1", den="9" * 4_097)
    with pytest.raises(ValidationError, match="4,096-digit bound"):
        RationalComplexIsolatingRectangle(
            real_lower=oversized_component,
            real_upper=_rational(1),
            imaginary_lower=_rational(-2),
            imaginary_upper=_rational(-1),
        )


def test_real_interval_is_bound_to_the_selected_real_root() -> None:
    result = embeddings(_field("1", "0", "-2"))
    negative, positive = result.records
    assert isinstance(negative, RealNumberFieldEmbeddingRecord)
    assert isinstance(positive, RealNumberFieldEmbeddingRecord)
    with pytest.raises(ValueError, match="indexed root"):
        require_real_interval_selects_root(
            negative.embedding,
            positive.isolating_interval,
        )

    oversized = negative.model_dump(mode="json")
    oversized["isolating_interval"]["lower"]["num"] = "1" * 4_097
    with pytest.raises(ValidationError, match="4,096-digit bound"):
        RealNumberFieldEmbeddingRecord.model_validate(oversized)

    oversized_component = CanonicalRational(num="-1", den="9" * 4_097)
    with pytest.raises(ValidationError, match="4,096-digit bound"):
        RealNumberFieldEmbeddingRecord(
            kind="REAL",
            embedding=negative.embedding,
            isolating_interval=RationalIsolatingInterval(
                lower=oversized_component,
                upper=_rational(0),
                interval_type="OPEN",
            ),
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        ("1", "0", "5", "0", "5"),
        ("1", "0", "0", "-1", "1"),
    ],
)
def test_profiles_are_deterministic_across_backend_cache_refinement(
    coefficients: tuple[str, ...],
) -> None:
    field = _field(*coefficients)
    first = compute_embeddings_worker_response(_worker_request(field)).model_dump_json()

    import sympy

    variable = sympy.Symbol("a")
    polynomial = sympy.Poly.from_list(
        [int(coefficient) for coefficient in coefficients],
        variable,
    )
    roots = polynomial.all_roots(radicals=False)
    intervals_before = tuple(root._get_interval() for root in roots)
    for root in roots:
        root.eval_rational(n=5)
    assert tuple(root._get_interval() for root in roots) != intervals_before

    assert (
        compute_embeddings_worker_response(_worker_request(field)).model_dump_json()
        == first
    )


def test_worker_executes_one_all_root_isolation_pass() -> None:
    profiler = cProfile.Profile()
    field = _field("1", "0", "5", "0", "5")

    profiler.runcall(compute_embeddings_worker_response, _worker_request(field))
    poly_intervals_calls = sum(
        entry.callcount
        for entry in profiler.getstats()
        if isinstance(entry.code, CodeType)
        and entry.code.co_filename.endswith("sympy/polys/polytools.py")
        and entry.code.co_name == "intervals"
    )
    complex_isolation_calls = sum(
        entry.callcount
        for entry in profiler.getstats()
        if isinstance(entry.code, CodeType)
        and entry.code.co_name == "dup_isolate_complex_roots_sqf"
    )

    assert poly_intervals_calls == 1
    assert complex_isolation_calls == 1


def test_profile_and_canonical_values_round_trip_without_backend_objects() -> None:
    profile = embeddings(_field("1", "0", "-1", "1"))
    encoded = encode_strict_json(profile.model_dump(mode="json"))
    profiler = cProfile.Profile()

    restored = profiler.runcall(
        lambda: NumberFieldEmbeddingProfile.model_validate_json(
            encoded,
            strict=True,
        )
    )

    assert restored == profile
    assert not any(
        isinstance(entry.code, CodeType)
        and "/sympy/" in entry.code.co_filename.replace("\\", "/")
        for entry in profiler.getstats()
    )
    assert "CRootOf" not in profile.model_dump_json()
    assert "sympy" not in profile.model_dump_json().lower()


def test_structural_complex_root_parsing_does_not_run_sympy() -> None:
    profiler = cProfile.Profile()
    value = profiler.runcall(
        lambda: ComplexAlgebraicValue(polynomial=("1", "0", "-1"), root_index=0)
    )

    assert value.root_index == 0
    assert not any(
        isinstance(entry.code, CodeType)
        and "/sympy/" in entry.code.co_filename.replace("\\", "/")
        for entry in profiler.getstats()
    )


def test_only_real_embedding_carrier_is_public_native_api() -> None:
    assert "ComplexNumberFieldEmbedding" not in number_fields.__all__
    assert "RealNumberFieldEmbedding" in number_fields.__all__
    assert "EmbeddedSimpleNumberFieldElement" not in number_fields.__all__


def test_profile_structural_validation_rejects_an_incomplete_result() -> None:
    profile = embeddings(_field("1", "0", "1"))
    incomplete = profile.model_dump(mode="json")
    incomplete["records"].pop()
    with pytest.raises(ValidationError, match="degree-many"):
        NumberFieldEmbeddingProfile.model_validate(incomplete)


def test_degree_coefficient_isolation_and_worker_bounds_are_preflighted() -> None:
    # The minimal polynomial of zeta_17 + zeta_17^-1 is irreducible of degree
    # eight and totally real, so it exercises the closed degree boundary
    # without making this unit test duplicate the complex-order stress case.
    degree_eight = _field("1", "1", "-7", "-5", "15", "6", "-10", "-1", "1")
    admission = _admit_number_field_embeddings(degree_eight)

    assert admission.predicted_worker_output_bytes > 0
    embeddings(degree_eight)

    degree_nine = _field("1", *("0",) * 8, "2")
    with pytest.raises(NumberFieldEmbeddingAdmissionError) as caught:
        embeddings(degree_nine)
    assert caught.value.reason == "degree_bound"
    with pytest.raises(ValidationError, match="256 digits"):
        _field("1", "0", "1" + "0" * 256)
    with pytest.raises(ValidationError, match="256 digits"):
        SimpleNumberFieldElement.model_validate(
            {
                "presentation": _field("1", "0").model_dump(mode="json"),
                "coefficients_ascending": [{"num": "1" * 257, "den": "1"}],
            }
        )

    # Eisenstein at 2: this is a valid 256-digit defining polynomial, but its
    # exact pair-ordering precision is rejected inside the bounded worker after
    # the static resultant-storage envelope has admitted the presolve.
    large_eisenstein = _field("1", *("0",) * 7, str(10**255 + 2))
    with pytest.raises(NumberFieldEmbeddingAdmissionError) as caught:
        embeddings(large_eisenstein)
    assert caught.value.reason == "pair_ordering_precision_bound"


def test_real_embedding_rejects_degree_above_its_runtime_carrier_bound() -> None:
    degree_nine = _field("1", *("0",) * 8, "-2")

    with pytest.raises(ValidationError, match="limited to degree 8"):
        RealNumberFieldEmbedding(
            kind="REAL",
            presentation=degree_nine,
            root={
                "polynomial": degree_nine.coefficients_descending,
                "real_root_index": 0,
            },
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        ("1", "0"),
        ("9" * 256, "1"),
        ("1", "0", "-2"),
        ("2", "0", "1"),
        ("1", "0", "1"),
        ("1", "0", "-1", "1"),
        ("1", "0", "5", "0", "5"),
        ("1", "1", "-7", "-5", "15", "6", "-10", "-1", "1"),
    ],
)
def test_kernel_isolators_and_result_satisfy_the_exact_contract(
    coefficients: tuple[str, ...],
) -> None:
    field = _field(*coefficients)
    admission = _admit_number_field_embeddings(field)
    result = embeddings(field)
    encoded = encode_strict_json(result.model_dump(mode="json"))
    worker_projection = NumberFieldEmbeddingWorkerComplete(
        kind="complete",
        real_intervals=tuple(
            record.isolating_interval
            for record in result.records
            if isinstance(record, RealNumberFieldEmbeddingRecord)
        ),
        negative_complex_rectangles=tuple(
            record.isolating_rectangle
            for record in result.records
            if isinstance(record, ComplexNumberFieldEmbeddingRecord)
            and record.half_plane == "NEGATIVE_IMAGINARY"
        ),
        defining_polynomial_discriminant=result.defining_polynomial_discriminant,
    )

    assert (
        len(worker_projection.model_dump_json().encode("utf-8"))
        <= admission.predicted_worker_output_bytes
    )
    assert (
        NumberFieldEmbeddingProfile.model_validate_json(encoded, strict=True) == result
    )
    for record in result.records:
        if isinstance(record, RealNumberFieldEmbeddingRecord):
            require_real_interval_selects_root(
                record.embedding,
                record.isolating_interval,
            )
        else:
            assert (
                require_rectangle_selects_root(
                    record.embedding.root,
                    record.isolating_rectangle,
                )
                == record.half_plane
            )


def test_catalog_operation_exposes_and_runs_the_advertised_gaussian_example() -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "number_field.embeddings.compute"
    )
    example = operation.examples[0]
    request = NumberFieldEmbeddingsRequest.model_validate(example.input)

    result = operation.run(request)

    assert isinstance(result, NumberFieldEmbeddingProfile)
    assert result.signature.complex_conjugate_pair_count == 1
    assert result.defining_polynomial_discriminant == "-4"


def test_reducible_presentation_is_recognized_inside_the_operation() -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "number_field.embeddings.compute"
    )
    request = NumberFieldEmbeddingsRequest.model_validate(
        {
            "field": {
                "domain": "QQ",
                "coefficients_descending": ["1", "0", "-1"],
            }
        }
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        operation.run(request)

    assert caught.value.errors()[0]["type"] == "number_field.embeddings.not_irreducible"


def test_embedding_worker_honors_an_already_expired_request_deadline() -> None:
    field = _field("1", "0", "1")

    with (
        request_execution(started_at=time.monotonic() - 121),
        pytest.raises(OperationExecutionTimeoutError, match="before"),
    ):
        embeddings(field)


def test_embedding_worker_honors_request_cancellation_before_launch() -> None:
    field = _field("1", "0", "1")
    cancellation = Event()
    cancellation.set()

    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="cancelled"),
    ):
        embeddings(field)


@pytest.mark.parametrize(
    "model",
    [
        RealNumberFieldEmbeddingRecord,
        ComplexNumberFieldEmbeddingRecord,
    ],
)
def test_embedding_discriminator_is_required_in_schema(model: type[object]) -> None:
    assert "kind" in model.model_json_schema()["required"]  # type: ignore[attr-defined]


def test_catalog_operation_projects_owner_local_admission_rejection() -> None:
    operation = next(
        tool for tool in TOOLS if tool.operation_id == "number_field.embeddings.compute"
    )
    field = _field("1", *("0",) * 7, str(10**255 + 2))
    request = NumberFieldEmbeddingsRequest(field=field)

    with pytest.raises(OperationDomainValidationError) as caught:
        operation.run(request)

    assert caught.value.errors()[0]["loc"] == ("field",)
    assert str(caught.value.errors()[0]["type"]).startswith("number_field.embeddings.")


def test_embedding_union_keeps_the_selected_parent_context() -> None:
    profile = embeddings(_field("1", "0", "1"))
    record = profile.records[1]
    assert isinstance(record, ComplexNumberFieldEmbeddingRecord)

    assert record.embedding.presentation == profile.field
    assert record.embedding.root.polynomial == profile.field.coefficients_descending
