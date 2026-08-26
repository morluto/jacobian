"""Exact embedding-profile tests for real quadratic elements."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicMultiplicationRequest,
)
from jacobian.math.algebraic_number_arithmetic._operations import (
    compute_algebraic_multiply,
)
from jacobian.math.arithmetic._real_quadratic import (
    REAL_QUADRATIC_OPERATIONS,
    RealQuadraticEmbeddingsRequest,
)
from jacobian.math.real_quadratic import (
    _MAX_EMBEDDING_PROFILE_RESULT_DIGITS,
    RealQuadraticEmbeddingProfile,
    RealQuadraticValue,
    real_quadratic_embeddings,
)


def _r(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _element(
    rational_part: int,
    radical_coefficient: int,
    radicand: int,
) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=_r(rational_part),
        radical_coefficient=_r(radical_coefficient),
        radicand=radicand,
    )


def _profile() -> RealQuadraticEmbeddingProfile:
    return real_quadratic_embeddings(_element(3, 2, 2))


def test_complete_embedding_profile_is_exact_and_ordered() -> None:
    profile = _profile()

    assert profile.real_embedding_count == 2
    assert profile.complex_conjugate_pair_count == 0
    assert tuple(image.embedding for image in profile.images) == (
        "POSITIVE_ROOT",
        "NEGATIVE_ROOT",
    )
    assert profile.images[0].value == _element(3, 2, 2)
    assert profile.images[1].value == _element(3, -2, 2)
    assert profile.trace.as_fraction() == 6
    assert profile.norm.as_fraction() == 1


def test_zero_has_two_labeled_embeddings_and_zero_invariants() -> None:
    profile = real_quadratic_embeddings(_element(0, 0, 5))

    assert tuple(image.embedding for image in profile.images) == (
        "POSITIVE_ROOT",
        "NEGATIVE_ROOT",
    )
    assert profile.images[0].value == profile.images[1].value == _element(0, 0, 5)
    assert profile.trace.as_fraction() == 0
    assert profile.norm.as_fraction() == 0


def test_profile_replay_rejects_forged_source_images_and_invariants() -> None:
    profile = _profile()
    payload = profile.model_dump(mode="json")

    with pytest.raises(ValidationError) as exc_info:
        RealQuadraticEmbeddingProfile.model_validate(
            {**payload, "images": list(reversed(payload["images"]))}
        )
    assert (
        exc_info.value.errors()[0]["type"] == "real_quadratic.embedding_images_mismatch"
    )
    with pytest.raises(ValidationError) as exc_info:
        RealQuadraticEmbeddingProfile.model_validate(
            {**payload, "norm": {"num": "2", "den": "1"}}
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "real_quadratic.embedding_invariants_mismatch"
    )
    with pytest.raises(ValidationError) as exc_info:
        RealQuadraticEmbeddingProfile.model_validate(
            {
                **payload,
                "source": {
                    **payload["source"],
                    "radicand": 3,
                },
            }
        )
    assert (
        exc_info.value.errors()[0]["type"] == "real_quadratic.embedding_images_mismatch"
    )


def test_embedding_images_compose_with_existing_field_multiplication() -> None:
    profile = _profile()
    product = compute_algebraic_multiply(
        AlgebraicMultiplicationRequest(
            left=profile.images[0].value,
            right=profile.images[1].value,
        )
    )

    assert product.rational_part.as_fraction() == profile.norm.as_fraction()
    assert product.radical_coefficient.as_fraction() == 0


def test_result_bound_is_proved_from_the_accepted_input_envelope() -> None:
    largest_component = 10**256 - 1
    profile = real_quadratic_embeddings(_element(largest_component, 0, 2))

    assert profile.trace.as_fraction() == 2 * largest_component
    assert profile.norm.as_fraction() == largest_component * largest_component
    with pytest.raises(ValidationError) as exc_info:
        _element(10**256, 0, 2)
    assert (
        exc_info.value.errors()[0]["type"] == "real_quadratic.rational_bound_exceeded"
    )


def test_embedding_declaration_adapter_serves_the_native_profile() -> None:
    tools = {tool.operation_id: tool for tool in REAL_QUADRATIC_OPERATIONS}
    tool = tools["arithmetic.real_quadratic.embeddings.compute"]
    profile = tool.run(RealQuadraticEmbeddingsRequest(element=_element(1, 1, 2)))

    assert profile == real_quadratic_embeddings(_element(1, 1, 2))
    assert profile.trace.as_fraction() == 2
    assert profile.norm.as_fraction() == -1


def test_embedding_declaration_is_native_only_with_a_supported_symbol() -> None:
    import importlib

    from jacobian.catalog.admission import AdmissionDecision
    from jacobian.math.arithmetic._admission import REGISTRATION

    record = next(
        admission
        for admission in REGISTRATION.admissions
        if admission.operation_id == "arithmetic.real_quadratic.embeddings.compute"
    )

    assert record.decision is AdmissionDecision.NATIVE_ONLY
    module_name, _, symbol_name = record.native_symbol.rpartition(".")
    module = importlib.import_module(module_name)
    assert symbol_name in module.__all__
    assert callable(getattr(module, symbol_name))


def test_embedding_profile_is_not_served_by_the_public_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {tool.operation_id for tool in BUILTIN_TOOLS}

    assert "arithmetic.real_quadratic.embeddings.compute" not in ids
    assert "arithmetic.real_quadratic.order.compute" in ids


def test_embedding_candidate_keeps_a_schema_visible_contract() -> None:
    tools = {tool.operation_id: tool for tool in REAL_QUADRATIC_OPERATIONS}
    tool = tools["arithmetic.real_quadratic.embeddings.compute"]

    assert f"{_MAX_EMBEDDING_PROFILE_RESULT_DIGITS:,}-digit" in tool.description
    schema = tool.request_type.model_json_schema()
    assert "square-free" in schema["properties"]["element"]["description"]
    assert tool.examples[0].input["element"]["radicand"] == 2


def test_fractional_trace_and_norm_are_exact() -> None:
    element = RealQuadraticValue(
        rational_part=_r(1, 2),
        radical_coefficient=_r(1, 3),
        radicand=3,
    )
    profile = real_quadratic_embeddings(element)

    assert profile.trace.as_fraction() == 1
    assert profile.norm.as_fraction() == Fraction(-1, 12)


def test_native_embedding_profile_api_is_explicit() -> None:
    from jacobian.math import real_quadratic

    assert "real_quadratic_embeddings" in real_quadratic.__all__
    assert real_quadratic.real_quadratic_embeddings is real_quadratic_embeddings
    assert all(
        not name.endswith(("Request", "Input")) for name in real_quadratic.__all__
    )
