"""Tests for linear code structural operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.code_linear._models import (
    CodeEqualRequest,
    CodewordCheckRequest,
    GeneratorMatrixRequest,
    MacWilliamsRequest,
    ParityCheckRequest,
    PunctureRequest,
    ShortenRequest,
    SyndromeRequest,
)
from jacobian.math.code_linear._operations import (
    compute_code_equal,
    compute_codeword_check,
    compute_dual_code,
    compute_from_generator,
    compute_macwilliams_transform,
    compute_parity_check,
    compute_puncture,
    compute_shorten,
    compute_syndrome,
)
from jacobian.math.code_linear._tools import TOOLS
from jacobian.math.code_linear.values import PrimeFieldLinearEncoder


def _encoder(
    generator: tuple[tuple[int, ...], ...],
    *,
    field_order: int = 2,
    coordinate_axis: tuple[str, ...] | None = None,
) -> PrimeFieldLinearEncoder:
    dimension = len(generator)
    length = len(generator[0]) if generator else 0
    return PrimeFieldLinearEncoder(
        field_order=field_order,
        message_axis=tuple(f"m{index}" for index in range(dimension)),
        coordinate_axis=(
            tuple(f"x{index}" for index in range(length))
            if coordinate_axis is None
            else coordinate_axis
        ),
        generator_matrix=generator,
    )


def test_catalog_contains_only_audited_operations() -> None:
    expected = {
        "code.linear.received_word_profile.compute",
        "code.linear.codeword.check",
        "code.linear.dual.compute",
        "code.linear.equal.decide",
        "code.linear.from_generator.compute",
        "code.linear.macwilliams_transform.compute",
        "code.linear.parity_check.compute",
        "code.linear.puncture.compute",
        "code.linear.shorten.compute",
        "code.linear.syndrome.compute",
    }
    assert {tool.operation_id for tool in TOOLS} == expected


def test_from_generator_canonicalizes_dependent_rows() -> None:
    request = GeneratorMatrixRequest(
        field_order=2,
        generator_matrix=((1, 1), (1, 1)),
        coordinate_axis=("left", "right"),
    )
    result = compute_from_generator(request)
    assert result.dimension == 1
    assert result.length == 2
    assert result.cardinality == 2
    assert result.encoder.generator_matrix == ((1, 1),)
    assert result.encoder.message_axis == ("m0",)
    assert result.encoder.coordinate_axis == ("left", "right")


def test_dual_of_repetition_is_parity_check() -> None:
    request = GeneratorMatrixRequest(
        field_order=2,
        generator_matrix=((1, 1),),
        coordinate_axis=("left", "right"),
    )
    result = compute_dual_code(request)
    assert result.dimension == 1
    assert result.dual_dimension == 1
    assert result.length == 2
    assert result.encoder.generator_matrix == result.parity_check.rows
    assert result.encoder.message_axis == ("m0",)
    assert result.encoder.coordinate_axis == ("left", "right")


def test_parity_check_matches_dual() -> None:
    request = ParityCheckRequest(field_order=2, generator_matrix=((1, 1),))
    result = compute_parity_check(request)
    assert result.dimension == 1
    assert result.rank_h == 1
    assert result.length == 2
    assert len(result.parity_check.rows) == 1


def test_codeword_check_member() -> None:
    request = CodewordCheckRequest(
        field_order=2, generator_matrix=((1, 1),), word=(1, 1)
    )
    result = compute_codeword_check(request)
    assert result.is_member is True
    assert result.hamming_weight == 2


def test_codeword_check_nonmember() -> None:
    request = CodewordCheckRequest(
        field_order=2, generator_matrix=((1, 1),), word=(1, 0)
    )
    result = compute_codeword_check(request)
    assert result.is_member is False
    assert result.hamming_weight == 1


def test_syndrome_zero_for_codeword() -> None:
    request = SyndromeRequest(
        parity_check={"field_order": 2, "column_count": 2, "rows": ((1, 1),)},
        word=(1, 1),
    )
    result = compute_syndrome(request)
    assert result.syndrome == (0,)
    assert result.is_member is True


def test_syndrome_nonzero_for_noncodeword() -> None:
    request = SyndromeRequest(
        parity_check={"field_order": 2, "column_count": 2, "rows": ((1, 1),)},
        word=(1, 0),
    )
    result = compute_syndrome(request)
    assert result.syndrome == (1,)
    assert result.is_member is False


def test_full_space_dual_composes_into_empty_syndrome() -> None:
    dual = compute_dual_code(
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((1, 0), (0, 1)),
            coordinate_axis=("left", "right"),
        )
    )
    assert dual.parity_check.rows == ()
    result = compute_syndrome(
        SyndromeRequest(parity_check=dual.parity_check, word=(1, 1))
    )
    assert result.syndrome == ()
    assert result.is_member is True


def test_rank_one_length_32_code_retains_all_dual_rows() -> None:
    result = compute_dual_code(
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=(tuple([1] + [0] * 31),),
            coordinate_axis=tuple(f"x{index}" for index in range(32)),
        )
    )
    assert len(result.parity_check.rows) == 31
    assert result.parity_check.column_count == 32
    assert len(result.encoder.message_axis) == 31


def test_code_equal_same_matrices() -> None:
    request = CodeEqualRequest(
        field_order=2,
        generator_matrix_a=((1, 1),),
        generator_matrix_b=((1, 1),),
    )
    result = compute_code_equal(request)
    assert result.equal is True
    assert result.witness_word is None


def test_code_equal_different_codes() -> None:
    request = CodeEqualRequest(
        field_order=2,
        generator_matrix_a=((1, 0),),
        generator_matrix_b=((0, 1),),
    )
    result = compute_code_equal(request)
    assert result.equal is False
    assert result.witness_word is not None


def test_macwilliams_self_dual_repetition_code() -> None:
    request = MacWilliamsRequest(
        field_order=2, code_cardinality=2, length=2, weights=(1, 0, 1)
    )
    result = compute_macwilliams_transform(request)
    assert result.dual_weights == (1, 0, 1)


def test_puncture_reduces_length() -> None:
    request = PunctureRequest(
        encoder=_encoder(((1, 1),), coordinate_axis=("left", "right")),
        coordinate=0,
    )
    result = compute_puncture(request)
    assert result.length == 1
    assert result.dimension == 1
    assert result.encoder.coordinate_axis == ("right",)
    assert result.encoder.message_axis == ("m0",)


def test_shorten_reduces_dimension_and_length() -> None:
    # Shortening the length-3 binary repetition code at coordinate 0
    # keeps codewords with c[0]=0 (only the zero word), so dimension drops to 0.
    request = ShortenRequest(
        encoder=_encoder(((1, 1, 1),), coordinate_axis=("left", "middle", "right")),
        coordinate=0,
    )
    result = compute_shorten(request)
    assert result.length == 2
    assert result.dimension == 0
    assert result.encoder.coordinate_axis == ("middle", "right")
    assert result.encoder.message_axis == ()


def test_shorten_2d_code() -> None:
    # Code generated by [[1,1,0],[1,0,1]] over F_2.
    # Codewords: 000, 110, 101, 011.
    # Shortening at coordinate 0: keep codewords with c[0]=0: {000, 011}.
    # Delete coordinate 0: {00, 11} -> dimension 1, length 2.
    request = ShortenRequest(
        encoder=_encoder(
            ((1, 1, 0), (1, 0, 1)), coordinate_axis=("left", "middle", "right")
        ),
        coordinate=0,
    )
    result = compute_shorten(request)
    assert result.length == 2
    assert result.dimension == 1


def test_requests_accept_serialized_producer_encoders_unchanged() -> None:
    from_generator = compute_from_generator(
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((1, 0, 1), (0, 1, 1)),
            coordinate_axis=("left", "middle", "right"),
        )
    )
    dual = compute_dual_code(
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((1, 1, 1),),
            coordinate_axis=("left", "middle", "right"),
        )
    )

    puncture_request = PunctureRequest.model_validate(
        {"encoder": from_generator.model_dump(mode="json")["encoder"], "coordinate": 2}
    )
    assert puncture_request.encoder == from_generator.encoder

    punctured = compute_puncture(puncture_request)
    chained_request = ShortenRequest.model_validate(
        {"encoder": punctured.model_dump(mode="json")["encoder"], "coordinate": 0}
    )
    assert chained_request.encoder == punctured.encoder

    dual_request = ShortenRequest.model_validate(
        {"encoder": dual.model_dump(mode="json")["encoder"], "coordinate": 0}
    )
    assert dual_request.encoder == dual.encoder

    chained_result = compute_shorten(chained_request)
    assert chained_result.dimension == 1
    assert chained_result.encoder.coordinate_axis == ("middle",)
    assert chained_result.encoder.generator_matrix == ((1,),)

    dual_result = compute_shorten(dual_request)
    assert dual_result.dimension == 1
    assert dual_result.encoder.generator_matrix == ((1, 1),)


def test_puncture_and_shorten_requests_reject_unselectable_coordinates() -> None:
    empty_encoder = _encoder(())
    with pytest.raises(ValidationError, match="at least one coordinate"):
        PunctureRequest(encoder=empty_encoder, coordinate=0)
    with pytest.raises(ValidationError, match="at least one coordinate"):
        ShortenRequest(encoder=empty_encoder, coordinate=0)

    full_rank = _encoder(((1, 1),))
    with pytest.raises(ValidationError, match="greater than or equal"):
        PunctureRequest(encoder=full_rank, coordinate=-1)
    with pytest.raises(ValidationError, match="out of range"):
        PunctureRequest(encoder=full_rank, coordinate=2)
    with pytest.raises(ValidationError, match="out of range"):
        ShortenRequest(encoder=full_rank, coordinate=5)


def test_requests_reject_rank_deficient_or_ambiguous_encoders() -> None:
    ambiguous = {
        "field_order": 2,
        "message_axis": ["m0"],
        "coordinate_axis": ["x0", "x0"],
        "generator_matrix": [[1, 1]],
    }
    with pytest.raises(ValidationError, match="unique"):
        PunctureRequest(encoder=ambiguous, coordinate=0)

    dependent = {
        "field_order": 2,
        "message_axis": ["m0", "m1"],
        "coordinate_axis": ["x0", "x1"],
        "generator_matrix": [[1, 1], [1, 1]],
    }
    with pytest.raises(ValidationError, match="full row rank"):
        PunctureRequest(encoder=dependent, coordinate=0)
    with pytest.raises(ValidationError, match="full row rank"):
        ShortenRequest(encoder=dependent, coordinate=1)


def test_macwilliams_ternary() -> None:
    # Ternary repetition code: C = {00, 11, 22}, q=3, n=2
    # Weight distribution: A_0=1, A_1=0, A_2=2
    # Dual weight distribution should also be (1, 0, 2) since it's self-dual.
    request = MacWilliamsRequest(
        field_order=3,
        code_cardinality=3,
        length=2,
        weights=(1, 0, 2),
    )
    result = compute_macwilliams_transform(request)
    assert result.dual_weights == (1, 0, 2)


def test_request_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError, match="prime"):
        GeneratorMatrixRequest(
            field_order=4,
            generator_matrix=((1,),),
            coordinate_axis=("x",),
        )


def test_request_rejects_bad_entry() -> None:
    with pytest.raises(ValidationError, match="residues"):
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((2,),),
            coordinate_axis=("x",),
        )


def test_code_producer_requests_reject_ambiguous_coordinate_axes() -> None:
    with pytest.raises(ValidationError, match="match the generator-matrix columns"):
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((1, 1),),
            coordinate_axis=("x",),
        )


def test_syndrome_request_rejects_bad_word_length() -> None:
    with pytest.raises(ValidationError, match="length"):
        SyndromeRequest(
            parity_check={"field_order": 2, "column_count": 2, "rows": ((1, 1),)},
            word=(1,),
        )


def test_codeword_check_request_rejects_bad_word_length() -> None:
    with pytest.raises(ValidationError, match="length"):
        CodewordCheckRequest(field_order=2, generator_matrix=((1, 1),), word=(1,))
