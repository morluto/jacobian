"""Prime-field linear-code canonicalization tests."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.codes.linear import _canonicalization
from jacobian.math.combinatorics.codes.linear._canonicalization import (
    LinearCodeCanonicalizationRequest,
    LinearCodeCanonicalizationResult,
    canonicalize_linear_code,
)
from jacobian.math.combinatorics.codes.linear._models import GeneratorMatrixRequest
from jacobian.math.combinatorics.codes.linear._tools import compute_from_generator
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder
from jacobian.math.groups._models import PermutationGroup


def _encoder() -> PrimeFieldLinearEncoder:
    return PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=("x", "y", "z"),
        generator_matrix=((1, 0, 1),),
    )


def test_full_symmetric_action_returns_least_rref_and_orbit_stabilizer() -> None:
    result = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=_encoder())
    )
    assert result.canonical_encoder.generator_matrix == ((0, 1, 1),)
    assert result.orbit_size == 3
    assert result.stabilizer_size == 2
    assert result.transported_axis == tuple(
        _encoder().coordinate_axis[index] for index in result.transporter
    )


def test_supplied_cyclic_action_is_respected() -> None:
    result = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(
            encoder=_encoder(),
            action="SUPPLIED_GROUP",
            permutation_group=PermutationGroup(degree=3, generators=((1, 2, 0),)),
        )
    )
    assert result.orbit_size == 3
    assert result.stabilizer_size == 1


def test_transporter_replays_through_the_public_generator_operation() -> None:
    source = _encoder()
    result = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    permuted = tuple(
        tuple(row[index] for index in result.transporter)
        for row in source.generator_matrix
    )
    replayed = compute_from_generator(
        GeneratorMatrixRequest(
            field_order=source.field_order,
            generator_matrix=permuted,
            coordinate_axis=result.transported_axis,
        )
    )
    assert replayed.encoder == result.canonical_encoder


def test_zero_code_and_full_space_keep_degenerate_rref_shapes() -> None:
    zero = PrimeFieldLinearEncoder(
        field_order=3,
        message_axis=(),
        coordinate_axis=("x0", "x1"),
        generator_matrix=(),
    )
    zero_result = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=zero)
    )
    assert zero_result.canonical_encoder.generator_matrix == ()
    assert zero_result.orbit_size == 1
    assert zero_result.stabilizer_size == 2

    full = PrimeFieldLinearEncoder(
        field_order=3,
        message_axis=("m0", "m1"),
        coordinate_axis=("x0", "x1"),
        generator_matrix=((1, 0), (0, 1)),
    )
    full_result = canonicalize_linear_code(
        LinearCodeCanonicalizationRequest(encoder=full)
    )
    assert full_result.canonical_encoder.generator_matrix == ((1, 0), (0, 1))
    assert full_result.orbit_size == 1
    assert full_result.stabilizer_size == 2


def test_structural_result_validation_does_not_accept_a_bad_transporter() -> None:
    source = _encoder()
    result = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    payload = result.model_dump()
    payload["transporter"] = (0, 0, 1)
    with pytest.raises(ValidationError):
        LinearCodeCanonicalizationResult.model_validate(payload)


def test_result_round_trip_keeps_transporter_composable() -> None:
    source = _encoder()
    result = canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))
    decoded = LinearCodeCanonicalizationResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    permuted = tuple(
        tuple(row[index] for index in decoded.transporter)
        for row in source.generator_matrix
    )
    replayed = compute_from_generator(
        GeneratorMatrixRequest(
            field_order=source.field_order,
            generator_matrix=permuted,
            coordinate_axis=decoded.transported_axis,
        )
    )
    assert replayed.encoder == decoded.canonical_encoder


def test_full_symmetric_action_is_admitted_before_orbit_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=tuple(f"x{index}" for index in range(10)),
        generator_matrix=((1, 0, 0, 0, 0, 0, 0, 0, 0, 0),),
    )

    def orbit_must_not_be_materialized(_width: int) -> object:
        raise AssertionError("permutation orbit was materialized before admission")

    monkeypatch.setattr(
        _canonicalization, "permutations", orbit_must_not_be_materialized
    )
    with pytest.raises(OperationResourceAdmissionError):
        canonicalize_linear_code(LinearCodeCanonicalizationRequest(encoder=source))


def test_supplied_action_is_admitted_before_orbit_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=("m",),
        coordinate_axis=tuple(f"x{index}" for index in range(64)),
        generator_matrix=((1,) + (0,) * 63,),
    )

    def cycle(start: int, length: int) -> tuple[int, ...]:
        permutation = list(range(64))
        for offset in range(length):
            permutation[start + offset] = start + (offset + 1) % length
        return tuple(permutation)

    generators = (*tuple(cycle(5 * block, 5) for block in range(6)), cycle(30, 2))
    group = PermutationGroup(degree=64, generators=generators)

    def orbit_must_not_be_materialized(_element: object, _width: int) -> object:
        raise AssertionError(
            "supplied permutation orbit was materialized before admission"
        )

    monkeypatch.setattr(
        _canonicalization, "_full_permutation_form", orbit_must_not_be_materialized
    )
    with pytest.raises(OperationResourceAdmissionError):
        canonicalize_linear_code(
            LinearCodeCanonicalizationRequest(
                encoder=source,
                action="SUPPLIED_GROUP",
                permutation_group=group,
            )
        )
