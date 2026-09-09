"""Prime-field linear-code canonicalization tests."""

from jacobian.math.combinatorics.codes.linear._canonicalization import (
    LinearCodeCanonicalizationRequest,
    canonicalize_linear_code,
)
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
