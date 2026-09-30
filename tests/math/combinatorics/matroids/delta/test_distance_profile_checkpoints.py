"""Distance-profile decoding shares the request's cancellation and deadline."""

from __future__ import annotations

import math
import time
from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
    request_execution,
)
from jacobian.math.combinatorics.matroids.delta import values
from jacobian.math.combinatorics.matroids.delta.operations import distance_profile
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidDistanceProfile,
    FiniteDeltaMatroid,
)


def _maximal_profile() -> DeltaMatroidDistanceProfile:
    # All subsets of the first six coordinates, with six loops. Distance is
    # exactly the number of loop coordinates, with a unique nearest subset.
    return distance_profile(
        FiniteDeltaMatroid(
            ground=tuple(f"e{i}" for i in range(12)),
            feasible=tuple(
                sorted(
                    tuple(i for i in range(6) if mask >> i & 1) for mask in range(64)
                )
            ),
        )
    )


def _decode(
    profile: DeltaMatroidDistanceProfile, representation: str
) -> DeltaMatroidDistanceProfile:
    if representation == "json":
        return DeltaMatroidDistanceProfile.model_validate_json(
            profile.model_dump_json()
        )
    if representation == "mapping":
        return DeltaMatroidDistanceProfile.model_validate(profile.model_dump())
    return DeltaMatroidDistanceProfile.model_validate(profile)


@pytest.mark.parametrize("representation", ("native", "mapping", "json"))
@pytest.mark.parametrize("interruption", ("cancellation", "deadline"))
def test_profile_replay_observes_interruption_after_it_starts(
    monkeypatch: pytest.MonkeyPatch, representation: str, interruption: str
) -> None:
    profile = _maximal_profile()
    signal = Event()
    clock = [0.0]
    replay_checkpoints: list[str] = []
    checkpoint = request_checkpoint

    def interrupt_after_first_checkpoint(stage: str) -> None:
        checkpoint(stage)
        if stage == "during delta-matroid distance-profile relation replay":
            replay_checkpoints.append(stage)
            signal.set()
            clock[0] = 2.0

    monkeypatch.setattr(values, "request_checkpoint", interrupt_after_first_checkpoint)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    error_type = (
        OperationExecutionCancelledError
        if interruption == "cancellation"
        else OperationExecutionTimeoutError
    )
    with (
        request_execution(
            0.0,
            cancellation_signal=signal if interruption == "cancellation" else None,
            outer_deadline=1.0 if interruption == "deadline" else None,
        ),
        pytest.raises(error_type),
    ):
        _decode(profile, representation)
    assert len(replay_checkpoints) == 1


@pytest.mark.parametrize("representation", ("native", "mapping", "json"))
def test_maximal_profile_still_decodes_with_exact_distance_rows(
    monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    profile = _maximal_profile()
    replay_checkpoints: list[str] = []
    checkpoint = request_checkpoint

    def observe_checkpoint(stage: str) -> None:
        checkpoint(stage)
        if stage == "during delta-matroid distance-profile relation replay":
            replay_checkpoints.append(stage)

    monkeypatch.setattr(values, "request_checkpoint", observe_checkpoint)
    decoded = _decode(profile, representation)
    assert decoded == profile
    assert decoded.distance_by_mask == tuple(
        (mask >> 6).bit_count() for mask in range(4096)
    )
    assert decoded.nearest_feasible_count_by_mask == (1,) * 4096
    assert (
        decoded.distance_histogram
        == tuple(64 * math.comb(6, i) for i in range(7)) + (0,) * 6
    )
    assert len(replay_checkpoints) == 64
