"""Process-boundary regressions for affine-torus FLINT execution."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from threading import Event, Timer
from time import monotonic

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_execution,
)
from jacobian.math.geometry.affine_tori import (
    RationalAffineTorusMap,
    _flint_process,
    affine_torus_fixed_locus,
)
from jacobian.math.geometry.affine_tori import _bounds as affine_bounds
from jacobian.math.geometry.affine_tori._bounds import build_affine_torus_plan


def _source() -> RationalAffineTorusMap:
    translation = Fraction(0)
    return RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[3]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [
                    {
                        "num": translation.numerator,
                        "den": translation.denominator,
                    }
                ],
            },
        }
    )


def _hanging_worker(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "worker-started"
    worker = tmp_path / "hanging_worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        "from time import sleep\n"
        f"Path({str(marker)!r}).write_text('started', encoding='utf-8')\n"
        "sleep(60)\n",
        encoding="utf-8",
    )
    return worker, marker


def test_cancellation_kills_the_affine_torus_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, marker = _hanging_worker(tmp_path)
    monkeypatch.setattr(_flint_process, "_AFFINE_TORUS_WORKER", worker)
    cancellation = Event()
    timer = Timer(1.0, cancellation.set)
    started = monotonic()
    timer.start()
    try:
        with (
            request_execution(started),
            request_cancellation(cancellation),
            pytest.raises(OperationExecutionCancelledError),
        ):
            affine_torus_fixed_locus(_source())
    finally:
        timer.cancel()
        timer.join()

    assert marker.is_file()
    assert monotonic() - started < 5.0


def test_deadline_kills_the_affine_torus_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, marker = _hanging_worker(tmp_path)
    monkeypatch.setattr(_flint_process, "_AFFINE_TORUS_WORKER", worker)
    monkeypatch.setattr(affine_bounds, "AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS", 1.5)
    monkeypatch.setattr(_flint_process, "_PARENT_FINALIZATION_SECONDS", 0.1)
    started = monotonic()

    with (
        request_execution(started),
        pytest.raises(OperationExecutionTimeoutError, match="worker deadline expired"),
    ):
        affine_torus_fixed_locus(_source())

    assert marker.is_file()
    assert monotonic() - started < 5.0


def _valid_worker_projection() -> tuple[RationalAffineTorusMap, dict[str, object]]:
    source = _source()
    # The one-dimensional A=3 example has fixed points 0 and 1/2, with no
    # positive-dimensional identity component.
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "NONEMPTY",
        "rank": 1,
        "nullity": 0,
        "base_point": [{"num": "0", "den": "1"}],
        "identity_embedding": [[]],
        "component_generators": [[{"num": "1", "den": "2"}]],
        "relation_matrix": [["2"]],
        "generator_orders": ["2"],
        "invariant_factors": ["2"],
        "component_count": "2",
    }
    return source, payload


def test_worker_projection_validation_accepts_a_mathematically_valid_response() -> None:
    source, payload = _valid_worker_projection()
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    decoded = _flint_process._decode_worker_projection(
        payload,
        request_digest="test-digest",
        source=source,
        plan=plan,
    )

    assert decoded.component_count == 2


def test_worker_projection_validation_accepts_identity_component_relations() -> None:
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 2},
            "linear_part": {
                "row_count": 2,
                "column_count": 2,
                "entries": [[1, 0], [-4, -4]],
            },
            "translation": {
                "torus": {"dimension": 2},
                "coordinates": [{"num": 0, "den": 1}] * 2,
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "NONEMPTY",
        "rank": 1,
        "nullity": 1,
        "base_point": [
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
        ],
        "identity_embedding": [["5"], ["-4"]],
        "component_generators": [[{"num": "3", "den": "4"}, {"num": "0", "den": "1"}]],
        "relation_matrix": [["1"]],
        "generator_orders": ["1"],
        "invariant_factors": [],
        "component_count": "1",
    }
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    decoded = _flint_process._decode_worker_projection(
        payload,
        request_digest="test-digest",
        source=source,
        plan=plan,
    )

    assert decoded.identity_embedding == ((5,), (-4,))


def test_worker_projection_validation_accepts_a_source_bound_empty_obstruction() -> (
    None
):
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[1]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [{"num": 1, "den": 2}],
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "EMPTY",
        "character": ["1"],
        "pairing": {"num": "1", "den": "2"},
    }
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    decoded = _flint_process._decode_worker_projection(
        payload,
        request_digest="test-digest",
        source=source,
        plan=plan,
    )

    assert decoded.character == (1,)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("character", ["0"]),
        ("character", ["2"]),
        ("pairing", {"num": "0", "den": "1"}),
    ),
)
def test_worker_projection_validation_rejects_one_mutated_empty_obstruction_claim(
    field: str,
    replacement: object,
) -> None:
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[1]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [{"num": 1, "den": 2}],
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "EMPTY",
        "character": ["1"],
        "pairing": {"num": "1", "den": "2"},
    }
    payload[field] = replacement
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )


def test_worker_projection_validation_rejects_an_empty_character_outside_the_left_kernel() -> (
    None
):
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[3]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [{"num": 1, "den": 3}],
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "EMPTY",
        "character": ["1"],
        "pairing": {"num": "1", "den": "3"},
    }
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )


def test_worker_projection_validation_rejects_a_nonsaturated_identity_kernel() -> None:
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[1]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [{"num": 0, "den": 1}],
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "NONEMPTY",
        "rank": 0,
        "nullity": 1,
        "base_point": [{"num": "0", "den": "1"}],
        "identity_embedding": [["2"]],
        "component_generators": [],
        "relation_matrix": [],
        "generator_orders": [],
        "invariant_factors": [],
        "component_count": "1",
    }
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )


def test_worker_projection_validation_rejects_a_trivial_generator_with_order_two() -> (
    None
):
    source, payload = _valid_worker_projection()
    payload = deepcopy(payload)
    payload["component_generators"] = [[{"num": "0", "den": "1"}]]
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )


def test_worker_projection_validation_rejects_duplicate_generators_in_z2_squared() -> (
    None
):
    source = RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 3},
            "linear_part": {
                "row_count": 3,
                "column_count": 3,
                "entries": [[3, 0, 0], [0, 3, 0], [0, 0, 1]],
            },
            "translation": {
                "torus": {"dimension": 3},
                "coordinates": [{"num": 0, "den": 1}] * 3,
            },
        }
    )
    payload: dict[str, object] = {
        "protocol_version": 1,
        "request_digest": "test-digest",
        "status": "NONEMPTY",
        "rank": 2,
        "nullity": 1,
        "base_point": [
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
            {"num": "0", "den": "1"},
        ],
        "identity_embedding": [["0"], ["0"], ["1"]],
        "component_generators": [
            [
                {"num": "1", "den": "2"},
                {"num": "0", "den": "1"},
                {"num": "0", "den": "1"},
            ],
            [
                {"num": "1", "den": "2"},
                {"num": "0", "den": "1"},
                {"num": "0", "den": "1"},
            ],
        ],
        "relation_matrix": [["2", "0"], ["0", "2"]],
        "generator_orders": ["2", "2"],
        "invariant_factors": ["2", "2"],
        "component_count": "4",
    }
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("component_count", "1"),
        ("invariant_factors", []),
        ("generator_orders", ["1"]),
        ("base_point", [{"num": "1", "den": "3"}]),
        ("identity_embedding", [["1"]]),
        ("component_generators", [[{"num": "1", "den": "3"}]]),
        ("relation_matrix", [["1"]]),
    ),
)
def test_worker_projection_validation_rejects_one_mutated_mathematical_claim(
    field: str,
    replacement: object,
) -> None:
    source, payload = _valid_worker_projection()
    payload = deepcopy(payload)
    payload[field] = replacement
    plan = build_affine_torus_plan(source, deadline=monotonic() + 120)

    with pytest.raises(ValueError):
        _flint_process._decode_worker_projection(
            payload,
            request_digest="test-digest",
            source=source,
            plan=plan,
        )
