import copy
import json
import shutil
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

TASK = "closed-set-distance-strengthening-audit"


def _oracle() -> dict[str, object]:
    return json.loads(
        (
            Path("benchmarks/datasets/public-reproductions-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(tmp_path: Path, submission: dict[str, object]):
    task = Path("benchmarks/datasets/public-reproductions-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps({"result": submission["result"]}))
    return task, app, logs


def _verify(tmp_path: Path, submission: dict[str, object]):
    return _run_verifier(*_prepare(tmp_path, submission))


def _q(numerator: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _pairs(start: int) -> list[dict[str, object]]:
    return [
        {
            "index": n,
            "a": [_q(n), _q(0)],
            "b": [_q(n), _q(1, n)],
            "distance": _q(1, n),
        }
        for n in range(start, start + 8)
    ]


def test_oracle_and_alternative_family_are_accepted(tmp_path: Path) -> None:
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"]["start_index"] = 7
    alternative["result"]["point_pairs"] = _pairs(7)
    alternative["result"]["epsilon_witnesses"] = [
        {"epsilon": _q(1, 4), "index": 7, "distance": _q(1, 7)},
        {"epsilon": _q(1, 8), "index": 9, "distance": _q(1, 9)},
        {"epsilon": _q(1, 16), "index": 17, "distance": _q(1, 17)},
        {"epsilon": _q(1, 32), "index": 33, "distance": _q(1, 33)},
    ]
    assert _verify(tmp_path / "alternative", alternative).reward == 1.0


def test_rejects_corrupt_geometry_and_nonvanishing_gap(tmp_path: Path) -> None:
    for name, mutation in [
        (
            "coordinate",
            lambda result: result["point_pairs"][3]["b"].__setitem__(1, _q(1, 8)),
        ),
        (
            "distance",
            lambda result: result["epsilon_witnesses"][2].update(distance=_q(1, 10)),
        ),
        (
            "ordering",
            lambda result: result["epsilon_witnesses"][2].update(epsilon=_q(1, 4)),
        ),
    ]:
        submission = copy.deepcopy(_oracle())
        mutation(submission["result"])
        assert _verify(tmp_path / name, submission).reward == 0.0


def test_public_instruction_does_not_require_lowest_terms() -> None:
    instruction = (
        Path("benchmarks/datasets/public-reproductions-v1") / TASK / "instruction.md"
    ).read_text()
    lowered = instruction.lower()
    assert "lowest terms" not in lowered
    assert "canonical rational" not in lowered
    assert "equivalent encodings" in lowered


def test_rejects_string_coercion_and_accepts_equivalent_rationals(
    tmp_path: Path,
) -> None:
    string_submission = copy.deepcopy(_oracle())
    string_submission["result"]["point_pairs"][0]["distance"] = "1/4"
    assert _verify(tmp_path / "string", string_submission).reward == 0.0

    noncanonical_submission = copy.deepcopy(_oracle())
    noncanonical_submission["result"]["point_pairs"][0]["distance"] = _q(2, 8)
    assert _verify(tmp_path / "noncanonical", noncanonical_submission).reward == 1.0


def test_accepts_epsilon_witnesses_above_one(tmp_path: Path) -> None:
    """T2: the public contract allows any positive epsilon; no hidden < 1 bound."""
    submission = copy.deepcopy(_oracle())
    submission["result"]["epsilon_witnesses"] = [
        {"epsilon": _q(2), "index": 4, "distance": _q(1, 4)},
        {"epsilon": _q(3, 2), "index": 6, "distance": _q(1, 6)},
        {"epsilon": _q(1), "index": 11, "distance": _q(1, 11)},
        {"epsilon": _q(1, 2), "index": 21, "distance": _q(1, 21)},
    ]
    result = _verify(tmp_path / "epsilon-above-one", submission)
    assert result.reward == 1.0
    assert result.details["correctness"] == 1.0


def test_rejects_float_point_pair_indices(tmp_path: Path) -> None:
    """T4: float indices like 4.0 must not bypass integer validation."""
    submission = copy.deepcopy(_oracle())
    for _i, row in enumerate(submission["result"]["point_pairs"]):
        row["index"] = float(row["index"])
    result = _verify(tmp_path / "float-indices", submission)
    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0


def test_extra_witness_key_is_rejected(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path / "extra-witness", submission)
    payload = json.loads((app / "submission.json").read_text())
    payload["witness"] = []
    (app / "submission.json").write_text(json.dumps(payload))
    result = _run_verifier(task, app, logs)
    assert result.reward == 0.0


def test_tampered_input_preserves_correctness_and_gates_reward(
    tmp_path: Path,
) -> None:
    """T6: input binding is a separate diagnostic; math stays correct."""
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path / "tampered-input", submission)
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    input_path.write_text(json.dumps(input_data))
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0
