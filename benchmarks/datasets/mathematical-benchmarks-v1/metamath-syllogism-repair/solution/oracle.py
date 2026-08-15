from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _consume(
    pattern: list[str], actual: list[str], values: dict[str, list[str]]
) -> None:
    cursor = 0
    for token in pattern:
        if token not in {"u", "v", "w"}:
            if cursor >= len(actual) or actual[cursor] != token:
                raise RuntimeError("oracle pattern mismatch")
            cursor += 1
            continue
        value = values.get(token)
        if value is None:
            if actual[cursor] == "(":
                depth, end = 0, cursor
                while end < len(actual):
                    depth += actual[end] == "("
                    depth -= actual[end] == ")"
                    end += 1
                    if depth == 0:
                        break
                value = actual[cursor:end]
            else:
                value = [actual[cursor]]
            values[token] = value
        if actual[cursor : cursor + len(value)] != value:
            raise RuntimeError("oracle substitution mismatch")
        cursor += len(value)
    if cursor != len(actual):
        raise RuntimeError("oracle left unconsumed tokens")


def _oracle_replay(input_data: dict, proof: list[str]) -> list[dict]:
    stack: list[list[str]] = []
    trace: list[dict] = []
    for position, label in enumerate(proof):
        values: dict[str, list[str]] = {}
        if label in input_data["atomic_entries"]:
            stack.append(list(input_data["atomic_entries"][label]))
        else:
            assertion = input_data["assertions"][label]
            count = len(assertion["hypotheses"])
            actuals = stack[-count:]
            del stack[-count:]
            for pattern, actual in zip(assertion["hypotheses"], actuals, strict=True):
                _consume(pattern, actual, values)
            conclusion: list[str] = []
            for token in assertion["conclusion"]:
                conclusion.extend(values.get(token, [token]))
            stack.append(conclusion)
        trace.append(
            {
                "position": position,
                "label": label,
                "substitution": values,
                "stack_depth": len(stack),
                "stack_top": stack[-1],
            }
        )
    if stack != [input_data["target"]]:
        raise RuntimeError("oracle proof did not reach target")
    return trace


def build_submission(output_root: Path) -> dict[str, object]:
    input_path = output_root / "input.json"
    if not input_path.is_file():
        input_path = ROOT / "environment" / "input.json"
    input_data = json.loads(input_path.read_text())
    proof = ["wu", "wv", "ww", "h1", "wv", "ww", "wi", "wu", "h2", "a1i", "mpd"]
    result = {
        "repaired_proof": proof,
        "changed_positions": [6, 9],
        "trace": _oracle_replay(input_data, proof),
        "final_expression": input_data["target"],
    }
    return {"result": result}


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")
    submission = build_submission(output)
    (output / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n"
    )
