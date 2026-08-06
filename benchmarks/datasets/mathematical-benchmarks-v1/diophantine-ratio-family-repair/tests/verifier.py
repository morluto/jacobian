import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_INPUT_BYTES,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "The checker proves the submitted polynomial-family identities over Z; it "
    "does not classify every solution of the original divisibility condition."
)


def _load_frozen_input() -> dict[str, Any]:
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if not workspace_input_is_bound() or not all(
            is_regular_bounded_file(path, max_bytes=MAX_INPUT_BYTES)
            for path in (workspace, frozen)
        ):
            return {}
        with frozen.open("rb") as stream:
            payload = stream.read(MAX_INPUT_BYTES + 1)
        if len(payload) > MAX_INPUT_BYTES:
            return {}
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _trim(poly: list[int]) -> list[int]:
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _add(left: list[int], right: list[int]) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for i in range(size):
        result[i] = (left[i] if i < len(left) else 0) + (
            right[i] if i < len(right) else 0
        )
    return _trim(result)


def _sub(left: list[int], right: list[int]) -> list[int]:
    return _add(left, [-item for item in right])


def _mul(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim(result)


def _eval(poly: list[int], value: int) -> int:
    total = 0
    for coefficient in reversed(poly):
        total = total * value + coefficient
    return total


def _poly(value: object) -> list[int] | None:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 12
        or any(type(item) is not int for item in value)
        or any(abs(item) > 1_000_000 for item in value)
    ):
        return None
    return _trim(value)


def _shifted(poly: list[int], origin: int = 2) -> list[int]:
    result = [0] * len(poly)
    for degree, coefficient in enumerate(poly):
        for power in range(degree + 1):
            result[power] += (
                coefficient * math.comb(degree, power) * (origin ** (degree - power))
            )
    return _trim(result)


def _positive_for_t_ge_2(poly: list[int]) -> bool:
    shifted = _shifted(poly)
    return shifted[0] > 0 and all(coefficient >= 0 for coefficient in shifted)


def _strictly_increasing_for_t_ge_2(poly: list[int]) -> bool:
    shifted = _shifted(poly)
    return all(coefficient >= 0 for coefficient in shifted) and any(
        degree > 0 and coefficient > 0 for degree, coefficient in enumerate(shifted)
    )


def _family_shape(value: object) -> bool:
    fields = {
        "a",
        "b",
        "d",
        "x",
        "y",
        "norm",
        "square_congruence_factor",
        "quotient",
        "divisibility_quotient",
        "ratio",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == fields
        and all(_poly(value[key]) is not None for key in fields)
    )


def _probes_shape(value: object) -> bool:
    if not isinstance(value, list) or not 3 <= len(value) <= 6:
        return False
    seen: set[int] = set()
    for probe in value:
        if not isinstance(probe, dict) or set(probe) != {
            "t",
            "x",
            "y",
            "divisor",
            "multiple",
            "ratio",
        }:
            return False
        if type(probe["t"]) is not int or not 2 <= probe["t"] <= 50:
            return False
        if probe["t"] in seen or any(
            type(probe[name]) is not int for name in ("x", "y", "divisor", "multiple")
        ):
            return False
        ratio = probe["ratio"]
        if (
            not isinstance(ratio, list)
            or len(ratio) != 2
            or any(type(item) is not int for item in ratio)
        ):
            return False
        seen.add(probe["t"])
    return True


def _result_shape_is_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "source_audit",
        "family",
        "probes",
        "conclusion",
    }:
        return False
    audit = result["source_audit"]
    return bool(
        isinstance(audit, dict)
        and set(audit)
        == {
            "invalid_step",
            "k",
            "claimed_partner",
            "status_for_d_ge_2",
            "downstream_recurrence_status",
        }
        and all(type(value) is str and value for value in audit.values())
        and _family_shape(result["family"])
        and _probes_shape(result["probes"])
        and type(result["conclusion"]) is str
    )


def _family_is_valid(value: object) -> bool:
    fields = {
        "a",
        "b",
        "d",
        "x",
        "y",
        "norm",
        "square_congruence_factor",
        "quotient",
        "divisibility_quotient",
        "ratio",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    polynomials = {key: _poly(value[key]) for key in fields}
    if any(poly is None for poly in polynomials.values()):
        return False
    p = {key: poly for key, poly in polynomials.items() if poly is not None}
    one = [1]
    # Bind x=d*a, y=d*b, and norm=a^2-a*b+b^2.
    if p["x"] != _mul(p["d"], p["a"]) or p["y"] != _mul(p["d"], p["b"]):
        return False
    norm = _add(_sub(_mul(p["a"], p["a"]), _mul(p["a"], p["b"])), _mul(p["b"], p["b"]))
    if p["norm"] != norm:
        return False
    # d^2-(1-a)=(t^2-1)*norm and d^2*a-1=quotient*norm.
    if _sub(_mul(p["d"], p["d"]), _sub(one, p["a"])) != _mul(
        p["square_congruence_factor"], norm
    ):
        return False
    if _sub(_mul(_mul(p["d"], p["d"]), p["a"]), one) != _mul(p["quotient"], norm):
        return False
    expected_divisor = _add(
        _sub(_mul(p["x"], p["x"]), _mul(p["x"], p["y"])), _mul(p["y"], p["y"])
    )
    expected_multiple = _mul(_mul(p["x"], p["y"]), _sub(_mul(p["x"], p["y"]), one))
    if expected_multiple != _mul(expected_divisor, p["divisibility_quotient"]):
        return False
    if p["x"] != _mul(p["ratio"], p["y"]):
        return False
    return (
        _positive_for_t_ge_2(p["x"])
        and _positive_for_t_ge_2(p["y"])
        and _positive_for_t_ge_2(p["ratio"])
        and _strictly_increasing_for_t_ge_2(p["ratio"])
    )


def _probes_are_valid(value: object, family: dict[str, Any]) -> bool:
    if not _probes_shape(value):
        return False
    parameters: list[int] = []
    for probe in value:
        t = probe["t"]
        if t in parameters:
            return False
        parameters.append(t)
        expected = {name: _eval(family[name], t) for name in ("x", "y")}
        x, y = expected["x"], expected["y"]
        ratio_value = _eval(family["ratio"], t)
        divisor = x * x - x * y + y * y
        multiple = x * y * (x * y - 1)
        if probe != {
            "t": t,
            "x": x,
            "y": y,
            "divisor": divisor,
            "multiple": multiple,
            "ratio": [ratio_value, 1],
        }:
            return False
        if (
            x != ratio_value * y
            or x <= 0
            or y <= 0
            or multiple < 1
            or multiple % divisor
        ):
            return False
    return len(set(parameters)) == len(parameters)


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    if (
        not isinstance(result, dict)
        or set(result) != {"source_audit", "family", "probes", "conclusion"}
        or source.get("source", {}).get("row_sha256")
        != "sha256:3a674d8336d1c61f3561109ac56ed2ba79f987476aa727ff4ec4e209e98b7ab8"
    ):
        return False
    audit = result["source_audit"]
    if not isinstance(audit, dict) or audit != {
        "invalid_step": "VIETA_PARTNER_INTEGRALITY",
        "k": "d^2-1",
        "claimed_partner": "d^2/(d^2-1)",
        "status_for_d_ge_2": "NONINTEGER",
        "downstream_recurrence_status": "UNSUPPORTED",
    }:
        return False
    if not _family_is_valid(result["family"]):
        return False
    family = {key: _poly(value) for key, value in result["family"].items()}
    if any(value is None for value in family.values()):
        return False
    exact_family = {key: value for key, value in family.items() if value is not None}
    return bool(
        _probes_are_valid(result["probes"], exact_family)
        and result["conclusion"] == "INFINITELY_MANY_DISTINCT_RATIOS_CERTIFIED"
    )


def _evidence_matches_result(evidence: object, result: dict[str, Any]) -> bool:
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(
        evidence[0],
        expected_path="evidence/answer.txt",
        max_bytes=MAX_EVIDENCE_BYTES,
    )
    if target is None:
        return False
    try:
        markers = [
            line.removeprefix("RESULT_JSON:").strip()
            for line in target.read_text().splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        return len(markers) == 1 and _json_exact_equal(json.loads(markers[0]), result)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return False


def _json_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _evidence_descriptors_ok(evidence: object) -> bool:
    """Check the evidence descriptor shape, path, and digest syntax.

    File-content binding is left to ``evidence_validity``; this predicate only
    ensures the envelope descriptor matches the public schema so a malformed
    descriptor such as ``[null]`` is reported as a protocol failure.
    """

    return bool(
        isinstance(evidence, list)
        and len(evidence) == 1
        and isinstance(evidence[0], dict)
        and set(evidence[0]) == {"path", "sha256"}
        and evidence[0].get("path") == "evidence/answer.txt"
        and isinstance(evidence[0].get("sha256"), str)
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = data.get("result")
    math_correct = bool(_result_is_valid(result, source))
    evidence_valid = bool(
        isinstance(result, dict)
        and _evidence_matches_result(data.get("evidence"), result)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(data.get("limitations") == [LIMITATION])
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    protocol = bool(
        contract
        and _result_shape_is_valid(result)
        and _evidence_descriptors_ok(data.get("evidence"))
        and scope_correct
        and assurance_correct
        and limitations_correct
    )
    correct = bool(
        protocol and math_correct and evidence_valid and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol),
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
