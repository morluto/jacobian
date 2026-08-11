import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    _public_submission_is_valid,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

E = Path("/tests")
WORKSPACE = Path("/app")


def _submission():
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _valid_design(result, source):
    if not isinstance(result, dict) or not {"order", "blocks"} <= set(result):
        return False
    order = result.get("order")
    blocks = result.get("blocks")
    if type(order) is not int or order != source.get("point_set", {}).get(
        "cardinality"
    ):
        return False
    if not isinstance(blocks, list) or len(blocks) != source.get(
        "required_block_count"
    ):
        return False
    canonical = []
    for block in blocks:
        if (
            not isinstance(block, list)
            or len(block) != 3
            or any(type(point) is not int or not 0 <= point < order for point in block)
            or len(set(block)) != 3
        ):
            return False
        canonical.append(tuple(sorted(block)))
    if len(set(canonical)) != len(canonical):
        return False
    pairs = Counter(pair for block in canonical for pair in combinations(block, 2))
    expected_pairs = set(combinations(range(order), 2))
    return bool(set(pairs) == expected_pairs and set(pairs.values()) == {1})


def _evidence(value, result):
    if (
        not evidence_list_is_bound(value, expected_path="evidence/answer.txt")
        or not isinstance(value, list)
        or len(value) != 1
        or not isinstance(result, dict)
        or not isinstance(result.get("blocks"), list)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_lines = [
        b"steiner-triple-system-certificate-v1",
        f"result_sha256: {digest}".encode(),
        f"order: {result.get('order')}".encode(),
        f"block_count: {len(result['blocks'])}".encode(),
    ]
    return _certificate_matches(path, expected_lines)


def _certificate_matches(path, expected_lines):
    """Match four certificate lines while streaming arbitrary blank padding."""

    max_line_bytes = max(map(len, expected_lines))
    matched = 0
    try:
        for line, overflow in _certificate_lines(path, max_line_bytes):
            if not _certificate_line_matches(line, overflow, expected_lines, matched):
                return False
            matched += 1
    except (OSError, UnicodeError):
        return False
    return matched == len(expected_lines)


def _certificate_lines(path, max_line_bytes):
    line = bytearray()
    started = False
    overflow = False
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            for byte in chunk:
                if byte == ord("\n"):
                    if started:
                        yield bytes(line), overflow
                    line.clear()
                    started = False
                    overflow = False
                    continue
                if not started and byte in b" \t\r\v\f":
                    continue
                started = True
                if len(line) < max_line_bytes:
                    line.append(byte)
                else:
                    overflow = True
        if started:
            yield bytes(line), overflow


def _certificate_line_matches(line, overflow, expected_lines, matched):
    if matched >= len(expected_lines) or overflow:
        return False
    return line.rstrip(b" \t\r\v\f") == expected_lines[matched]


def main():
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    source = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    envelope_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    contract = bool(envelope_contract and _public_submission_is_valid(submission))
    math_correct = _valid_design(data.get("result"), source)
    evidence_valid = _evidence(data.get("evidence"), data.get("result"))
    scope_correct = bool(
        data.get("scope") == expected["required_scope"]
        and data.get("limitations") == expected["limitations"]
    )
    assurance_correct = data.get("claimed_assurance") == expected["maximum_assurance"]
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        input_bound
        and contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "input_binding": float(input_bound),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "protocol_compliance": float(contract),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(output)


if __name__ == "__main__":
    main()
