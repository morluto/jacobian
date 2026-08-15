import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    witness_list_is_bound,
)

W = Path("/app")
E = Path("/tests")


def rat(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    n, d = value["numerator"], value["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError
    return Fraction(n, d)


def frozen_valid():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        return (
            not workspace.is_symlink()
            and not frozen.is_symlink()
            and workspace.read_bytes() == frozen.read_bytes()
            and json.loads(frozen.read_text())["source"]["row_sha256"]
            == "sha256:ee2ab1f4d77914e3d6e39f3a90258db21adf9dc081faa4708c8bc1bd3693a131"
        )
    except (OSError, ValueError, KeyError):
        return False


def certificate_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "collision",
        "fourier_block",
        "norm_direction",
        "radical_domain",
    }:
        return False
    try:
        c = result["collision"]
        if set(c) != {"first_index", "second_index", "alpha_first", "alpha_second"}:
            return False
        i, j, ai, aj = (
            c["first_index"],
            c["second_index"],
            c["alpha_first"],
            c["alpha_second"],
        )
        collision = (
            all(type(x) is int for x in (i, j, ai, aj))
            and all(-20 <= x <= 20 for x in (i, j, ai, aj))
            and i != j
            and i + ai == j + aj
        )

        f = result["fourier_block"]
        if set(f) != {"size", "operator_norm_squared"}:
            return False
        n = f["size"]
        fourier = (
            type(n) is int
            and 3 <= n <= 15
            and n % 2 == 1
            and rat(f["operator_norm_squared"]) == Fraction(n, 4)
        )

        d = result["norm_direction"]
        if set(d) != {
            "valid_relation",
            "diagonal_entries",
            "operator_norm_squared",
            "hilbert_schmidt_norm_squared",
        }:
            return False
        entries = [rat(x) for x in d["diagonal_entries"]]
        op2 = max(abs(x) for x in entries) ** 2
        hs2 = sum(x * x for x in entries)
        norms = (
            d["valid_relation"] == "OPERATOR_NORM_LE_HILBERT_SCHMIDT_NORM"
            and 2 <= len(entries) <= 8
            and sum(x != 0 for x in entries) >= 2
            and rat(d["operator_norm_squared"]) == op2
            and rat(d["hilbert_schmidt_norm_squared"]) == hs2
            and op2 < hs2
        )

        r = result["radical_domain"]
        if set(r) != {"m", "radicand", "real_status"}:
            return False
        m = r["m"]
        radical = (
            type(m) is int
            and 2 <= m <= 100
            and r["radicand"] == 1 - m < 0
            and r["real_status"] == "NOT_REAL"
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return collision and fourier and norms and radical


def evidence_valid(evidence, result):
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not witness_list_is_bound(
            evidence, expected_path="evidence/audit-certificate.json"
        )
    ):
        return False
    target = resolve_evidence(
        evidence[0], expected_path="evidence/audit-certificate.json"
    )
    try:
        return (
            target is not None
            and target.stat().st_size <= 1_048_576
            and json.loads(target.read_text()) == result
        )
    except (OSError, ValueError, RecursionError):
        return False


def main():
    submission_path = W / "submission.json"
    try:
        oversized = submission_path.stat().st_size > 1_048_576
    except OSError:
        oversized = True
    submission = None if oversized else load_submission()
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    correctness = bool(protocol_ok and frozen_valid() and certificate_valid(result))
    evidence = bool(protocol_ok and evidence_valid(submission.get("witness"), result))
    reward = aggregate_reward(
        correctness=correctness,
        witness_validity=evidence,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correctness),
                "witness_validity": float(evidence),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier") / "reward.json")


if __name__ == "__main__":
    main()
