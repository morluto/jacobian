import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    witness_list_is_bound,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def image(p):
    return [p[0] + p[1], p[0] * p[1]]


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    r = submission.get("result") if protocol_ok else None
    r = r if isinstance(r, dict) else {}
    p = x["point_p"]
    q = x["point_q"]
    ip = image(p)
    iq = image(q)
    math_correct = bool(
        protocol_ok
        and workspace_input_is_bound()
        and p != q
        and r.get("image_p") == ip
        and r.get("image_q") == iq
        and ip == iq
    )
    witness = submission.get("witness") if protocol_ok else None
    evidence_ok = bool(
        protocol_ok
        and witness_list_is_bound(witness, expected_path="evidence/answer.txt")
        and resolve_evidence(witness[0], expected_path="evidence/answer.txt")
        is not None
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=evidence_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(evidence_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
