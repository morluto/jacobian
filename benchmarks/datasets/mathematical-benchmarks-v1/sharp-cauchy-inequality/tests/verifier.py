import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")
ZERO = (0, 0, 0, 0, 0, 0)


def _load_frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (
            (W / "input.json").is_symlink()
            or (E / "input.json").is_symlink()
            or (W / "input.json").read_bytes() != raw
        ):
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _poly(terms):
    if not isinstance(terms, list):
        return None
    out, order = {}, []
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"exponents", "coefficient"}:
            return None
        exps, coefficient = term["exponents"], term["coefficient"]
        if (
            not isinstance(exps, list)
            or len(exps) != 6
            or any(type(e) is not int or e < 0 for e in exps)
            or type(coefficient) is not int
            or coefficient == 0
        ):
            return None
        key = tuple(exps)
        if key in out:
            return None
        out[key], order = coefficient, [*order, key]
    return out if order == sorted(order) else None


def _add(*parts):
    out = {}
    for scale, poly in parts:
        for monomial, coefficient in poly.items():
            out[monomial] = out.get(monomial, 0) + scale * coefficient
            if out[monomial] == 0:
                del out[monomial]
    return out


def _mul(left, right):
    out = {}
    for lm, lc in left.items():
        for rm, rc in right.items():
            key = tuple(a + b for a, b in zip(lm, rm, strict=True))
            out[key] = out.get(key, 0) + lc * rc
    return {key: value for key, value in out.items() if value}


def _var(i):
    exponent = [0] * 6
    exponent[i] = 1
    return {tuple(exponent): 1}


def _square(poly):
    return _mul(poly, poly)


def _expected():
    a, b, c, x, y, z = (_var(i) for i in range(6))
    d = _add((1, _mul(a, x)), (1, _mul(b, y)), (1, _mul(c, z)))
    u = _add((1, _mul(a, b)), (1, _mul(b, c)), (1, _mul(c, a)))
    v = _add((1, _mul(x, y)), (1, _mul(y, z)), (1, _mul(z, x)))
    a2 = _add((1, _square(a)), (1, _square(b)), (1, _square(c)))
    x2 = _add((1, _square(x)), (1, _square(y)), (1, _square(z)))
    residual = _add((1, {ZERO: 1}), (-1, d), (-1, u), (-1, v))
    sos = _add(
        (1, _square(_add((1, a), (-1, x)))),
        (1, _square(_add((1, b), (-1, y)))),
        (1, _square(_add((1, c), (-1, z)))),
    )
    gram = _add((1, _mul(a2, x2)), (-1, _square(d)))
    minors = _add(
        (1, _square(_add((1, _mul(a, y)), (-1, _mul(b, x))))),
        (1, _square(_add((1, _mul(a, z)), (-1, _mul(c, x))))),
        (1, _square(_add((1, _mul(b, z)), (-1, _mul(c, y))))),
    )
    total_a = _square(_add((1, a), (1, b), (1, c)))
    total_x = _square(_add((1, x), (1, y), (1, z)))
    constraint_residual = _add((2, {ZERO: 1}), (-1, total_a), (-1, total_x))
    return {
        "d": d,
        "u": u,
        "v": v,
        "residual": residual,
        "sos_twice": sos,
        "constraint_residual": constraint_residual,
        "a2": a2,
        "x2": x2,
        "across": _add((2, u)),
        "xcross": _add((2, v)),
        "gram_residual": gram,
        "gram_sos": minors,
        "total_a_square": total_a,
        "total_x_square": total_x,
    }


def _certificate_ok(mode, cert):
    if not isinstance(cert, dict):
        return False
    expected = _expected()
    if mode == "DIRECT_SOS":
        keys = {"d", "u", "v", "residual", "constraint_residual", "sos_factors"}
    elif mode == "AMGM_SQUARES":
        keys = {"d", "u", "v", "residual", "sos_twice", "constraint_residual"}
    else:
        keys = {
            "d",
            "a2",
            "x2",
            "across",
            "xcross",
            "gram_residual",
            "gram_sos",
            "total_a_square",
            "total_x_square",
        }
    if set(cert) != keys:
        return False
    core_keys = [k for k in keys if k != "sos_factors"]
    parsed = {key: _poly(cert[key]) for key in core_keys}
    if any(value is None for value in parsed.values()) or any(
        parsed[key] != expected[key] for key in core_keys
    ):
        return False
    if mode == "AMGM_SQUARES":
        return (
            _add((2, parsed["residual"]), (-1, parsed["sos_twice"]))
            == parsed["constraint_residual"]
        )
    if mode == "DIRECT_SOS":
        factors_raw = cert["sos_factors"]
        if not isinstance(factors_raw, list) or not factors_raw:
            return False
        factors = [_poly(item) for item in factors_raw]
        if any(item is None for item in factors):
            return False
        target = _add((2, parsed["residual"]), (-1, parsed["constraint_residual"]))
        return _add(*((1, _square(f)) for f in factors)) == target
    return (
        parsed["gram_residual"] == parsed["gram_sos"]
        and _add((1, parsed["a2"]), (1, parsed["across"])) == parsed["total_a_square"]
        and _add((1, parsed["x2"]), (1, parsed["xcross"])) == parsed["total_x_square"]
    )


def _rat(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    return Fraction(value["numerator"], value["denominator"])


def _sqrt_fraction(value):
    if value < 0:
        return None
    n, d = isqrt(value.numerator), isqrt(value.denominator)
    return (
        Fraction(n, d)
        if n * n == value.numerator and d * d == value.denominator
        else None
    )


def _witness_ok(witness, constant):
    if not isinstance(witness, dict) or set(witness) != {"a", "b", "c", "x", "y", "z"}:
        return False
    values = [_rat(witness[key]) for key in ("a", "b", "c", "x", "y", "z")]
    if any(value is None or value <= 0 for value in values):
        return False
    a, b, c, x, y, z = values
    if x + y + z != 1:
        return False
    radical = _sqrt_fraction((x * y + y * z + z * x) * (a * b + b * c + c * a))
    return (
        radical is not None and a * x + b * y + c * z + constant * radical == a + b + c
    )


def _result_ok(result, frozen):
    if (
        not isinstance(result, dict)
        or set(result) != {"constant", "proof_mode", "certificate", "equality_witness"}
        or frozen.get("coefficient_domain") != "ZZ"
    ):
        return False
    constant = _rat(result["constant"])
    return (
        constant == 2
        and result["proof_mode"] in frozen.get("proof_modes", [])
        and _certificate_ok(result["proof_mode"], result["certificate"])
        and _witness_ok(result["equality_witness"], constant)
    )


def main():
    submission, frozen = load_submission(), _load_frozen()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_ok(submission.get("result"), frozen))
    evidence = None
    if (
        contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    ):
        evidence = read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/inequality-certificate.json",
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and evidence["result"] == submission.get("result")
        and evidence["limitations"] == submission.get("limitations")
    )
    scope_ok = bool(
        contract
        and submission.get("scope")
        == "FROZEN_INEQUALITY_AND_SUBMITTED_SYMBOLIC_CERTIFICATE"
        and submission.get("limitations")
        == [
            "ELEMENTARY_REAL_ORDER_AND_SQUARE_ROOT_LEMMAS_TRUSTED",
            "POSITIVE_HOMOGENEITY_NORMALIZATION_TRUSTED",
        ]
    )
    assurance_ok = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = math_correct and evidence_valid and scope_ok and not false_certification
    reward = float(correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
