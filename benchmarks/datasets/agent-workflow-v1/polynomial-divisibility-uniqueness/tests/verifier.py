from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "This exact certificate covers only the frozen dividend and divisor family; it is not a general polynomial-factorization result."
RESULT_KEYS = frozenset(
    {"parameter", "remainder_constant", "remainder_x", "common_gcd", "quotient"}
)


def trim(p: list[Fraction]) -> list[Fraction]:
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def add(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * max(len(p), len(q))
    for i, value in enumerate(p):
        out[i] += value
    for i, value in enumerate(q):
        out[i] += value
    return trim(out)


def scale(p: list[Fraction], c: Fraction) -> list[Fraction]:
    return trim([c * value for value in p])


def shift(p: list[Fraction]) -> list[Fraction]:
    return [Fraction(0), *p]


def divmod_poly(
    numerator: list[Fraction], denominator: list[Fraction]
) -> tuple[list[Fraction], list[Fraction]]:
    rem = trim(numerator[:])
    den = trim(denominator[:])
    if den == [0]:
        raise ZeroDivisionError
    quotient = [Fraction(0)] * max(1, len(rem) - len(den) + 1)
    while rem != [0] and len(rem) >= len(den):
        degree = len(rem) - len(den)
        coefficient = rem[-1] / den[-1]
        quotient[degree] += coefficient
        for i, value in enumerate(den):
            rem[i + degree] -= coefficient * value
        trim(rem)
    return trim(quotient), trim(rem)


def monic_gcd(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    left, right = trim(p[:]), trim(q[:])
    while right != [0]:
        _, remainder = divmod_poly(left, right)
        left, right = right, remainder
    return scale(left, Fraction(1, 1) / left[-1])


def derive_unique_parameter(gcd: list[Fraction]) -> int | None:
    """Derive the unique integer parameter from a monic linear gcd c0 + c1*a.

    A monic linear gcd with c1 != 0 has exactly one rational root a = -c0/c1.
    Return that root as an int when it is integral, otherwise None.  Nonlinear
    or constant gcds also return None: they cannot certify a *unique* integer
    parameter.  Extracted from ``main`` so the derivation logic can be exercised
    with controlled gcds whose root is not the canonical value 2.
    """
    if len(gcd) != 2 or gcd[1] == 0:
        return None
    root = -gcd[0] / gcd[1]
    if root.denominator != 1:
        return None
    return int(root.numerator)


def symbolic_remainder() -> tuple[list[Fraction], list[Fraction]]:
    # A residue is c0(a) + c1(a)x. Multiplication by x uses x^2=x-a.
    c0, c1 = [Fraction(1)], [Fraction(0)]
    powers: list[tuple[list[Fraction], list[Fraction]]] = []
    for _ in range(14):
        powers.append((c0, c1))
        c0, c1 = scale(shift(c1), Fraction(-1)), add(c0, c1)
    r0 = add(add(scale(powers[0][0], Fraction(90)), powers[1][0]), powers[13][0])
    r1 = add(add(scale(powers[0][1], Fraction(90)), powers[1][1]), powers[13][1])
    return trim(r0), trim(r1)


def multiply(p: list[int], q: list[int]) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, left in enumerate(p):
        for j, right in enumerate(q):
            out[i + j] += left * right
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def integers(value: Any) -> list[int] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or (isinstance(x, float) and not x.is_integer())
            for x in value
        )
    ):
        return None
    return [int(x) for x in value]


def integer_value(value: Any) -> int | None:
    """Normalize a JSON number accepted by the schema's ``integer`` type."""

    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def load_regular_submission() -> dict[str, Any] | None:
    """Parse the submission, rejecting symlinked submission artifacts.

    The shared ``load_submission`` helper follows symlinks; this local wrapper
    requires the declared submission path to be a regular file rooted under
    the workspace so an agent cannot alias it to another workspace file.
    Deeply nested JSON can raise ``RecursionError`` from ``json.loads``; catch
    it here so a malformed submission scores zero instead of crashing the
    verifier without writing ``reward.json``.
    """
    path = WORKSPACE / "submission.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
            return None
        root = WORKSPACE.resolve()
        target = path.resolve(strict=True)
        if not target.is_relative_to(root):
            return None
    except OSError:
        return None
    try:
        return load_submission(path)
    except RecursionError:
        return None


def evidence_matches_result(evidence: object, result: object) -> bool:
    """Bind the evidence file to the submitted result and require a derivation.

    Rejects empty or arbitrary evidence text: the file must contain a
    ``RESULT_JSON:`` marker matching the submitted result object and a
    non-empty human-readable derivation body with concrete certificate values.
    The evidence list must contain
    exactly one descriptor (submission_schema.json sets maxItems: 1).
    """
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    try:
        if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
            return False
        target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    except RecursionError:
        return False
    if target is None:
        return False
    try:
        if target.stat().st_size > 1_048_576:
            return False
        text = target.read_text()
        lines = text.splitlines()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in lines
            if line.startswith("RESULT_JSON:")
        )
        body = "\n".join(line for line in lines if not line.startswith("RESULT_JSON:"))
        lowered = body.lower()
        compact = "".join(lowered.split())
        parameter = (
            integer_value(result.get("parameter")) if isinstance(result, dict) else None
        )
        expected_fragments = tuple(
            str(result.get(key)).replace(" ", "")
            for key in (
                "remainder_constant",
                "remainder_x",
                "common_gcd",
                "quotient",
            )
        )
        parameter_is_stated = bool(
            parameter is not None
            and any(
                form in compact
                for form in (
                    f"a={parameter}",
                    f"rootis{parameter}",
                    f"root={parameter}",
                    f"parameteris{parameter}",
                )
            )
        )
        return (
            json.loads(marker) == result
            and len(body) >= 80
            and "gcd" in lowered
            and "remainder" in lowered
            and ("product" in lowered or "multiplication" in lowered)
            and parameter_is_stated
            and "unique" in lowered
            and "root" in lowered
            and all(fragment in compact for fragment in expected_fragments)
        )
    except (OSError, StopIteration, UnicodeError, ValueError, RecursionError):
        return False


def limitation_is_bounded(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    bounded_language = (
        "only",
        "specific",
        "frozen",
        "limited",
        "restricted",
        "supplied",
        "given",
        "not a general",
        "does not generalize",
    )
    subject = ("polynomial", "family", "divisibility", "certificate", "input")
    return any(word in lowered for word in subject) and any(
        phrase in lowered for phrase in bounded_language
    )


def main() -> None:
    submission = load_regular_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    try:
        workspace_input = WORKSPACE / "input.json"
        frozen_input = TESTS / "input.json"
        frozen_input_ok = bool(
            all(
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= 1_048_576
                for path in (workspace_input, frozen_input)
            )
            and workspace_input.read_bytes() == frozen_input.read_bytes()
        )
    except OSError:
        frozen_input_ok = False
    result = data.get("result", {})
    result_typed = result if isinstance(result, dict) else {}
    # Thread 1: reject result objects outside the advertised schema
    # (submission_schema.json sets additionalProperties: false on result).
    result_schema_ok = isinstance(result, dict) and set(result) == RESULT_KEYS
    r0 = integers(result_typed.get("remainder_constant"))
    r1 = integers(result_typed.get("remainder_x"))
    gcd = integers(result_typed.get("common_gcd"))
    quotient = integers(result_typed.get("quotient"))
    parameter = result_typed.get("parameter")
    parameter_value = integer_value(parameter)

    computed_r0, computed_r1 = symbolic_remainder()
    computed_gcd = monic_gcd(computed_r0, computed_r1)
    # Uniqueness and the parameter are consequences of the recomputed gcd,
    # not hard-coded expectations.  A monic linear gcd c0 + c1*a (c1 != 0)
    # has exactly one rational root a = -c0/c1; that root is the unique
    # integer parameter, and its integrality is what licenses UNIQUE_PARAMETER.
    gcd_is_linear = len(computed_gcd) == 2 and computed_gcd[1] != 0
    derived_parameter_int = derive_unique_parameter(computed_gcd)
    parameter_is_integer = derived_parameter_int is not None
    symbolic_ok = (
        r0 is not None
        and r1 is not None
        and trim([Fraction(x) for x in r0]) == computed_r0
        and trim([Fraction(x) for x in r1]) == computed_r1
        and gcd is not None
        and trim([Fraction(x) for x in gcd]) == computed_gcd
        and gcd_is_linear
        and parameter_is_integer
    )
    quotient_ok = (
        parameter_value is not None
        and parameter_value == derived_parameter_int
        and quotient is not None
        and trim(multiply([parameter_value, -1, 1], trim(quotient)))
        == [90, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )
    # Thread 4: keep mathematical correctness independent of input integrity.
    math_correct = bool(contract and result_schema_ok and symbolic_ok and quotient_ok)
    # Thread 2: validate evidence content, not just path and digest.
    # Input integrity is reported separately and gates only the aggregate
    # reward; coupling it here would make evidence failures indistinguishable
    # from input-tampering failures.
    evidence_valid = bool(
        math_correct and evidence_matches_result(data.get("evidence"), result_typed)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(limitation_is_bounded(item) for item in data["limitations"])
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    # Thread 4: gate the aggregate reward on input integrity without
    # corrupting the independently computed mathematical-correctness signal.
    passed = bool(
        math_correct
        and frozen_input_ok
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "input_integrity": float(frozen_input_ok),
                "reward": float(passed),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
