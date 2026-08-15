import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
MAX_SUBMISSION_BYTES = 1_048_576


def trim(poly: list[Fraction]) -> list[Fraction]:
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def add_poly(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    size = max(len(left), len(right))
    return trim(
        [
            (left[i] if i < len(left) else Fraction(0))
            + (right[i] if i < len(right) else Fraction(0))
            for i in range(size)
        ]
    )


def multiply_poly(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return trim(result)


def scale_poly(poly: list[Fraction], scalar: Fraction) -> list[Fraction]:
    return trim([scalar * value for value in poly])


class RationalFunction:
    def __init__(self, numerator: list[Fraction], denominator: list[Fraction]):
        self.numerator = trim(numerator)
        self.denominator = trim(denominator)
        if self.denominator == [0]:
            raise ValueError("zero denominator")

    def __add__(self, other: "RationalFunction") -> "RationalFunction":
        return RationalFunction(
            add_poly(
                multiply_poly(self.numerator, other.denominator),
                multiply_poly(other.numerator, self.denominator),
            ),
            multiply_poly(self.denominator, other.denominator),
        )

    def __mul__(self, other: "RationalFunction") -> "RationalFunction":
        return RationalFunction(
            multiply_poly(self.numerator, other.numerator),
            multiply_poly(self.denominator, other.denominator),
        )

    def scale(self, scalar: Fraction) -> "RationalFunction":
        return RationalFunction(scale_poly(self.numerator, scalar), self.denominator)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RationalFunction) and trim(
            multiply_poly(self.numerator, other.denominator)
        ) == trim(multiply_poly(other.numerator, self.denominator))


ZERO = RationalFunction([Fraction(0)], [Fraction(1)])
ONE = RationalFunction([Fraction(1)], [Fraction(1)])
T = RationalFunction([Fraction(0), Fraction(1)], [Fraction(1)])
ONE_MINUS_T = RationalFunction([Fraction(1), Fraction(-1)], [Fraction(1)])


def canonical_fraction(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("invalid rational object")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError("invalid rational components")
    parsed = Fraction(numerator, denominator)
    if parsed.numerator != numerator or parsed.denominator != denominator:
        raise ValueError("noncanonical rational")
    return parsed


def rational_list(
    value: object,
    *,
    exact_length: int | None = None,
    max_length: int | None = None,
) -> list[Fraction]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid coefficient list")
    if exact_length is not None and len(value) != exact_length:
        raise ValueError("wrong coefficient count")
    if max_length is not None and len(value) > max_length:
        raise ValueError("coefficient list exceeds bound")
    return [canonical_fraction(item) for item in value]


def evaluate_polynomial(
    coefficients: list[Fraction], argument: RationalFunction
) -> RationalFunction:
    result = ZERO
    power = ONE
    for coefficient in coefficients:
        result = result + power.scale(coefficient)
        power = power * argument
    return result


def load_frozen() -> dict[str, Any]:
    try:
        workspace_input = WORKSPACE / "input.json"
        test_input = TESTS / "input.json"
        if (
            any(
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_SUBMISSION_BYTES
                for path in (workspace_input, test_input)
            )
            or workspace_input.read_bytes() != test_input.read_bytes()
        ):
            return {}
        value = json.loads(test_input.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_bounded_submission() -> dict[str, Any] | None:
    path = WORKSPACE / "submission.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return None
    except OSError:
        return None
    try:
        return load_submission(path)
    except RecursionError:
        return None


def construction_valid(result: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "parameter_a",
        "h_coefficients",
        "v_coefficients",
        "zeta",
        "audit",
        "formal_identities",
        "moment_conclusion",
    }:
        return False
    try:
        parameter = canonical_fraction(result["parameter_a"])
        bounds = frozen["frozen_problem"]["parameter_bounds"]
        if (
            parameter == 0
            or abs(parameter.numerator) > bounds["max_abs_numerator"]
            or parameter.denominator > bounds["max_denominator"]
        ):
            return False
        h = rational_list(result["h_coefficients"], exact_length=2)
        v = rational_list(result["v_coefficients"], exact_length=3)
        zeta_data = result["zeta"]
        if not isinstance(zeta_data, dict) or set(zeta_data) != {
            "numerator",
            "denominator",
        }:
            return False
        zeta = RationalFunction(
            rational_list(zeta_data["numerator"], max_length=16),
            rational_list(zeta_data["denominator"], max_length=16),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False

    h_at_zeta = evaluate_polynomial(h, zeta)
    v_at_zeta = evaluate_polynomial(v, zeta)
    branch_exact = zeta == T * h_at_zeta
    correction = ONE + (T * v_at_zeta).scale(Fraction(-2))
    target_correction = RationalFunction(
        [Fraction(1)],
        multiply_poly([Fraction(1), Fraction(-1)], [Fraction(1), Fraction(-1)]),
    )
    q_mgf = zeta.scale(Fraction(1, 1) / parameter)

    return bool(
        h == [parameter, Fraction(1)]
        and branch_exact
        and correction == target_correction
        and ONE + T.scale(Fraction(-1)) == ONE_MINUS_T
        and q_mgf
        == RationalFunction([Fraction(0), Fraction(1)], [Fraction(1), Fraction(-1)])
        and result["audit"]
        == {
            "finite_checks_suffice": False,
            "repair_kind": "FORMAL_RATIONAL_IDENTITY",
        }
        and result["formal_identities"]
        == {
            "branch_equation": "EXACT",
            "correction_factor": "ONE_MINUS_T_TO_NEGATIVE_TWO",
            "square_root_branch": "CONSTANT_TERM_ONE",
            "combined_prefactor": "ONE",
            "ordinary_mgf": "ONE",
            "mixed_mgf": "T_OVER_ONE_MINUS_T",
        }
        and result["moment_conclusion"]
        == {
            "ordinary": "ZERO",
            "mixed": "M_FACTORIAL",
            "quantifier": "EVERY_INTEGER_M_AT_LEAST_ONE",
        }
    )


def main() -> None:
    submission = load_bounded_submission()
    data = submission if isinstance(submission, dict) else {}
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and construction_valid(data.get("result"), load_frozen())
    )
    reward = float(math_correct)
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
