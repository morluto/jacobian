import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_INPUT_BYTES = 1_048_576
# Keep exact-rational arithmetic bounded before repeated squaring and summation
# build progressively larger common denominators in the tail checks.
MAX_FRACTION_BITS = 1_024
# Minimum number of submitted limit coordinates so the tail bound is exercised
# well past the prefix instead of only at the truncation point.
MIN_VERIFICATION_TERMS = 100
PREFIX_LENGTH = 12
# Thread PRRT_kwDOThEfjc6VxiRv: cap the tail-bound exponent before exact
# exponentiation. Fraction(m) ** exponent builds an unbounded integer; a
# schema-valid bound_exponent of 10^10 would require over 1 GB for the
# integer alone, exceeding the verifier memory limit before reward.json is
# written. This conservative cap is far above any mathematically meaningful
# decay rate for a square-summability tail bound.
MAX_BOUND_EXPONENT = 100
_RESULT_FIELDS = {
    "operator_kind",
    "operator_bound",
    "prefixes",
    "limit_coordinates",
    "tail_bound",
}
_PREFIX_FIELDS = {
    "n",
    "weight",
    "preimage_coordinate",
    "limit_norm_sq_partial",
    "preimage_norm_sq_partial",
}
_TAIL_BOUND_FIELDS = {"bound_coefficient", "bound_exponent", "verification_terms"}
_GROWTH_FIELDS = {"bound_coefficient", "bound_exponent"}


def _source() -> dict[str, Any]:
    try:
        frozen_path = TESTS / "input.json"
        visible_path = WORKSPACE / "input.json"
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_INPUT_BYTES
            for path in (frozen_path, visible_path)
        ):
            return {}
        raw = frozen_path.read_bytes()
        if visible_path.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, RecursionError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator < 1
        or numerator.bit_length() > MAX_FRACTION_BITS
        or denominator.bit_length() > MAX_FRACTION_BITS
    ):
        return None
    result = Fraction(numerator, denominator)
    return result


def _positive_fraction(value: object) -> Fraction | None:
    parsed = _fraction(value)
    return parsed if parsed is not None and parsed > 0 else None


def _parse_tail_bound(
    value: object,
) -> tuple[Fraction, int, int] | None:
    if not isinstance(value, dict) or set(value) != _TAIL_BOUND_FIELDS:
        return None
    coefficient = _positive_fraction(value["bound_coefficient"])
    exponent = value["bound_exponent"]
    terms = value["verification_terms"]
    if (
        coefficient is None
        or not isinstance(exponent, int)
        or isinstance(exponent, bool)
        or exponent < 1
        or exponent > MAX_BOUND_EXPONENT
        or type(terms) is not int
        or terms < MIN_VERIFICATION_TERMS
    ):
        return None
    return coefficient, exponent, terms


def _parse_growth(value: object) -> tuple[Fraction, int] | None:
    if not isinstance(value, dict) or set(value) != _GROWTH_FIELDS:
        return None
    coefficient = _positive_fraction(value["bound_coefficient"])
    exponent = value["bound_exponent"]
    if (
        coefficient is None
        or type(exponent) is not int
        or exponent < 1
        or exponent > MAX_BOUND_EXPONENT
    ):
        return None
    return coefficient, exponent


def _parse_limit_coordinates(value: object, terms: int) -> list[Fraction] | None:
    if not isinstance(value, list) or len(value) != terms:
        return None
    parsed: list[Fraction] = []
    for entry in value:
        coordinate = _fraction(entry)
        if coordinate is None:
            return None
        parsed.append(coordinate)
    return parsed


def _prefixes_ok(
    prefixes: object,
    limit_coordinates: list[Fraction],
    bound: Fraction,
    length: int,
    growth: tuple[Fraction, int],
    weighted_shift: bool,
) -> bool:
    if not isinstance(prefixes, list) or len(prefixes) != length:
        return False
    limit_partial = Fraction(0)
    preimage_partial = Fraction(0)
    for index, item in enumerate(prefixes, start=1):
        if not isinstance(item, dict) or set(item) != _PREFIX_FIELDS:
            return False
        # Thread PRRT_kwDOThEfjc6VxiRy: reject JSON booleans for prefix indices.
        # Python treats True == 1, so a bare equality check accepts boolean n
        # values that violate the agent-visible integer schema.
        if type(item["n"]) is not int or item["n"] != index:
            return False
        weight = _positive_fraction(item["weight"])
        preimage_coordinate = _fraction(item["preimage_coordinate"])
        if (
            weight is None
            or preimage_coordinate is None
            or preimage_coordinate == 0
            or weight > bound
        ):
            return False
        # A diagonal witness relates coordinates at the same index. A weighted
        # shift has y_1=0 and y_n=w_{n-1}x_{n-1} for n>=2, so the row's weight
        # is applied to the preceding forced preimage coordinate.
        if weighted_shift:
            relation_ok = (
                limit_coordinates[index - 1] == 0
                if index == 1
                else limit_coordinates[index - 1]
                == weight * _fraction(prefixes[index - 2]["preimage_coordinate"])
            )
        else:
            relation_ok = limit_coordinates[index - 1] == weight * preimage_coordinate
        if not relation_ok:
            return False
        limit_partial += limit_coordinates[index - 1] ** 2
        preimage_partial += preimage_coordinate**2
        if (
            _fraction(item["limit_norm_sq_partial"]) != limit_partial
            or _fraction(item["preimage_norm_sq_partial"]) != preimage_partial
            or preimage_partial < growth[0] * index ** growth[1]
        ):
            return False
    return True


def _tail_bound_ok(
    limit_coordinates: list[Fraction],
    coefficient: Fraction,
    exponent: int,
    terms: int,
    length: int,
) -> bool:
    # sum_{n=m+1}^{terms} y_n^2 <= C / m^d for each prefix index m. exponent >= 1
    # forces the bound to zero, so sum y_n^2 converges and the declared limit is
    # square-summable.
    suffix_sums = [Fraction(0)] * (terms + 2)
    running = Fraction(0)
    for n in range(terms, 0, -1):
        running += limit_coordinates[n - 1] ** 2
        suffix_sums[n] = running
    return all(
        suffix_sums[m + 1] <= coefficient / Fraction(m) ** exponent
        for m in range(1, length + 1)
    )


def _witness(value: object, source: dict[str, Any]) -> bool:
    """Validate a diagonal-operator graph counterexample generically.

    Accepts any bounded positive diagonal weights with a square-summable limit
    ``y`` whose forced preimage ``x`` (related by ``y_n = w_n x_n``) is not
    square-summable, plus a tail bound proving convergence of ``sum y_n^2``.
    The hidden Oracle's exact construction is not required.
    """
    length = source.get("prefix_length")
    if not isinstance(length, int) or length != PREFIX_LENGTH:
        return False
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        return False
    operator_kind = value["operator_kind"]
    if operator_kind not in {"DIAGONAL", "WEIGHTED_SHIFT"}:
        return False
    weighted_shift = operator_kind == "WEIGHTED_SHIFT"
    bound = _positive_fraction(value["operator_bound"])
    if bound is None:
        return False
    tail = _parse_tail_bound(value["tail_bound"])
    if tail is None:
        return False
    coefficient, exponent, terms = tail
    limit_coordinates = _parse_limit_coordinates(value["limit_coordinates"], terms)
    if limit_coordinates is None:
        return False
    growth = _parse_growth(value.get("preimage_growth")) or (Fraction(1), 1)
    return bool(
        growth
        and _prefixes_ok(
            value["prefixes"],
            limit_coordinates,
            bound,
            length,
            growth,
            weighted_shift,
        )
        and _tail_bound_ok(limit_coordinates, coefficient, exponent, terms, length)
    )


def main() -> None:
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if protocol_ok else {}
    source = _source()
    result = data.get("result")
    math_correct = bool(_witness(result, source))
    reward = aggregate_reward(
        correctness=math_correct,
        protocol_ok=protocol_ok,
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
