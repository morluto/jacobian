import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    is_regular_bounded_file,
    load_submission_raw,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
VARIABLES = (
    "a20",
    "a11",
    "a02",
    "b20",
    "b11",
    "b02",
    "b30",
    "b21",
    "b12",
    "b03",
    "t",
)
QUADRATIC = ("a20", "a11", "a02")
CUBIC = ("b30", "b21", "b12", "b03")
GENERATOR_IDS = (
    "j30",
    "j21",
    "j12",
    "j03",
    "j20",
    "j11",
    "j02",
    "j10",
    "j01",
    "rabinowitsch",
)
INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
MAX_INPUT_BYTES = 1_000_000


def _load_frozen_input():
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if not all(
            is_regular_bounded_file(path, max_bytes=MAX_INPUT_BYTES)
            for path in (workspace, frozen)
        ):
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value):
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError
    numerator, denominator = value["num"], value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or INTEGER.fullmatch(numerator) is None
        or INTEGER.fullmatch(denominator) is None
        or len(numerator.lstrip("-")) > 256
        or len(denominator) > 256
    ):
        raise ValueError
    parsed = Fraction(int(numerator), int(denominator))
    if str(parsed.numerator) != numerator or str(parsed.denominator) != denominator:
        raise ValueError
    return parsed


def _polynomial(value):
    if not isinstance(value, dict) or set(value) != {"terms"}:
        raise ValueError
    terms = value["terms"]
    if not isinstance(terms, list) or len(terms) > 1024:
        raise ValueError
    parsed_terms = []
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError
        coefficient = _rational(term["coefficient"])
        exponents = term["exponents"]
        if (
            coefficient == 0
            or not isinstance(exponents, list)
            or len(exponents) != 11
            or any(
                type(exponent) is not int or not 0 <= exponent <= 32
                for exponent in exponents
            )
            or sum(exponents) > 32
        ):
            raise ValueError
        parsed_terms.append((tuple(exponents), coefficient))
    result = {}
    for monomial, coefficient in sorted(parsed_terms, reverse=True):
        if monomial in result:
            raise ValueError
        result[monomial] = coefficient
    return result


def _add(left, right):
    result = dict(left)
    for exponent, coefficient in right.items():
        value = result.get(exponent, Fraction(0)) + coefficient
        if value:
            result[exponent] = value
        else:
            result.pop(exponent, None)
    return result


def _multiply(left, right):
    result = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                a + b for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            if sum(exponents) > 64:
                raise ValueError
            result[exponents] = (
                result.get(exponents, Fraction(0))
                + left_coefficient * right_coefficient
            )
            if result[exponents] == 0:
                del result[exponents]
            if len(result) > 4096:
                raise ValueError
    return result


def _mono(**powers):
    return tuple(powers.get(variable, 0) for variable in VARIABLES)


def _base():
    return {
        "j30": {_mono(a20=1, b21=1): Fraction(2), _mono(a11=1, b30=1): Fraction(-3)},
        "j21": {
            _mono(a20=1, b12=1): Fraction(4),
            _mono(a11=1, b21=1): Fraction(-1),
            _mono(a02=1, b30=1): Fraction(-6),
        },
        "j12": {
            _mono(a20=1, b03=1): Fraction(6),
            _mono(a11=1, b12=1): Fraction(1),
            _mono(a02=1, b21=1): Fraction(-4),
        },
        "j03": {_mono(a11=1, b03=1): Fraction(3), _mono(a02=1, b12=1): Fraction(-2)},
        "j20": {
            _mono(b21=1): Fraction(1),
            _mono(a20=1, b11=1): Fraction(2),
            _mono(a11=1, b20=1): Fraction(-2),
        },
        "j11": {
            _mono(b12=1): Fraction(2),
            _mono(a20=1, b02=1): Fraction(4),
            _mono(a02=1, b20=1): Fraction(-4),
        },
        "j02": {
            _mono(b03=1): Fraction(3),
            _mono(a11=1, b02=1): Fraction(2),
            _mono(a02=1, b11=1): Fraction(-2),
        },
        "j10": {_mono(b11=1): Fraction(1), _mono(a20=1): Fraction(2)},
        "j01": {_mono(b02=1): Fraction(2), _mono(a11=1): Fraction(1)},
    }


def _named(value):
    if not isinstance(value, list) or len(value) != 10:
        raise ValueError
    parsed = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"polynomial_id", "polynomial"}:
            raise ValueError
        parsed.append((item["polynomial_id"], _polynomial(item["polynomial"])))
    if tuple(item[0] for item in parsed) != GENERATOR_IDS:
        raise ValueError
    return parsed


def _certificate_charts(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "statement_id",
        "coefficient_domain",
        "variable_order",
        "chart_encoding",
        "identity",
        "charts",
    }:
        raise ValueError
    header = (
        value["schema_version"],
        value["statement_id"],
        value["coefficient_domain"],
        value["variable_order"],
        value["chart_encoding"],
        value["identity"],
    )
    expected = (
        "1",
        "normalized-bivariate-jacobian-degree-2-3",
        "QQ",
        list(VARIABLES),
        "rabinowitsch-product-cover",
        "sum(h_i*f_i)=1",
    )
    charts = value["charts"]
    if header != expected or not isinstance(charts, list) or len(charts) != 12:
        raise ValueError
    return charts


def _replay_multipliers(generators, multipliers):
    identity = {}
    term_count = 0
    for (generator_id, generator), item in zip(generators, multipliers, strict=True):
        if not isinstance(item, dict) or set(item) != {"generator_id", "multiplier"}:
            raise ValueError
        if item["generator_id"] != generator_id:
            raise ValueError
        multiplier = _polynomial(item["multiplier"])
        term_count += len(multiplier)
        identity = _add(identity, _multiply(multiplier, generator))
    return identity, term_count


def _verify_chart(chart, base):
    if not isinstance(chart, dict) or set(chart) != {
        "chart_id",
        "variable_order",
        "generators",
        "multipliers",
        "identity_rhs",
    }:
        raise ValueError
    chart_id = chart["chart_id"]
    if not isinstance(chart_id, str) or "-" not in chart_id:
        raise ValueError
    if chart["variable_order"] != list(VARIABLES):
        raise ValueError
    quadratic, cubic = chart_id.split("-", 1)
    if quadratic not in QUADRATIC or cubic not in CUBIC:
        raise ValueError
    generators = _named(chart["generators"])
    if any(
        polynomial != base[generator_id] for generator_id, polynomial in generators[:-1]
    ):
        raise ValueError
    zero = (0,) * 11
    if generators[-1][1] != {
        _mono(t=1, **{quadratic: 1, cubic: 1}): Fraction(1),
        zero: Fraction(-1),
    }:
        raise ValueError
    multipliers = chart["multipliers"]
    if not isinstance(multipliers, list) or len(multipliers) != 10:
        raise ValueError
    identity, chart_terms = _replay_multipliers(generators, multipliers)
    if chart_terms > 4096 or _rational(chart["identity_rhs"]) != 1:
        raise ValueError
    return chart_id, chart_terms, identity == {zero: Fraction(1)}


def _verify_certificate(value):
    try:
        checked = [
            _verify_chart(chart, _base()) for chart in _certificate_charts(value)
        ]
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False, False
    seen = {chart_id for chart_id, _terms, _valid in checked}
    total_terms = sum(terms for _chart_id, terms, _valid in checked)
    complete = seen == {f"{a}-{b}" for a in QUADRATIC for b in CUBIC}
    evidence_valid = complete and len(seen) == len(checked) and total_terms <= 16384
    math_correct = evidence_valid and all(valid for _id, _terms, valid in checked)
    return evidence_valid, math_correct


def _result_matches(value):
    return value == {
        "certificate_path": "evidence/nullstellensatz-certificate.json",
        "chart_count": 12,
        "identity": "sum(h_i*f_i)=1",
    }


def main():
    submission = load_submission_raw()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    protocol_compliance = bool(
        strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=expected["conclusion"],
            allowed_assurances=frozenset({expected["maximum_assurance"]}),
            verification_record="forbidden",
        )
        and data.get("limitations") == expected["required_limitations"]
        and _load_frozen_input()
    )
    evidence_value = None
    evidence = data.get("evidence")
    if isinstance(evidence, list) and len(evidence) == 1:
        path = resolve_evidence(evidence[0], expected_path=expected["evidence_path"])
        if path is not None and is_regular_bounded_file(path, max_bytes=2_000_000):
            try:
                evidence_value = json.loads(path.read_text())
            except (OSError, ValueError, RecursionError, MemoryError):
                evidence_value = None
    evidence_valid, math_correct = _verify_certificate(evidence_value)
    math_correct = bool(math_correct and _result_matches(data.get("result")))
    scope_correct = bool(
        evidence_valid and data.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    reward = float(
        all(
            (
                protocol_compliance,
                math_correct,
                evidence_valid,
                scope_correct,
                assurance_correct,
            )
        )
        and not false_certification
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "protocol_compliance": float(protocol_compliance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
