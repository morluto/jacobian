"""Standard-library replay for exact finite-probability results.

This independent checker imports neither Python-FLINT nor producer modules.
Only passive, artifact-bound JSON values cross the checker boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request as _bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_META = {
    "exactness": "EXACT_RATIONAL",
    "determinism": "DETERMINISTIC",
    "backend": "python-flint",
    "backend_version": "0.9.0",
    "verification": "UNVERIFIED",
}
_GAUSSIAN_META = {
    "gaussian_model": "INDEPENDENT_STANDARD_REAL",
    "completeness": "COMPLETE_BOUNDED_EXPANSION",
    "exactness": "EXACT_COMPLEX_RATIONAL",
    "determinism": "DETERMINISTIC",
    "backend": "python-flint",
    "backend_version": "0.9.0",
    "verification": "UNVERIFIED",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(operation_id: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": f"independent standard-library Fraction replay accepted {operation_id}",
    }


def _integer(value: object) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("integer is not canonical")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("integer is not canonical")
    return parsed


def _fraction(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("rational is not reduced")
    return result


def _complex_fraction(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict) or set(value) != {"real", "imaginary"}:
        raise ValueError("exact complex rational is malformed")
    return _fraction(value["real"]), _fraction(value["imaginary"])


def _complex_multiply(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _gaussian_univariate_moment(exponent: int) -> int:
    if exponent % 2:
        return 0
    result = 1
    for factor in range(1, exponent, 2):
        result *= factor
    return result


def _atom(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict) or set(value) != {"value", "probability"}:
        raise ValueError("finite-distribution atom is malformed")
    probability = _fraction(value["probability"])
    if probability < 0:
        raise ValueError("finite-distribution probability is negative")
    return _fraction(value["value"]), probability


def _atoms(value: object, *, canonical: bool) -> list[tuple[Fraction, Fraction]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 256:
        raise ValueError("finite distribution is malformed")
    atoms = [_atom(item) for item in value]
    support = [atom[0] for atom in atoms]
    if len(support) != len(set(support)):
        raise ValueError("finite distribution repeats a support value")
    if canonical and support != sorted(support):
        raise ValueError("finite distribution is not canonical")
    if sum((atom[1] for atom in atoms), start=Fraction()) != 1:
        raise ValueError("finite distribution is not normalized")
    return atoms


def _distribution(value: object) -> list[tuple[Fraction, Fraction]]:
    if not isinstance(value, dict) or set(value) != {"atoms"}:
        raise ValueError("finite distribution wrapper is malformed")
    return _atoms(value["atoms"], canonical=True)


def _metadata(result: dict[str, Any], fields: set[str]) -> bool:
    return set(result) == fields | set(_META) and all(
        result.get(key) == value for key, value in _META.items()
    )


def _event(
    source: dict[str, Any],
) -> tuple[list[tuple[Fraction, Fraction]], list[Fraction]]:
    if set(source) != {"distribution", "event_values"}:
        raise ValueError("finite event source is malformed")
    atoms = _distribution(source["distribution"])
    raw_event = source["event_values"]
    if not isinstance(raw_event, list) or len(raw_event) > 256:
        raise ValueError("finite event is malformed")
    event = [_fraction(value) for value in raw_event]
    if event != sorted(set(event)) or not set(event).issubset(
        {value for value, _ in atoms}
    ):
        raise ValueError("finite event is not a canonical support subset")
    return atoms, event


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    try:
        source, result = _bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject("declared result does not match independent Fraction replay")
        return _accept(operation_id)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _replay_raw_moment(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"atoms", "order"} or not _metadata(
        result, {"order", "moment", "contributions"}
    ):
        return False
    atoms = _atoms(source["atoms"], canonical=False)
    order = source["order"]
    if type(order) is not int or not 0 <= order <= 128 or result["order"] != order:
        return False
    contributions = result["contributions"]
    if not isinstance(contributions, list) or len(contributions) != len(atoms):
        return False
    total = Fraction()
    for atom, contribution in zip(atoms, contributions, strict=True):
        if not isinstance(contribution, dict) or set(contribution) != {
            "value",
            "probability",
            "powered_value",
            "contribution",
        }:
            return False
        value, probability = atom
        powered = value**order
        term = probability * powered
        if (
            _fraction(contribution["value"]) != value
            or _fraction(contribution["probability"]) != probability
            or _fraction(contribution["powered_value"]) != powered
            or _fraction(contribution["contribution"]) != term
        ):
            return False
        total += term
    return _fraction(result["moment"]) == total


def _replay_event_probability(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if not _metadata(result, {"event_probability", "selected_atoms"}):
        return False
    atoms, event = _event(source)
    selected = [atom for atom in atoms if atom[0] in set(event)]
    return (
        isinstance(result["selected_atoms"], list)
        and [_atom(item) for item in result["selected_atoms"]] == selected
        and _fraction(result["event_probability"])
        == sum((probability for _, probability in selected), start=Fraction())
    )


def _replay_condition(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if not _metadata(result, {"event_probability", "distribution", "contributions"}):
        return False
    atoms, event = _event(source)
    selected = [atom for atom in atoms if atom[0] in set(event)]
    mass = sum((probability for _, probability in selected), start=Fraction())
    if mass <= 0 or _fraction(result["event_probability"]) != mass:
        return False
    expected = [(value, probability / mass) for value, probability in selected]
    if _distribution(result["distribution"]) != expected:
        return False
    contributions = result["contributions"]
    if not isinstance(contributions, list) or len(contributions) != len(selected):
        return False
    for item, ((value, probability), (_, conditioned)) in zip(
        contributions, zip(selected, expected, strict=True), strict=True
    ):
        if not isinstance(item, dict) or set(item) != {
            "value",
            "source_probability",
            "conditioned_probability",
        }:
            return False
        if (
            _fraction(item["value"]) != value
            or _fraction(item["source_probability"]) != probability
            or _fraction(item["conditioned_probability"]) != conditioned
        ):
            return False
    return True


def _replay_pushforward(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"distribution", "mapping"} or not _metadata(
        result, {"distribution", "contributions"}
    ):
        return False
    atoms = _distribution(source["distribution"])
    mapping = source["mapping"]
    contributions = result["contributions"]
    if (
        not isinstance(mapping, list)
        or not isinstance(contributions, list)
        or len(mapping) != len(atoms)
        or len(contributions) != len(atoms)
    ):
        return False
    aggregated: dict[Fraction, Fraction] = {}
    for atom, item, contribution in zip(atoms, mapping, contributions, strict=True):
        if not isinstance(item, dict) or set(item) != {"source", "target"}:
            return False
        if not isinstance(contribution, dict) or set(contribution) != {
            "source",
            "target",
            "probability",
        }:
            return False
        value, probability = atom
        source_value, target = _fraction(item["source"]), _fraction(item["target"])
        if source_value != value or (
            _fraction(contribution["source"]),
            _fraction(contribution["target"]),
            _fraction(contribution["probability"]),
        ) != (value, target, probability):
            return False
        aggregated[target] = aggregated.get(target, Fraction()) + probability
    return _distribution(result["distribution"]) == sorted(aggregated.items())


def _replay_convolution(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"left", "right"} or not _metadata(
        result, {"distribution", "contributions", "independence"}
    ):
        return False
    if result["independence"] != "PRODUCT_MEASURE":
        return False
    left, right = _distribution(source["left"]), _distribution(source["right"])
    contributions = result["contributions"]
    if not isinstance(contributions, list) or len(contributions) != len(left) * len(
        right
    ):
        return False
    expected: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
    aggregated: dict[Fraction, Fraction] = {}
    for left_value, left_probability in left:
        for right_value, right_probability in right:
            value = left_value + right_value
            probability = left_probability * right_probability
            expected.append((left_value, right_value, value, probability))
            aggregated[value] = aggregated.get(value, Fraction()) + probability
    actual = []
    for item in contributions:
        if not isinstance(item, dict) or set(item) != {
            "left_value",
            "right_value",
            "sum_value",
            "probability",
        }:
            return False
        actual.append(
            (
                _fraction(item["left_value"]),
                _fraction(item["right_value"]),
                _fraction(item["sum_value"]),
                _fraction(item["probability"]),
            )
        )
    return actual == expected and _distribution(result["distribution"]) == sorted(
        aggregated.items()
    )


def _replay_gaussian_polynomial_moment(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if set(source) != {"polynomial", "order"} or set(result) != {
        "order",
        "moment",
        "expansion_path_count",
        "expanded_monomial_count",
        "contractions",
        *_GAUSSIAN_META,
    }:
        return False
    if any(result.get(key) != value for key, value in _GAUSSIAN_META.items()):
        return False
    order = source["order"]
    if type(order) is not int or not 0 <= order <= 16 or result["order"] != order:
        return False
    polynomial = source["polynomial"]
    if not isinstance(polynomial, dict) or set(polynomial) != {
        "variable_count",
        "terms",
    }:
        return False
    variable_count = polynomial["variable_count"]
    terms = polynomial["terms"]
    if (
        type(variable_count) is not int
        or not 1 <= variable_count <= 8
        or not isinstance(terms, list)
        or not 1 <= len(terms) <= 16
        or len(terms) ** order > 4096
    ):
        return False
    base: list[tuple[tuple[int, ...], tuple[Fraction, Fraction]]] = []
    previous_exponents: tuple[int, ...] | None = None
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            return False
        raw_exponents = term["exponents"]
        if (
            not isinstance(raw_exponents, list)
            or len(raw_exponents) != variable_count
            or any(
                type(exponent) is not int or exponent < 0 for exponent in raw_exponents
            )
            or sum(raw_exponents) > 8
        ):
            return False
        exponents = tuple(raw_exponents)
        if previous_exponents is not None and exponents <= previous_exponents:
            return False
        previous_exponents = exponents
        coefficient = _complex_fraction(term["coefficient"])
        if coefficient == (Fraction(), Fraction()):
            return False
        base.append((exponents, coefficient))

    expanded: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {
        (0,) * variable_count: (Fraction(1), Fraction())
    }
    for _ in range(order):
        next_expanded: dict[tuple[int, ...], tuple[Fraction, Fraction]] = {}
        for left_exponents, left_coefficient in sorted(expanded.items()):
            for right_exponents, right_coefficient in base:
                exponents = tuple(
                    left + right
                    for left, right in zip(
                        left_exponents,
                        right_exponents,
                        strict=True,
                    )
                )
                product = _complex_multiply(left_coefficient, right_coefficient)
                previous = next_expanded.get(
                    exponents,
                    (Fraction(), Fraction()),
                )
                next_expanded[exponents] = (
                    previous[0] + product[0],
                    previous[1] + product[1],
                )
        expanded = {
            exponents: coefficient
            for exponents, coefficient in next_expanded.items()
            if coefficient != (Fraction(), Fraction())
        }

    contractions = result["contractions"]
    if (
        result["expansion_path_count"] != len(base) ** order
        or result["expanded_monomial_count"] != len(expanded)
        or not isinstance(contractions, list)
        or len(contractions) != len(expanded)
    ):
        return False
    total = (Fraction(), Fraction())
    for item, (exponents, coefficient) in zip(
        contractions,
        sorted(expanded.items()),
        strict=True,
    ):
        if not isinstance(item, dict) or set(item) != {
            "exponents",
            "expanded_coefficient",
            "variable_moment_factors",
            "gaussian_moment_factor",
            "contribution",
        }:
            return False
        factors = tuple(_gaussian_univariate_moment(exponent) for exponent in exponents)
        gaussian_factor = 1
        for factor in factors:
            gaussian_factor *= factor
        contribution = (
            coefficient[0] * gaussian_factor,
            coefficient[1] * gaussian_factor,
        )
        raw_factors = item["variable_moment_factors"]
        if (
            item["exponents"] != list(exponents)
            or _complex_fraction(item["expanded_coefficient"]) != coefficient
            or not isinstance(raw_factors, list)
            or tuple(_integer(value) for value in raw_factors) != factors
            or _integer(item["gaussian_moment_factor"]) != gaussian_factor
            or _complex_fraction(item["contribution"]) != contribution
        ):
            return False
        total = (total[0] + contribution[0], total[1] + contribution[1])
    return _complex_fraction(result["moment"]) == total


def check_finite_raw_moment(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.finite_distribution.raw_moment.compute",
        witness_format="probability.finite-raw-moment.fraction-replay",
        replay=_replay_raw_moment,
    )


def check_finite_event_probability(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.finite_distribution.event_probability.compute",
        witness_format="probability.finite-event.fraction-replay",
        replay=_replay_event_probability,
    )


def check_finite_condition(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.finite_distribution.condition.compute",
        witness_format="probability.finite-condition.fraction-replay",
        replay=_replay_condition,
    )


def check_finite_pushforward(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.finite_distribution.pushforward.compute",
        witness_format="probability.finite-pushforward.fraction-replay",
        replay=_replay_pushforward,
    )


def check_finite_convolution(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.finite_distribution.convolution.compute",
        witness_format="probability.finite-convolution.fraction-replay",
        replay=_replay_convolution,
    )


def check_gaussian_polynomial_moment(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="probability.gaussian_polynomial.moment.compute",
        witness_format="probability.gaussian-polynomial-moment.fraction-replay",
        replay=_replay_gaussian_polynomial_moment,
    )


__all__ = [
    "check_finite_condition",
    "check_finite_convolution",
    "check_finite_event_probability",
    "check_finite_pushforward",
    "check_finite_raw_moment",
    "check_gaussian_polynomial_moment",
]
