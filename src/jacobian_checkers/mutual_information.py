"""Independent standard-library replay for finite-table mutual information."""

from __future__ import annotations

import re
from fractions import Fraction
from math import lcm
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_SCALE_BITS_BOUND = 1_024
_POWER_COST_BITS_BOUND = 32_768
_MAX_ROWS = 16
_MAX_COLUMNS = 16
_MAX_CELLS = 64
_MAX_INPUT_DIGITS = 256


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _integer(value: object, *, max_digits: int | None = None) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("integer is not canonical")
    digits = value[1:] if value.startswith("-") else value
    if max_digits is not None and len(digits) > max_digits:
        raise ValueError("integer exceeds checker scope")
    parsed = parse_canonical_integer(value)
    if format_canonical_integer(parsed) != value:
        raise ValueError("integer is not canonical")
    return parsed


def _q(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"], max_digits=_MAX_INPUT_DIGITS)
    denominator = _integer(value["den"], max_digits=_MAX_INPUT_DIGITS)
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError("rational is not reduced")
    return result


def _wire(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _small_prime_factorization(value: int) -> dict[int, int]:
    remaining = value
    factors: dict[int, int] = {}
    prime = 2
    while prime * prime <= remaining:
        while remaining % prime == 0:
            factors[prime] = factors.get(prime, 0) + 1
            remaining //= prime
        prime += 1
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def _valuations(value: int, primes: tuple[int, ...]) -> tuple[dict[int, int], int]:
    remaining = value
    exponents: dict[int, int] = {}
    for prime in primes:
        exponent = 0
        while remaining > 1 and remaining % prime == 0:
            remaining //= prime
            exponent += 1
        exponents[prime] = exponent
    return exponents, remaining


def _rational_base_exponent(value: Fraction, base: int) -> Fraction | None:
    base_factors = _small_prime_factorization(base)
    primes = tuple(base_factors)
    numerator_exponents, numerator_remainder = _valuations(value.numerator, primes)
    denominator_exponents, denominator_remainder = _valuations(
        value.denominator,
        primes,
    )
    if numerator_remainder != 1 or denominator_remainder != 1:
        return None
    exponent: Fraction | None = None
    for prime, base_exponent in base_factors.items():
        current = Fraction(
            numerator_exponents[prime] - denominator_exponents[prime],
            base_exponent,
        )
        if exponent is None:
            exponent = current
        elif current != exponent:
            return None
    return exponent if exponent is not None else Fraction()


def _table(source: dict[str, Any]) -> tuple[list[list[Fraction]], int]:
    if set(source) != {"row_labels", "column_labels", "probabilities", "log_base"}:
        raise ValueError("finite joint-table source has an invalid shape")
    rows = source["row_labels"]
    columns = source["column_labels"]
    raw_table = source["probabilities"]
    base = source["log_base"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= _MAX_ROWS
        or any(not isinstance(label, str) or not label for label in rows)
        or len(set(rows)) != len(rows)
        or not isinstance(columns, list)
        or not 1 <= len(columns) <= _MAX_COLUMNS
        or any(not isinstance(label, str) or not label for label in columns)
        or len(set(columns)) != len(columns)
        or len(rows) * len(columns) > _MAX_CELLS
        or type(base) is not int
        or not 2 <= base <= 36
        or not isinstance(raw_table, list)
        or len(raw_table) != len(rows)
        or len(raw_table) > _MAX_ROWS
    ):
        raise ValueError("finite joint-table source is malformed")
    table: list[list[Fraction]] = []
    parsed_cells = 0
    for row in raw_table:
        if (
            not isinstance(row, list)
            or len(row) != len(columns)
            or len(row) > _MAX_COLUMNS
        ):
            raise ValueError("finite joint table is not rectangular")
        parsed_cells += len(row)
        if parsed_cells > _MAX_CELLS:
            raise ValueError("finite joint table exceeds checker scope")
        parsed = [_q(value) for value in row]
        if any(value < 0 for value in parsed):
            raise ValueError("finite joint table has negative mass")
        table.append(parsed)
    if sum((sum(row, Fraction()) for row in table), Fraction()) != 1:
        raise ValueError("finite joint table is not normalized")
    return table, base


def _bounded_candidate(candidate: object) -> dict[str, Any]:
    if not isinstance(candidate, dict) or set(candidate) != {
        "row_marginals",
        "column_marginals",
        "positive_support",
        "log_base",
        "log_product_certificate",
        "exact_value",
        "sign",
        "zero_cell_convention",
    }:
        raise ValueError("mutual-information candidate has an invalid shape")
    row_marginals = candidate["row_marginals"]
    column_marginals = candidate["column_marginals"]
    support = candidate["positive_support"]
    if (
        not isinstance(row_marginals, list)
        or not 1 <= len(row_marginals) <= _MAX_ROWS
        or not isinstance(column_marginals, list)
        or not 1 <= len(column_marginals) <= _MAX_COLUMNS
        or not isinstance(support, list)
        or not 1 <= len(support) <= _MAX_CELLS
    ):
        raise ValueError("mutual-information candidate exceeds checker scope")
    return candidate


def _replay_support(
    table: list[list[Fraction]],
) -> tuple[list[dict[str, object]], list[tuple[Fraction, Fraction]]]:
    row_marginals = [sum(row, Fraction()) for row in table]
    column_marginals = [
        sum((table[row][column] for row in range(len(table))), Fraction())
        for column in range(len(table[0]))
    ]
    support: list[dict[str, object]] = []
    weighted_ratios: list[tuple[Fraction, Fraction]] = []
    for row_index, row in enumerate(table):
        for column_index, probability in enumerate(row):
            if probability == 0:
                continue
            denominator = row_marginals[row_index] * column_marginals[column_index]
            if denominator == 0:
                raise ValueError("positive joint mass has zero marginal support")
            ratio = probability / denominator
            weighted_ratios.append((probability, ratio))
            support.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "probability": _wire(probability),
                    "row_marginal": _wire(row_marginals[row_index]),
                    "column_marginal": _wire(column_marginals[column_index]),
                    "likelihood_ratio": _wire(ratio),
                }
            )
    return support, weighted_ratios


def _replay_product(
    scale: int,
    weighted_ratios: list[tuple[Fraction, Fraction]],
) -> Fraction:
    if scale.bit_length() > _SCALE_BITS_BOUND:
        raise ValueError("mutual-information scale exceeds checker scope")
    power_cost = 0
    product = Fraction(1)
    for probability, ratio in weighted_ratios:
        exponent = scale * probability.numerator // probability.denominator
        if ratio == 1:
            continue
        power_cost += exponent * (
            ratio.numerator.bit_length() + ratio.denominator.bit_length()
        )
        if power_cost > _POWER_COST_BITS_BOUND:
            raise ValueError("mutual-information product exceeds checker scope")
        product *= ratio**exponent
    return product


def _expected(source: dict[str, Any]) -> dict[str, object]:
    table, base = _table(source)
    row_marginals = [sum(row, Fraction()) for row in table]
    column_marginals = [
        sum((table[row][column] for row in range(len(table))), Fraction())
        for column in range(len(table[0]))
    ]
    support, weighted_ratios = _replay_support(table)
    scale = lcm(*(probability.denominator for probability, _ in weighted_ratios))
    product = _replay_product(scale, weighted_ratios)
    if product < 1:
        raise ValueError("mutual-information product contradicts nonnegativity")
    base_exponent = _rational_base_exponent(product, base)
    exact_value = None
    if base_exponent is not None:
        exact_value = _wire(base_exponent / scale)
    return {
        "row_marginals": [_wire(value) for value in row_marginals],
        "column_marginals": [_wire(value) for value in column_marginals],
        "positive_support": support,
        "log_base": base,
        "log_product_certificate": {
            "scale": format_canonical_integer(scale),
            "product": _wire(product),
            "identity": "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT",
        },
        "exact_value": exact_value,
        "sign": "ZERO" if product == 1 else "POSITIVE",
        "zero_cell_convention": "ZERO_MASS_TERMS_OMITTED",
    }


def check_finite_joint_mutual_information(request: object) -> dict[str, Any]:
    try:
        source, candidate = bound_request(
            request,
            operation_id="probability.joint.mutual_information.compute",
            witness_format="probability.finite-joint-mutual-information.fraction-replay",
        )
        if _bounded_candidate(candidate) != _expected(source):
            return _reject("candidate does not match independent exact replay")
        return _accept("independent Fraction logarithmic-product replay accepted")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_finite_joint_mutual_information"]
