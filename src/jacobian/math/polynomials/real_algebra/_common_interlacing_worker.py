"""Worker entry point for exact common-interlacing computation."""

from __future__ import annotations

import hashlib
import json
import sys
from math import gcd

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._common_interlacing import (
    _admit_common_interlacing,
    _common_interlacing_outcome,
    _PrimitiveSourcePlan,
    _root_profile,
    _SourcePlan,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    MAX_COMMON_INTERLACING_SOURCE_TERMS,
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
    SourceRootProfile,
)


def _primitive_source_from_worker(value: object) -> _PrimitiveSourcePlan:
    """Decode the parent's admitted primitive-source projection."""

    if not isinstance(value, dict):
        raise ValueError("worker admitted source projection is malformed")
    raw_coefficients = value.get("coefficients")
    degree = value.get("degree")
    height_digits = value.get("height_digits")
    term_count = value.get("term_count")
    if (
        not isinstance(raw_coefficients, list)
        or type(degree) is not int
        or type(height_digits) is not int
        or type(term_count) is not int
        or not 1 <= degree <= MAX_COMMON_INTERLACING_SOURCE_DEGREE
        or not 1 <= term_count <= MAX_COMMON_INTERLACING_SOURCE_TERMS
        or len(raw_coefficients) != degree + 1
        or height_digits < 1
    ):
        raise ValueError("worker admitted source projection is malformed")
    if any(not isinstance(coefficient, str) for coefficient in raw_coefficients):
        raise ValueError("worker admitted source projection is malformed")
    try:
        coefficients = tuple(
            parse_canonical_integer(coefficient) for coefficient in raw_coefficients
        )
    except (TypeError, ValueError):
        raise ValueError("worker admitted source projection is malformed") from None
    if any(
        format_canonical_integer(coefficient) != raw
        for coefficient, raw in zip(coefficients, raw_coefficients, strict=True)
    ):
        raise ValueError("worker admitted source projection is malformed")
    if coefficients[0] <= 0:
        raise ValueError("worker admitted source projection is malformed")
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    if content != 1:
        raise ValueError("worker admitted source projection is malformed")
    if (
        max(
            len(format_canonical_integer(coefficient).lstrip("-"))
            for coefficient in coefficients
        )
        != height_digits
    ):
        raise ValueError("worker admitted source projection is malformed")
    return _PrimitiveSourcePlan(
        coefficients=coefficients,
        degree=degree,
        height_digits=height_digits,
        term_count=term_count,
    )


def _factor_root_counts(
    source_plan: _SourcePlan,
    profile: SourceRootProfile,
) -> list[int]:
    """Count factor rows from the root profile isolated on the source axis."""

    factor_indices = {
        factor.canonical_coefficients: index
        for index, factor in enumerate(source_plan.factors)
    }
    counts = [0] * len(source_plan.factors)
    for root in profile.roots:
        factor_index = factor_indices.get(tuple(root.value.polynomial))
        if factor_index is None:  # pragma: no cover - worker invariant
            raise RuntimeError("worker root profile contains an unknown factor")
        counts[factor_index] += 1
    return counts


def main() -> int:
    """Decode one family, compute its profile, and emit one bounded payload."""

    try:
        input_bytes = sys.stdin.buffer.read()
        raw_payload = json.loads(input_bytes.decode("utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("worker payload must be an object")
        raw_family = raw_payload.get("family")
        raw_primitives = raw_payload.get("primitive_sources")
        if not isinstance(raw_family, list) or not isinstance(raw_primitives, list):
            raise ValueError("worker payload is missing admitted sources")
        family = tuple(
            LabelledRationalPolynomial.model_validate(source) for source in raw_family
        )
        primitive_sources = tuple(
            _primitive_source_from_worker(item) for item in raw_primitives
        )
        if len(primitive_sources) != len(family):
            raise ValueError("worker admitted source projection is malformed")
        # Factor the family once and reuse the plan for both the factor
        # projection and the profile computation, avoiding double work.
        plan = _admit_common_interlacing(family, primitive_sources=primitive_sources)
        root_profiles = tuple(
            _root_profile(source_index, source)
            for source_index, source in enumerate(plan.sources)
        )
        outcome = _common_interlacing_outcome(root_profiles, plan.common_degree)
        result = CommonInterlacingProfile._from_kernel(
            family=plan.family,
            root_profiles=root_profiles,
            outcome=outcome,
        )
        # Send the declared irreducible factors and their real-root counts.
        # Root counts are derived from the already-computed root profiles
        # (one row per real root of each factor) rather than calling
        # intervals() again for each factor.
        source_factors = []
        source_factor_root_counts = []
        for source_index, source_plan in enumerate(plan.sources):
            factors = [
                [
                    list(factor_plan.canonical_coefficients),
                    factor_plan.multiplicity,
                ]
                for factor_plan in source_plan.factors
            ]
            root_counts = _factor_root_counts(
                source_plan,
                root_profiles[source_index],
            )
            source_factors.append(factors)
            source_factor_root_counts.append(root_counts)
    except OperationDomainValidationError as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "kind": "domain", "errors": exc.errors()})
        )
        return 0
    except Exception:
        sys.stderr.write(
            f"common-interlacing worker failed: {type(sys.exc_info()[1]).__name__}\n"
        )
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                "source_factors": source_factors,
                "source_factor_root_counts": source_factor_root_counts,
                "root_profiles": [
                    profile.model_dump(mode="json") for profile in result.root_profiles
                ],
                "outcome": result.outcome.model_dump(mode="json"),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
