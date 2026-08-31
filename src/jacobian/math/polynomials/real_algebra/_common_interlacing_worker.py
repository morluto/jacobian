"""Worker entry point for exact common-interlacing computation."""

from __future__ import annotations

import hashlib
import json
import sys

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._common_interlacing import (
    _admit_common_interlacing,
    _common_interlacing_outcome,
    _PrimitiveSourcePlan,
    _root_profile,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
)


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
            _PrimitiveSourcePlan(
                coefficients=tuple(
                    parse_canonical_integer(value) for value in item["coefficients"]
                ),
                degree=int(item["degree"]),
                height_digits=int(item["height_digits"]),
                term_count=int(item["term_count"]),
            )
            for item in raw_primitives
            if isinstance(item, dict)
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
        for _source_index, source_plan in enumerate(plan.sources):
            factors = [
                [
                    list(factor_plan.canonical_coefficients),
                    factor_plan.multiplicity,
                ]
                for factor_plan in source_plan.factors
            ]
            # Authenticate the root-count projection from the source-bound
            # irreducible factors inside the killable worker.  The parent only
            # checks that the projected rows agree with these counts; it does
            # not replay root isolation after the worker returns.
            root_counts = [
                len(factor_plan.polynomial.intervals())
                for factor_plan in source_plan.factors
            ]
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
