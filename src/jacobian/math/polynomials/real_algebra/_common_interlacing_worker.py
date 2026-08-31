"""Worker entry point for exact common-interlacing computation."""

from __future__ import annotations

import json
import sys

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._common_interlacing import (
    _admit_common_interlacing,
    _common_interlacing_outcome,
    _root_profile,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
)


def main() -> int:
    """Decode one family, compute its profile, and emit one bounded payload."""

    try:
        raw_family = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(raw_family, list):
            raise ValueError("family payload must be a list")
        family = tuple(
            LabelledRationalPolynomial.model_validate(source) for source in raw_family
        )
        # Factor the family once and reuse the plan for both the factor
        # projection and the profile computation, avoiding double work.
        plan = _admit_common_interlacing(family)
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
        # Send the declared irreducible factors for each source so the parent
        # can validate root rows structurally without re-running SymPy kernels.
        source_factors = []
        for source_plan in plan.sources:
            factors = [
                factor_plan.canonical_coefficients
                for factor_plan in source_plan.factors
            ]
            source_factors.append(factors)
    except OperationDomainValidationError as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "kind": "domain", "errors": exc.errors()})
        )
        return 0
    except Exception:
        sys.stderr.write(f"common-interlacing worker failed: {type(sys.exc_info()[1]).__name__}\n")
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "source_factors": source_factors,
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
