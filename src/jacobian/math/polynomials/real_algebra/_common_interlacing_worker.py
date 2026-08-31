"""Worker entry point for exact common-interlacing computation."""

from __future__ import annotations

import json
import sys

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.real_algebra._common_interlacing import (
    _common_interlacing_profile_in_process,
)
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
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
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )

        from jacobian.math.polynomials.real_algebra._common_interlacing import (
            _admit_common_interlacing,
        )

        plan = _admit_common_interlacing(family)
        result = _common_interlacing_profile_in_process(family)
        # Send the declared irreducible factors for each source so the parent
        # can validate root rows structurally without re-running SymPy kernels.
        source_factors = []
        for source_index, source_plan in enumerate(plan.sources):
            factors = []
            for factor_plan in source_plan.factors:
                factors.append(factor_plan.canonical_coefficients)
            source_factors.append(factors)
    except OperationDomainValidationError as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "kind": "domain", "errors": exc.errors()})
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"common-interlacing worker failed: {type(exc).__name__}\n")
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
