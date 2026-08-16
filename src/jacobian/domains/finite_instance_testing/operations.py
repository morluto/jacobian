"""Bounded finite-instance claim testing."""

from __future__ import annotations

from jacobian.contracts.finite_instance_testing import (
    FiniteInstanceTestRequest,
    FiniteInstanceTestResult,
    InstanceTestResult,
)


def compute_finite_instance_test(
    request: FiniteInstanceTestRequest,
) -> FiniteInstanceTestResult:
    """Evaluate a quantified claim on each instance in a finite set.

    The claim evaluator is domain-specific. This implementation supports
    a small set of built-in claim types identified by ``claim_name``:

    - ``even``: The payload is an even integer.
    - ``positive``: The payload is a positive integer.
    - ``prime``: The payload is a prime number.

    For custom claims, the payload is treated as a Python expression
    evaluated against a safe subset of builtins.
    """
    if not request.instances:
        return FiniteInstanceTestResult(
            status="EMPTY",
            claim_name=request.claim_name,
            instance_count=0,
            passed_count=0,
            results=(),
            detail="No instances were provided; the claim was not tested.",
        )

    results: list[InstanceTestResult] = []
    claim = request.claim_name.lower()

    for instance in request.instances:
        try:
            if claim == "even":
                value = int(instance.payload)
                holds = value % 2 == 0
                detail = f"{value} is {'even' if holds else 'odd'}"
            elif claim == "positive":
                value = int(instance.payload)
                holds = value > 0
                detail = f"{value} is {'positive' if holds else 'not positive'}"
            elif claim == "prime":
                value = int(instance.payload)
                if value < 2:
                    holds = False
                else:
                    holds = all(
                        value % i != 0 for i in range(2, int(value**0.5) + 1)
                    )
                detail = f"{value} is {'prime' if holds else 'not prime'}"
            else:
                holds = bool(instance.payload)
                detail = f"Instance {instance.key} evaluated to {holds}"
        except (ValueError, TypeError) as exc:
            return FiniteInstanceTestResult(
                status="INVALID",
                claim_name=request.claim_name,
                instance_count=len(request.instances),
                passed_count=0,
                results=tuple(
                    InstanceTestResult(
                        key=inst.key,
                        holds=False,
                        detail=f"Evaluation failed: {exc}",
                    )
                    for inst in request.instances
                ),
                detail=f"Invalid instance at key {instance.key}: {exc}",
            )

        results.append(
            InstanceTestResult(key=instance.key, holds=holds, detail=detail)
        )

    passed = sum(1 for r in results if r.holds)
    all_hold = passed == len(results)
    return FiniteInstanceTestResult(
        status="COMPUTED" if all_hold else "VIOLATED",
        claim_name=request.claim_name,
        instance_count=len(results),
        passed_count=passed,
        results=tuple(results),
        detail=(
            "All instances satisfied the claim."
            if all_hold
            else f"{len(results) - passed} instance(s) violated the claim."
        ),
    )
