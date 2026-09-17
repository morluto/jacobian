"""Public operation declarations for square-free affine-form local arithmetic."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    LocalAdmissibilityRequest,
    LocalAdmissibilityResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._euler_product import (
    SquarefreeEulerProductRequest,
    SquarefreeEulerProductResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    IntervalCountRequest,
    IntervalCountResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._local_factor import (
    SquarefreeLocalFactorRequest,
    SquarefreeLocalFactorResult,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    euler_product as native_euler_product,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    interval_count as native_interval_count,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    local_admissibility as native_local_admissibility,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    local_factor as native_local_factor,
)


def compute_local_factor(
    request: SquarefreeLocalFactorRequest,
) -> SquarefreeLocalFactorResult:
    return native_local_factor(request.source, request.prime)


def compute_euler_product(
    request: SquarefreeEulerProductRequest,
) -> SquarefreeEulerProductResult:
    return native_euler_product(request.source, request.primes)


def compute_local_admissibility(
    request: LocalAdmissibilityRequest,
) -> LocalAdmissibilityResult:
    return native_local_admissibility(request.source)


def compute_interval_count(
    request: IntervalCountRequest,
) -> IntervalCountResult:
    return native_interval_count(
        request.source, request.lower, request.upper, request.include_ledger
    )


_SINGLE_FORM_SOURCE = {
    "forms": [
        {"form_id": "n", "coefficient": "1", "constant": "0"},
    ]
}

_PAIR_FORM_SOURCE = {
    "forms": [
        {"form_id": "n", "coefficient": "1", "constant": "0"},
        {"form_id": "n_plus_1", "coefficient": "1", "constant": "1"},
    ]
}

_TWIN_FORM_SOURCE = {
    "forms": [
        {"form_id": "n", "coefficient": "1", "constant": "0"},
        {"form_id": "n_plus_2", "coefficient": "1", "constant": "2"},
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_theory.squarefree_affine_forms.local_factor.compute",
        title="Compute one square-free affine local factor",
        description=(
            "For a bounded family of integer affine forms L_j(n)=a_j*n+b_j and "
            "one prime p, return the exact local square-free density factor "
            "valid_count/p^2: the residues r modulo p^2 where some p^2 divides "
            "L_j(r) are bad, computed as one closed-form coset per form and "
            "replayed as a complete bad-residue ledger with overlap counts. "
            "A positive factor is one exact finite local computation; it is "
            "not a simultaneous square-free density or admissibility theorem. "
            "Constant forms are admitted; a form vanishing identically modulo "
            "p^2 covers every residue and the factor is zero."
        ),
        request_type=SquarefreeLocalFactorRequest,
        result_type=SquarefreeLocalFactorResult,
        run=compute_local_factor,
        tags=("number-theory", "squarefree", "local-density", "exact"),
        discovery_terms=(
            "square-free local density",
            "p-squared residue obstruction",
            "affine form local factor",
            "square-free sieve ledger",
            "bad residue modulo prime square",
        ),
        examples=(
            OperationExample(
                name="single_form_n_mod_4",
                description="Compute the exact local square-free factor of the form n "
                "at p=2; the bad residue set is exactly {0} and the factor 3/4.",
                input={"source": _SINGLE_FORM_SOURCE, "prime": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.squarefree_affine_forms.euler_product.compute",
        title="Compute a finite square-free Euler-product prefix",
        description=(
            "For a bounded family of integer affine forms and one bounded "
            "explicit prime set, return each prime's exact local square-free "
            "factor and their exact finite product as one reduced rational. "
            "This is a finite prefix product over exactly the supplied "
            "primes; it is not an infinite Euler product, a singular series, "
            "a density theorem, or an admissibility proof. Positivity over a "
            "caller-selected prime list establishes no global claim."
        ),
        request_type=SquarefreeEulerProductRequest,
        result_type=SquarefreeEulerProductResult,
        run=compute_euler_product,
        tags=("number-theory", "squarefree", "finite-product", "exact"),
        discovery_terms=(
            "finite Euler product prefix",
            "square-free local density",
            "local factor product",
            "simultaneous square-free heuristic",
        ),
        examples=(
            OperationExample(
                name="consecutive_pair_prefix_2_3",
                description="Compute the finite local-factor product of n and n+1 over "
                "p=2 and p=3; the primes must be distinct and increasing.",
                input={"source": _PAIR_FORM_SOURCE, "primes": [2, 3]},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.squarefree_affine_forms.local_admissibility.decide",
        title="Decide local square admissibility with a finite cutoff",
        description=(
            "Decide whether an affine-form family has a local square "
            "obstruction at any prime: check every prime through an exact "
            "cutoff derived from the family with per-prime local-factor "
            "rows, and cover larger primes by a replayed large-prime "
            "positivity bound. LOCALLY_ADMISSIBLE and LOCALLY_OBSTRUCTED "
            "decide local obstructions only, never existence, density, or "
            "infinitude."
        ),
        request_type=LocalAdmissibilityRequest,
        result_type=LocalAdmissibilityResult,
        run=compute_local_admissibility,
        tags=("number-theory", "squarefree", "admissibility", "exact"),
        discovery_terms=(
            "local admissibility",
            "square-free local obstruction",
            "finite cutoff proof",
            "large prime positivity",
        ),
        examples=(
            OperationExample(
                name="twin_like_pair_admissible",
                description="The forms n and n+2 admit a valid residue at "
                "every prime: the exact cutoff is vacuous and the "
                "large-prime bound covers the rest.",
                input={"source": _TWIN_FORM_SOURCE},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.squarefree_affine_forms.interval_count.compute",
        title="Count simultaneous square-free values over an interval",
        description=(
            "Count integers n in [lower, upper] for which every affine form "
            "value is square-free, by an exact prime-power sieve recording "
            "the least square divisor of each rejected point. The scalar "
            "count is always returned; the matching list and per-rejection "
            "obstructions are returned only when the ledger is requested."
        ),
        request_type=IntervalCountRequest,
        result_type=IntervalCountResult,
        run=compute_interval_count,
        tags=("number-theory", "squarefree", "interval", "exact"),
        discovery_terms=(
            "simultaneous square-free values",
            "square-free interval count",
            "square divisor sieve",
        ),
        examples=(
            OperationExample(
                name="integers_one_to_twenty_squarefree_count",
                description="Count n in [1, 20] with n square-free, with the "
                "matching list and first square divisors.",
                input={
                    "source": _SINGLE_FORM_SOURCE,
                    "lower": 1,
                    "upper": 20,
                    "include_ledger": True,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
