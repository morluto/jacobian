"""Installation bundle for exact rational polynomial operations."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.polynomial.checkers import POLYNOMIAL_EXACT_REPLAY_CHECKERS
from jacobian.domains.polynomial.elementary import (
    INTEGER_POLYNOMIAL_CAPABILITIES,
    RATIONAL_POLYNOMIAL_CAPABILITIES,
)
from jacobian.domains.polynomial.groebner import POLYNOMIAL_GROEBNER_CAPABILITY
from jacobian.domains.polynomial.invariants import (
    POLYNOMIAL_INVARIANT_CAPABILITIES,
)
from jacobian.domains.polynomial.jacobian_syzygy import (
    GRADED_JACOBIAN_SYZYGY_CAPABILITY,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_CAPABILITY,
)
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_polynomial_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="polynomial",
        schema_namespace="jacobian.polynomial",
        semantics=DomainSemantics(
            name="jacobian.sparse-rational-polynomial-operations",
            version="1",
            definition={
                "coefficient_field": "QQ",
                "wire_term_order": "descending lexicographic",
                "zero_terms": "omitted",
                "gcd_normalization": "monic with an exact Bezout identity",
                "resultant": "Sylvester determinant in the named variable",
                "discriminant": (
                    "standard univariate convention: linear is 1; constant and zero are 0"
                ),
                "square_free_normalization": "separate coefficient and monic factors",
                "factorization": (
                    "univariate content and monic irreducible factors over QQ; "
                    "irreducibility is computed, not independently certified"
                ),
                "integer_polynomials": (
                    "dense canonical descending-degree coefficient strings over ZZ"
                ),
                "elementary_rational_polynomials": (
                    "sparse descending-lexicographic terms over QQ"
                ),
                "graded_jacobian_syzygies": (
                    "three-variable homogeneous coefficient maps use descending "
                    "lexicographic monomial bases and bounded exact rank search"
                ),
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=("exact-rational-polynomial-operations",),
        ),
        backend_version=SYMPY_VERSION,
        capabilities=(
            *POLYNOMIAL_INVARIANT_CAPABILITIES,
            POLYNOMIAL_GROEBNER_CAPABILITY,
            GRADED_JACOBIAN_SYZYGY_CAPABILITY,
            JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_CAPABILITY,
            *INTEGER_POLYNOMIAL_CAPABILITIES,
            *RATIONAL_POLYNOMIAL_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_POLYNOMIAL_REQUEST",
                stage="polynomial_input_validation",
                message="Input does not satisfy the bounded rational-polynomial contract.",
                hint="Use canonical sparse QQ polynomials and inspect the operation limits.",
            )
        ),
        checker_declarations=POLYNOMIAL_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_polynomial_bundle"]
