"""Installation bundle for exact matrix operations."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.matrix_lattice.capabilities import MATRIX_CAPABILITIES
from jacobian.domains.matrix_lattice.checkers import MATRIX_EXACT_REPLAY_CHECKERS
from jacobian.domains.matrix_lattice.hnf import HERMITE_NORMAL_FORM_CAPABILITY
from jacobian.operations import DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_matrix_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="matrix",
        schema_namespace="jacobian.matrix",
        semantics=DomainSemantics(
            name="jacobian.exact-matrix-operations",
            version="1",
            definition={
                "domains": ["QQ", "ZZ"],
                "maximum_rows": 32,
                "maximum_columns": 32,
                "maximum_decimal_digits_per_scalar_component": 256,
                "multiplication": "standard row-by-column product over QQ",
                "rref": "unique reduced row echelon form over QQ",
                "nullspace": "RREF fundamental basis ordered by ascending free column",
                "characteristic_polynomial": "dense det(lambda I - A) coefficients",
                "smith_normal_form": (
                    "positive divisibility diagonal; transformations unavailable"
                ),
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=(
                "exact-rational-matrix",
                "matrix-multiplication",
                "determinant",
                "rank",
                "rref",
                "nullspace",
                "characteristic-polynomial",
                "smith-normal-form",
            ),
        ),
        backend_version=SYMPY_VERSION,
        capabilities=(*MATRIX_CAPABILITIES, HERMITE_NORMAL_FORM_CAPABILITY),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_EXACT_MATRIX_REQUEST",
                stage="matrix_input_validation",
                message="Input does not satisfy the bounded exact matrix contract.",
                hint=(
                    "Use a nonempty 1..32 by 1..32 matrix with canonical QQ or ZZ "
                    "entries of at most 256 decimal digits."
                ),
            )
        ),
        checker_declarations=MATRIX_EXACT_REPLAY_CHECKERS,
    )
