"""Installation bundle for exact combinatorics capabilities."""

from __future__ import annotations

import platform

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.combinatorics.checkers import COMBINATORICS_EXACT_REPLAY_CHECKERS
from jacobian.domains.combinatorics.counting import COUNTING_CAPABILITIES
from jacobian.domains.combinatorics.difference_sets import DIFFERENCE_SET_CAPABILITIES
from jacobian.domains.combinatorics.partitions import PARTITION_CAPABILITIES
from jacobian.domains.combinatorics.recurrence import RECURRENCE_CAPABILITIES
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_combinatorics_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="combinatorics",
        schema_namespace="jacobian.combinatorics",
        semantics=DomainSemantics(
            name="jacobian.exact-combinatorics",
            version="2",
            definition={
                "description": (
                    "Exact finite combinatorics, bounded linear recurrences, and "
                    "finite rational-series truncations"
                ),
                "arithmetic": "exact integer and rational via maintained SymPy and stdlib APIs",
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=("exact-combinatorics",),
        ),
        backend_version=f"python-{platform.python_version()};sympy-{SYMPY_VERSION}",
        capabilities=(
            *COUNTING_CAPABILITIES,
            *PARTITION_CAPABILITIES,
            *RECURRENCE_CAPABILITIES,
            *DIFFERENCE_SET_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_COMBINATORICS_REQUEST",
                stage="combinatorics_input_validation",
                message="Input does not satisfy the exact combinatorics contract.",
                hint="Provide bounded non-negative integers within each operation's limits.",
            )
        ),
        checker_declarations=COMBINATORICS_EXACT_REPLAY_CHECKERS,
    )
