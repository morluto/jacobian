"""Declarations for the Sidon extension-profile operation."""

from jacobian.catalog._examples import example
from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionProfileRequest,
    SidonExtensionProfileResult,
)
from jacobian.math.combinatorics._sidon_extension_operations import (
    compute_sidon_extension_profile,
)
from jacobian.math.combinatorics._support import combinatorics_operation

SIDON_EXTENSION_OPERATION = (
    combinatorics_operation(
        "combinatorics.integer_set.sidon.extension_profile.compute",
        "Compute Sidon extension profile",
        (
            "Given a source integer set A and a candidate set C disjoint from "
            "A, partition C into candidates x for which A plus x is Sidon and "
            "candidates for which it is not, each with a replayable "
            "repeated-difference obstruction."
        ),
        SidonExtensionProfileRequest,
        SidonExtensionProfileResult,
        compute_sidon_extension_profile,
        "combinatorics",
        "additive-combinatorics",
        "sidon",
        examples=(
            example(
                "sidon_extension_basic",
                (
                    "With source Sidon set {1, 2} and candidates {3, 4}, "
                    "candidate 3 fails because difference 1 repeats, while "
                    "candidate 4 succeeds. Source and candidate elements must "
                    "be unique and disjoint."
                ),
                {
                    "source_elements": ["1", "2"],
                    "candidate_elements": ["3", "4"],
                },
            ),
        ),
    ),
)

__all__ = ["SIDON_EXTENSION_OPERATION"]
