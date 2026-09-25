"""Public bounded semistandard-tableau enumeration operation."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    FixedContentCountRequest,
    FixedContentCountResult,
    SemistandardTableauEnumerationRequest,
    SemistandardTableauEnumerationResult,
)
from jacobian.math.combinatorics.semistandard_tableaux.content_count import (
    fixed_content_count,
)
from jacobian.math.combinatorics.semistandard_tableaux.enumeration import (
    enumerate_semistandard_young_tableaux,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorics.semistandard_young_tableaux.enumerate",
        title="Enumerate semistandard Young tableaux",
        description=(
            "Return every semistandard tableau of a straight partition shape "
            "with entries in a supplied finite alphabet, in lexicographic row "
            "order. The hook-content count bounds output before construction."
        ),
        request_type=SemistandardTableauEnumerationRequest,
        result_type=SemistandardTableauEnumerationResult,
        run=lambda request: enumerate_semistandard_young_tableaux(
            request.partition, request.max_entry
        ),
        tags=("combinatorics", "tableau", "semistandard", "exact"),
        discovery_terms=(
            "enumerate semistandard Young tableaux",
            "all semistandard tableaux with bounded entries",
        ),
        examples=(
            OperationExample(
                name="semistandard_tableaux_shape_2_1_alphabet_2",
                description=(
                    "Enumerate the two semistandard tableaux of shape (2,1) "
                    "with entries from {1,2}."
                ),
                input={"partition": {"parts": [2, 1]}, "max_entry": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorics.semistandard_young_tableaux.fixed_content_count",
        title="Count semistandard tableaux of fixed content",
        description=(
            "Return the exact number of semistandard Young tableaux of a "
            "straight partition shape with a specified sparse entry-to-"
            "multiplicity map. This is a fixed-content Kostka count, distinct "
            "from counting all entries in an alphabet 1..m. Entries are exact "
            "labels and missing labels have multiplicity zero. Admission bounds "
            "the complete multiset-prefix search before construction; one-row "
            "and standard-content cases use direct exact reductions."
        ),
        request_type=FixedContentCountRequest,
        result_type=FixedContentCountResult,
        run=fixed_content_count,
        tags=("combinatorics", "young-tableaux", "kostka", "exact"),
        discovery_terms=(
            "Kostka number for shape and content",
            "count semistandard tableaux with fixed content",
            "fixed weight semistandard Young tableau count",
        ),
        examples=(
            OperationExample(
                name="kostka_shape_21_content_122",
                description=(
                    "Count semistandard tableaux of shape (2,1) with one 1 and "
                    "two 2s. The sparse content terms retain their entry labels."
                ),
                input={
                    "partition": {"parts": [2, 1]},
                    "content": {
                        "terms": [
                            {"entry": 1, "multiplicity": 1},
                            {"entry": 2, "multiplicity": 2},
                        ]
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
