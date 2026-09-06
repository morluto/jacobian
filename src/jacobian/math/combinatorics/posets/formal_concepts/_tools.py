"""Formal concept analysis operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.posets.formal_concepts._models import (
    AttributeSubsetRequest,
    ClosureResult,
    ConceptLatticeResult,
    ConceptResult,
    DerivationResult,
    DuquenneGuiguesBasisRequest,
    EnumerateConceptsRequest,
    EnumerateConceptsResult,
    ImplicationClosureRequest,
    ObjectSubsetRequest,
)
from jacobian.math.combinatorics.posets.formal_concepts.basis import (
    CanonicalImplicationBasisResult,
)
from jacobian.math.combinatorics.posets.formal_concepts.operations import (
    attribute_closure_result,
    attribute_derivation,
    concept_from_attributes,
    concept_from_objects,
    concept_lattice,
    duquenne_guigues_basis,
    enumerate_concepts,
    implication_closure,
    object_closure_result,
    object_derivation,
)
from jacobian.math.combinatorics.posets.formal_concepts.values import (
    FormalAttributeSubset,
    FormalConcept,
    FormalObjectSubset,
    ImplicationClosureResult,
)


def compute_object_derivation(request: ObjectSubsetRequest) -> DerivationResult:
    context = request.context
    subset = FormalObjectSubset(context=context, indices=tuple(sorted(request.subset)))
    derived = FormalAttributeSubset(
        context=context,
        indices=tuple(sorted(object_derivation(context, frozenset(request.subset)))),
    )
    return DerivationResult(
        context=context,
        subset=subset,
        side="OBJECT",
        derived=derived,
    )


def compute_attribute_derivation(
    request: AttributeSubsetRequest,
) -> DerivationResult:
    context = request.context
    subset = FormalAttributeSubset(
        context=context, indices=tuple(sorted(request.subset))
    )
    derived = FormalObjectSubset(
        context=context,
        indices=tuple(
            sorted(attribute_derivation(context, frozenset(request.subset)))
        ),
    )
    return DerivationResult(
        context=context,
        subset=subset,
        side="ATTRIBUTE",
        derived=derived,
    )


def compute_object_closure(request: ObjectSubsetRequest) -> ClosureResult:
    return object_closure_result(request.context, frozenset(request.subset))


def compute_attribute_closure(request: AttributeSubsetRequest) -> ClosureResult:
    return attribute_closure_result(request.context, frozenset(request.subset))


def compute_implication_closure(
    request: ImplicationClosureRequest,
) -> ImplicationClosureResult:
    return implication_closure(request.system, frozenset(request.seed))


def compute_duquenne_guigues_basis(
    request: DuquenneGuiguesBasisRequest,
) -> CanonicalImplicationBasisResult:
    return duquenne_guigues_basis(request.context)


def compute_concept_from_objects(request: ObjectSubsetRequest) -> ConceptResult:
    return concept_from_objects(request.context, frozenset(request.subset))


def compute_concept_from_attributes(request: AttributeSubsetRequest) -> ConceptResult:
    return concept_from_attributes(request.context, frozenset(request.subset))


def compute_enumerate_concepts(
    request: EnumerateConceptsRequest,
) -> EnumerateConceptsResult:
    concepts = enumerate_concepts(request.context)
    concepts_result = tuple(
        FormalConcept(
            context=request.context,
            extent=FormalObjectSubset(
                context=request.context, indices=tuple(sorted(concept["extent"]))
            ),
            intent=FormalAttributeSubset(
                context=request.context, indices=tuple(sorted(concept["intent"]))
            ),
        )
        for concept in concepts
    )
    return EnumerateConceptsResult(
        context=request.context,
        concepts=concepts_result,
        count=len(concepts_result),
    )


def compute_concept_lattice(
    request: EnumerateConceptsRequest,
) -> ConceptLatticeResult:
    return concept_lattice(request.context)


# A simple 2x2 context: objects {o0, o1}, attributes {a0, a1}
# o0 has a0; o1 has a1.
_CONTEXT = {
    "objects": ["o0", "o1"],
    "attributes": ["a0", "a1"],
    "incidence": [[0, 0], [1, 1]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="formal_context.objects.derivation.compute",
        title="Compute A' = {m : every g in A has m}",
        description="Return the exact derived attribute set for an object subset. Under "
        "standard FCA semantics, the derivation of the empty object set is "
        "every attribute.",
        request_type=ObjectSubsetRequest,
        result_type=DerivationResult,
        run=compute_object_derivation,
        tags=("formal-concept-analysis", "derivation", "exact"),
        examples=(
            OperationExample(
                name="empty_object_set",
                description="Derivation of the empty object set is every attribute.",
                input={"context": _CONTEXT, "subset": []},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.attributes.derivation.compute",
        title="Compute B' = {g : every m in B is possessed by g}",
        description="Return the exact derived object set for an attribute subset. Under "
        "standard FCA semantics, the derivation of the empty attribute set is "
        "every object.",
        request_type=AttributeSubsetRequest,
        result_type=DerivationResult,
        run=compute_attribute_derivation,
        tags=("formal-concept-analysis", "derivation", "exact"),
        examples=(
            OperationExample(
                name="empty_attribute_set",
                description="Derivation of the empty attribute set is every object.",
                input={"context": _CONTEXT, "subset": []},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.objects.closure.compute",
        title="Compute A'' = (A')' with added objects and closed status",
        description="Return the object closure A'' with the derived attributes A', added "
        "objects A''\\A, and whether A is already closed.",
        request_type=ObjectSubsetRequest,
        result_type=ClosureResult,
        run=compute_object_closure,
        tags=("formal-concept-analysis", "closure", "exact"),
        examples=(
            OperationExample(
                name="empty_object_closure",
                description="Closure of the empty object set.",
                input={"context": _CONTEXT, "subset": []},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.attributes.closure.compute",
        title="Compute B'' = (B')' with added attributes and closed status",
        description="Return the attribute closure B'' with the derived objects B', "
        "added attributes B''\\B, and whether B is already closed.",
        request_type=AttributeSubsetRequest,
        result_type=ClosureResult,
        run=compute_attribute_closure,
        tags=("formal-concept-analysis", "closure", "exact"),
        examples=(
            OperationExample(
                name="empty_attribute_closure",
                description="Closure of the empty attribute set.",
                input={"context": _CONTEXT, "subset": []},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.concept.from_objects.compute",
        title="Construct the concept (A'', A') from an object subset",
        description="Return the unique concept generated by an object subset: (A'', A').",
        request_type=ObjectSubsetRequest,
        result_type=ConceptResult,
        run=compute_concept_from_objects,
        tags=("formal-concept-analysis", "concept", "exact"),
        examples=(
            OperationExample(
                name="concept_from_o0",
                description="Concept from object o0.",
                input={"context": _CONTEXT, "subset": [0]},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.concept.from_attributes.compute",
        title="Construct the concept (B', B'') from an attribute subset",
        description="Return the unique concept generated by an attribute subset: (B', B'').",
        request_type=AttributeSubsetRequest,
        result_type=ConceptResult,
        run=compute_concept_from_attributes,
        tags=("formal-concept-analysis", "concept", "exact"),
        examples=(
            OperationExample(
                name="concept_from_a0",
                description="Concept from attribute a0.",
                input={"context": _CONTEXT, "subset": [0]},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.concepts.enumerate.compute",
        title="Enumerate every formal concept exactly once",
        description="Return the complete concept family of closed attribute intents "
        "using Ganter's NextClosure algorithm over the declared attribute "
        "order. The family, not the enumeration order, is mathematical. "
        "Admission proves the complete family fits the declared concept "
        "budget before enumeration.",
        request_type=EnumerateConceptsRequest,
        result_type=EnumerateConceptsResult,
        run=compute_enumerate_concepts,
        tags=("formal-concept-analysis", "enumeration", "exact"),
        examples=(
            OperationExample(
                name="enumerate_concepts",
                description="Enumerate all concepts of a 2x2 context.",
                input={"context": _CONTEXT},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.concept_lattice.compute",
        title="Compute the concept lattice",
        description="Return the complete concepts, partial order by extent inclusion, "
        "cover relation/Hasse diagram, and top and bottom concepts.",
        request_type=EnumerateConceptsRequest,
        result_type=ConceptLatticeResult,
        run=compute_concept_lattice,
        tags=("formal-concept-analysis", "lattice", "exact"),
        examples=(
            OperationExample(
                name="concept_lattice",
                description="Concept lattice of a 2x2 context.",
                input={"context": _CONTEXT},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_context.duquenne_guigues_basis.compute",
        title="Compute the exact Duquenne-Guigues implication basis",
        description="Return every pseudo-intent and its context closure, the complete "
        "canonical implication system, an exhaustive subset-closure matrix, "
        "explicit source-coordinate binding, and exact work/output accounting. "
        "One complete producer plan is admitted before enumeration; there is "
        "no partial-result branch.",
        request_type=DuquenneGuiguesBasisRequest,
        result_type=CanonicalImplicationBasisResult,
        run=compute_duquenne_guigues_basis,
        tags=(
            "formal-concept-analysis",
            "implication-system",
            "canonical-basis",
            "exact",
        ),
        examples=(
            OperationExample(
                name="empty_premise_canonical_basis",
                description="Compute the basis of a context whose empty-set closure is nonempty.",
                input={
                    "context": {
                        "objects": ["g0", "g1"],
                        "attributes": ["always", "sometimes"],
                        "incidence": [[0, 0], [1, 0]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="implication_system.closure.compute",
        title="Compute closure under a finite attribute implication system",
        description="Return the unique least superset of a seed that satisfies every finite "
        "attribute implication, together with the first canonical derivation of "
        "each added attribute and exact canonical closure work.",
        request_type=ImplicationClosureRequest,
        result_type=ImplicationClosureResult,
        run=compute_implication_closure,
        tags=("formal-concept-analysis", "implication-system", "closure", "exact"),
        examples=(
            OperationExample(
                name="two_round_implication_closure",
                description="Close {has_wings} through has_wings -> flies -> is_mobile; "
                "all rule and seed indices must refer to the declared attribute axis.",
                input={
                    "system": {
                        "attributes": ["has_wings", "flies", "is_mobile"],
                        "implications": [
                            {"premise": [0], "conclusion": [1]},
                            {"premise": [1], "conclusion": [2]},
                        ],
                    },
                    "seed": [0],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
