"""Static family-operation cards for ``math.find``.

These records are discovery metadata compiled from public request and result
models without constructing operation adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.finite_coverage import (
    FiniteCoverageVerifyOutput,
    FiniteCoverageVerifyRequest,
)
from jacobian.contracts.graph_composition import (
    GraphCompositionOutput,
    GraphCompositionRequest,
    GraphEnumerationOutput,
    GraphEnumerationRequest,
    GraphExplicitConstructionOutput,
    GraphExplicitConstructionRequest,
)
from jacobian.contracts.graph_degree_sequence import (
    GraphDegreeSequenceOutput,
    GraphDegreeSequenceRequest,
)
from jacobian.contracts.graph_invariants import (
    GraphAtlasSearchOutput,
    GraphAtlasSearchRequest,
    GraphInvariantBatchOutput,
    GraphInvariantBatchRequest,
    GraphNeighborhoodIndependenceOutput,
    GraphNeighborhoodIndependenceRequest,
)
from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismVerifyOutput,
    GraphIsomorphismVerifyRequest,
    SimpleUndirectedGraph,
)
from jacobian.contracts.lean import (
    LeanCheckOutput,
    LeanCheckRequest,
    LeanDeclarationInspectOutput,
    LeanDeclarationInspectRequest,
    LeanDeclarationSearchOutput,
    LeanDeclarationSearchRequest,
    LeanDependencyGraphArtifact,
    LeanDependencyGraphRequest,
)
from jacobian.contracts.lean_exploration import (
    LeanPremiseRetrievalOutput,
    LeanPremiseRetrievalRequest,
    LeanProofStateArtifact,
    LeanProofStateOutput,
    LeanProofStateRequest,
)
from jacobian.contracts.lean_metavariable_fields import (
    LeanMetavariableFieldsOutput,
    LeanMetavariableFieldsRequest,
)
from jacobian.contracts.lean_proof_axioms import (
    LeanProofAxiomsInspectOutput,
    LeanProofAxiomsInspectRequest,
)
from jacobian.contracts.lean_proof_edit import LeanProofEditOutput, LeanProofEditRequest
from jacobian.contracts.lean_proof_state_inspect import (
    LeanProofStateInspectOutput,
    LeanProofStateInspectRequest,
)
from jacobian.contracts.lean_statement import (
    LeanStatementComparisonOutput,
    LeanStatementComparisonRequest,
    LeanStatementProposalOutput,
    LeanStatementProposalRequest,
)
from jacobian.contracts.lean_term_apply import LeanTermApplyOutput, LeanTermApplyRequest
from jacobian.contracts.nullstellensatz import (
    JacobianDegreeSliceMaterializeOutput,
    JacobianDegreeSliceMaterializeRequest,
    NormalizedJacobianDegreeSliceSystem,
    NullstellensatzCertificateBundle,
    NullstellensatzCertificateOutput,
    NullstellensatzCertificateRequest,
    NullstellensatzVerificationOutput,
    NullstellensatzVerificationRequest,
)
from jacobian.contracts.operations import OperationExample, OperationInputKind
from jacobian.contracts.polynomial_expressions import (
    PolynomialExpressionNormalizationVerificationOutput,
    PolynomialExpressionNormalizationVerificationRequest,
    PolynomialExpressionNormalizeOutput,
    PolynomialExpressionNormalizeRequest,
)
from jacobian.contracts.polynomial_intervals import (
    PolynomialIntervalEnclosureOutput,
    PolynomialIntervalEnclosureRequest,
    PolynomialIntervalEnclosureVerifyOutput,
    PolynomialIntervalEnclosureVerifyRequest,
)
from jacobian.contracts.polynomial_positivity import (
    PolynomialIntervalPositivityOutput,
    PolynomialIntervalPositivityRequest,
    PolynomialIntervalPositivityVerifyOutput,
    PolynomialIntervalPositivityVerifyRequest,
)
from jacobian.contracts.polynomial_systems import (
    PolynomialSystemRationalSearchOutput,
    PolynomialSystemRationalSearchRequest,
    PolynomialSystemSolutionOutput,
    PolynomialSystemSolutionRequest,
)
from jacobian.contracts.polynomials import (
    PolynomialCollisionOutput,
    PolynomialCollisionRequest,
    PolynomialCollisionSearchOutput,
    PolynomialCollisionSearchRequest,
    PolynomialCollisionVerifyOutput,
    PolynomialCollisionVerifyRequest,
    PolynomialEvaluationOutput,
    PolynomialEvaluationRequest,
    PolynomialIdentityOutput,
    PolynomialIdentityRequest,
    PolynomialJacobianOutput,
    PolynomialJacobianRequest,
    PolynomialKellerConditionVerifyOutput,
    PolynomialKellerConditionVerifyRequest,
    PolynomialMapInverseCollisionVerifyOutput,
    PolynomialMapInverseCollisionVerifyRequest,
    PolynomialMapInverseSynthesisOutput,
    PolynomialMapInverseSynthesisRequest,
    PolynomialMapInverseVerifyOutput,
    PolynomialMapInverseVerifyRequest,
    RationalFunctionIdentityOutput,
    RationalFunctionIdentityRequest,
)
from jacobian.contracts.polytope import PolytopeSeparateRequest, PolytopeSeparateResult
from jacobian.contracts.results import ContractModel, VerificationResult
from jacobian.contracts.sat import (
    SatAssignmentArtifact,
    SatAssignmentVerificationOutput,
    SatAssignmentVerificationRequest,
    SatCnfMaterializationOutput,
    SatCnfMaterializationRequest,
    SatExplorationRequest,
    SatLratVerificationOutput,
    SatLratVerificationRequest,
    SatModelFindOutput,
    SatProofArtifact,
    SatUnsatProofFindOutput,
    SatUnsatProofVerificationOutput,
    SatUnsatProofVerificationRequest,
)
from jacobian.contracts.smt import (
    SmtAletheProofArtifact,
    SmtUnsatProofFindOutput,
    SmtUnsatProofFindRequest,
    SmtUnsatProofVerificationOutput,
    SmtUnsatProofVerificationRequest,
)
from jacobian.contracts.universal_algebra import (
    FiniteMagmaTableEnumerationOutput,
    FiniteMagmaTableEnumerationRequest,
    UniversalAlgebraCountermodelSearchOutput,
    UniversalAlgebraCountermodelSearchRequest,
    UniversalAlgebraEvaluationOutput,
    UniversalAlgebraEvaluationRequest,
)
from jacobian.schema_compiler import SCHEMA_COMPILER
from jacobian.schema_registry import json_schema_uri, model_schema_uri

# Keep this list aligned with ``jacobian.graphs.invariants.PROPERTY_NAMES``.
# Do not import that module here: index generation must not load NetworkX.
_GRAPH_COMPUTE_PROPERTIES_INVARIANTS = (
    "average_eccentricity",
    "bipartite",
    "connected",
    "degree_sequence",
    "diameter",
    "eccentricities",
    "girth",
    "harmonic_index",
    "havel_hakimi_trace",
    "independence_number",
    "maximum_degree",
    "minimum_degree",
    "order",
    "radius",
    "residue",
    "size",
    "tree",
    "triangle_count",
    "triangle_frequencies",
)

_GRAPH_SCHEMA_URI = model_schema_uri(
    name="jacobian.simple-undirected-graph",
    version="1",
    model=SimpleUndirectedGraph,
)
_SAT_ASSIGNMENT_SCHEMA_URI = model_schema_uri(
    name="jacobian.sat-assignment",
    version="1",
    model=SatAssignmentArtifact,
)
_SAT_PROOF_SCHEMA_URI = model_schema_uri(
    name="jacobian.sat-proof",
    version="1",
    model=SatProofArtifact,
)
_SMT_PROOF_SCHEMA_URI = model_schema_uri(
    name="jacobian.smt-alethe-proof",
    version="1",
    model=SmtAletheProofArtifact,
)
_LEAN_PROOF_STATE_SCHEMA_URI = json_schema_uri(
    name="jacobian.lean4-proof-state",
    version="1",
    schema=LeanProofStateArtifact.model_json_schema(),
)
_NULLSTELLENSATZ_SYSTEM_SCHEMA_URI = model_schema_uri(
    name="jacobian.normalized-jacobian-degree-2-3-system",
    version="1",
    model=NormalizedJacobianDegreeSliceSystem,
)
_NULLSTELLENSATZ_BUNDLE_SCHEMA_URI = model_schema_uri(
    name="jacobian.nullstellensatz-chart-cover-certificate",
    version="1",
    model=NullstellensatzCertificateBundle,
)


class CertificateReplayRequest(ContractModel):
    """Index-only schema twin of the certificate checker replay request."""

    certificate_uri: ArtifactUri


class WitnessReplayRequest(ContractModel):
    """Index-only schema twin of the witness checker replay request."""

    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    witness_uri: ArtifactUri


@dataclass(frozen=True, slots=True)
class FamilyIndexSpec:
    """One family operation's packaged discovery card."""

    operation_id: str
    version: str
    family: str
    title: str
    description: str
    tags: tuple[str, ...]
    request_type: type[BaseModel]
    result_type: type[BaseModel]
    examples: tuple[OperationExample, ...] = ()
    read_only: bool = False
    accepted_input_kinds: tuple[OperationInputKind, ...] = (
        OperationInputKind.STRUCTURED_REQUEST,
    )
    accepted_artifact_types: tuple[str, ...] = ()
    produced_artifact_types: tuple[str, ...] = ()


FAMILY_INDEX_SPECS: tuple[FamilyIndexSpec, ...] = (
    FamilyIndexSpec(
        operation_id="finite.coverage.verify",
        version="1",
        family="core",
        title="Verify exactly-once coverage of a finite paged archive",
        description="Independently verify exactly-once coverage of a finite paged archive.",
        tags=("finite", "coverage", "verification", "paged-archive"),
        request_type=FiniteCoverageVerifyRequest,
        result_type=FiniteCoverageVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="finite_magma.table.enumerate",
        version="1",
        family="core",
        title="Enumerate finite magma tables",
        description="Enumerate finite magma tables of one exact order.",
        tags=("universal-algebra", "finite-model", "enumeration"),
        request_type=FiniteMagmaTableEnumerationRequest,
        result_type=FiniteMagmaTableEnumerationOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.compute.neighborhood_independence",
        version="1",
        family="graph",
        title="Compute neighborhood independence",
        description="Compute an exact maximum independent set in every open neighborhood.",
        tags=("graph", "neighborhood", "independence-number", "exact-computation"),
        request_type=GraphNeighborhoodIndependenceRequest,
        result_type=GraphNeighborhoodIndependenceOutput,
        accepted_input_kinds=(OperationInputKind.TYPED_ARTIFACT,),
        accepted_artifact_types=(_GRAPH_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="graph.compute.properties",
        version="2",
        family="graph",
        title="Compute exact graph properties",
        description="Classify and compute a requested batch against the fixed exact graph-invariant registry.",
        tags=("graph", "properties", "exact-computation"),
        request_type=GraphInvariantBatchRequest,
        result_type=GraphInvariantBatchOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.construct.compose",
        version="1",
        family="graph",
        title="Compose graphs",
        description="Apply one deterministic graph composition operation to existing simple-undirected-graph artifacts.",
        tags=("graph", "construction", "composition"),
        request_type=GraphCompositionRequest,
        result_type=GraphCompositionOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.construct.explicit",
        version="1",
        family="graph",
        title="Materialize an explicit simple graph",
        description="Validate and canonicalize one bounded explicit finite simple undirected graph.",
        tags=("graph", "construction", "explicit", "artifact-materialization"),
        request_type=GraphExplicitConstructionRequest,
        result_type=GraphExplicitConstructionOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.degree_sequence.verify",
        version="1",
        family="graph",
        title="Verify a graph degree-sequence realization",
        description="Independently replay one exact realization or Erdos-Gallai obstruction with the installed graph checker.",
        tags=("graph", "degree-sequence", "verification"),
        request_type=CertificateReplayRequest,
        result_type=VerificationResult,
    ),
    FamilyIndexSpec(
        operation_id="graph.enumerate.nonisomorphic",
        version="1",
        family="graph",
        title="Enumerate nonisomorphic graphs",
        description="Enumerate all nonisomorphic simple undirected graphs of one exact Graph Atlas order.",
        tags=("graph", "enumeration", "nonisomorphic", "bounded-search"),
        request_type=GraphEnumerationRequest,
        result_type=GraphEnumerationOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.isomorphism.verify",
        version="2",
        family="graph",
        title="Verify an explicit graph-isomorphism mapping",
        description="Independently check that one explicit vertex bijection preserves all adjacency and nonadjacency.",
        tags=(
            "graph",
            "isomorphism",
            "mapping",
            "bijection",
            "adjacency",
            "verification",
            "adjacency-violation",
            "counter-witness",
        ),
        request_type=GraphIsomorphismVerifyRequest,
        result_type=GraphIsomorphismVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.neighborhood_independence.verify",
        version="1",
        family="graph",
        title="Verify graph neighborhood independence values",
        description="Independently replay one exact neighborhood-independence ledger.",
        tags=("graph", "neighborhood-independence", "verification"),
        request_type=CertificateReplayRequest,
        result_type=VerificationResult,
    ),
    FamilyIndexSpec(
        operation_id="graph.realize.degree_sequence",
        version="1",
        family="graph",
        title="Realize a simple-graph degree sequence",
        description="Construct a simple graph with the requested degree multiset, or return an obstruction.",
        tags=("graph", "degree-sequence", "construction", "counterexample"),
        request_type=GraphDegreeSequenceRequest,
        result_type=GraphDegreeSequenceOutput,
    ),
    FamilyIndexSpec(
        operation_id="graph.search.atlas",
        version="1",
        family="graph",
        title="Search the Graph Atlas",
        description="Search all Graph Atlas representatives of one exact order using exact computed constraints.",
        tags=("graph", "construction", "bounded-search"),
        request_type=GraphAtlasSearchRequest,
        result_type=GraphAtlasSearchOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.check",
        version="2",
        family="lean",
        title="Independently check an exact Lean proof",
        description="Independently check an exact Lean proof against the authorized Lean checker.",
        tags=(
            "lean",
            "proof",
            "checker",
            "verification",
            "core",
            "mathlib",
            "finite-witness",
            "proof-repair",
            "diagnostics",
            "type-mismatch",
            "source-span",
        ),
        request_type=LeanCheckRequest,
        result_type=LeanCheckOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.declaration.dependencies",
        version="2",
        family="lean",
        title="Extract Lean declaration dependencies",
        description="Extract the dependency graph of one pinned Lean or Mathlib declaration.",
        tags=("lean", "declaration", "dependency-graph", "formal-artifact"),
        request_type=LeanDependencyGraphRequest,
        result_type=LeanDependencyGraphArtifact,
    ),
    FamilyIndexSpec(
        operation_id="lean.declaration.inspect",
        version="2",
        family="lean",
        title="Inspect an exact Lean or Mathlib declaration",
        description="Inspect one exact Lean or Mathlib declaration from the pinned environment.",
        tags=(
            "lean",
            "mathlib",
            "declaration",
            "theorem-inspection",
            "formal-environment",
            "retrieval",
            "inspection",
        ),
        request_type=LeanDeclarationInspectRequest,
        result_type=LeanDeclarationInspectOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.declaration.search",
        version="2",
        family="lean",
        title="Search pinned Lean and Mathlib declarations",
        description="Search pinned Lean and Mathlib declarations in the authorized environment.",
        tags=(
            "lean",
            "mathlib",
            "declaration",
            "theorem-search",
            "formal-environment",
            "retrieval",
            "premise-discovery",
        ),
        request_type=LeanDeclarationSearchRequest,
        result_type=LeanDeclarationSearchOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.proof.axioms.inspect",
        version="1",
        family="lean",
        title="Inspect a Lean proof's reported axiom closure",
        description="Inspect the axiom closure reported for one Lean proof.",
        tags=("lean", "proof", "axioms", "trust-base", "inspection"),
        request_type=LeanProofAxiomsInspectRequest,
        result_type=LeanProofAxiomsInspectOutput,
        read_only=True,
    ),
    FamilyIndexSpec(
        operation_id="lean.proof_edit.validate",
        version="3",
        family="lean",
        title="Independently validate an exact Lean proof edit",
        description="Independently validate an exact Lean proof edit.",
        tags=(
            "lean",
            "proof-edit",
            "validation",
            "checker",
            "proof-repair",
            "diagnostics",
            "type-mismatch",
            "source-span",
        ),
        request_type=LeanProofEditRequest,
        result_type=LeanProofEditOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.proof_state.apply_tactic",
        version="3",
        family="lean",
        title="Apply one Lean tactic and inspect resulting goals",
        description="Apply one Lean tactic and inspect the resulting goals.",
        tags=(
            "lean",
            "proof-state",
            "tactic",
            "goals",
            "exploration",
            "proof-repair",
            "diagnostics",
            "tactic-error",
            "source-span",
        ),
        request_type=LeanProofStateRequest,
        result_type=LeanProofStateOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.proof_state.inspect",
        version="1",
        family="lean",
        title="Inspect an immutable Lean proof state without replay",
        description="Inspect an immutable Lean proof state without replay.",
        tags=("lean", "proof-state", "inspection", "exploration"),
        request_type=LeanProofStateInspectRequest,
        result_type=LeanProofStateInspectOutput,
        read_only=True,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_LEAN_PROOF_STATE_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="lean.proof_state.metavariable_fields",
        version="1",
        family="lean",
        title="Expose structured Lean metavariable and elaboration fields",
        description="Expose structured Lean metavariable and elaboration fields.",
        tags=("lean", "proof-state", "metavariable", "exploration"),
        request_type=LeanMetavariableFieldsRequest,
        result_type=LeanMetavariableFieldsOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_LEAN_PROOF_STATE_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="lean.retrieve.premises",
        version="2",
        family="lean",
        title="Retrieve Lean premises",
        description="Retrieve Lean premises for one proof state.",
        tags=("lean", "mathlib", "premise-retrieval", "exploration"),
        request_type=LeanPremiseRetrievalRequest,
        result_type=LeanPremiseRetrievalOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.statement.compare",
        version="1",
        family="lean",
        title="Compare two Lean statements and axiom sets (fail-closed)",
        description="Compare two Lean statements and axiom sets.",
        tags=("lean", "statement", "comparison", "axiom-set"),
        request_type=LeanStatementComparisonRequest,
        result_type=LeanStatementComparisonOutput,
    ),
    FamilyIndexSpec(
        operation_id="lean.statement.propose",
        version="2",
        family="lean",
        title="Propose one Lean statement with type-check status",
        description="Propose one Lean statement with type-check status.",
        tags=("lean", "statement", "elaboration", "proposal", "proposition"),
        request_type=LeanStatementProposalRequest,
        result_type=LeanStatementProposalOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.FORMAL_PROPOSITION,
        ),
    ),
    FamilyIndexSpec(
        operation_id="lean.term.apply",
        version="2",
        family="lean",
        title="Elaborate one Lean proof term against a proof state",
        description="Elaborate one Lean proof term against a proof state.",
        tags=(
            "lean",
            "term",
            "proof-term",
            "term-elaboration",
            "proof-state",
            "goals",
            "exploration",
            "proof-repair",
            "diagnostics",
            "type-mismatch",
            "source-span",
        ),
        request_type=LeanTermApplyRequest,
        result_type=LeanTermApplyOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_LEAN_PROOF_STATE_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="polynomial.expression.normalize",
        version="1",
        family="polynomial",
        title="Normalize a typed rational polynomial expression",
        description="Normalize a typed rational polynomial expression.",
        tags=(
            "polynomial",
            "symbolic",
            "normalization",
            "expansion",
            "product",
            "power",
            "coefficients",
            "typed-expression",
            "exact-rational",
            "sympy",
        ),
        request_type=PolynomialExpressionNormalizeRequest,
        result_type=PolynomialExpressionNormalizeOutput,
        examples=(
            OperationExample(
                name="combine_like_terms",
                description=(
                    "Normalize x + x to canonical sparse coefficients over QQ."
                ),
                input={
                    "expression": {
                        "variables": ["x"],
                        "expression": {
                            "kind": "add",
                            "operands": [
                                {"kind": "variable", "name": "x"},
                                {"kind": "variable", "name": "x"},
                            ],
                        },
                    }
                },
            ),
        ),
    ),
    FamilyIndexSpec(
        operation_id="polynomial.expression_normalization.verify",
        version="1",
        family="polynomial",
        title="Verify a typed polynomial normalization",
        description="Verify a typed polynomial normalization.",
        tags=(
            "polynomial",
            "symbolic",
            "normalization",
            "typed-expression",
            "exact-rational",
            "verification",
        ),
        request_type=PolynomialExpressionNormalizationVerificationRequest,
        result_type=PolynomialExpressionNormalizationVerificationOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.identity.verify",
        version="2",
        family="polynomial",
        title="Compare exact polynomials coefficient by coefficient",
        description="Compare exact polynomials coefficient by coefficient.",
        tags=(
            "polynomial",
            "identity",
            "equality",
            "verification",
            "exact-rational",
            "coefficient-equality",
            "sum-of-squares",
            "coefficient-mismatch",
            "counter-witness",
        ),
        request_type=PolynomialIdentityRequest,
        result_type=PolynomialIdentityOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.interval.enclose",
        version="1",
        family="polynomial",
        title="Enclose a univariate polynomial on a rational interval",
        description="Enclose a univariate polynomial on a rational interval.",
        tags=(
            "polynomial",
            "univariate",
            "interval",
            "enclosure",
            "bernstein",
            "exact-computation",
        ),
        request_type=PolynomialIntervalEnclosureRequest,
        result_type=PolynomialIntervalEnclosureOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.interval.enclosure.verify",
        version="1",
        family="polynomial",
        title="Verify a polynomial interval Bernstein enclosure",
        description="Verify a polynomial interval Bernstein enclosure.",
        tags=(
            "polynomial",
            "univariate",
            "interval",
            "enclosure",
            "bernstein",
            "verification",
        ),
        request_type=PolynomialIntervalEnclosureVerifyRequest,
        result_type=PolynomialIntervalEnclosureVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.interval.positivity.decide",
        version="1",
        family="polynomial",
        title="Decide strict positivity on a rational interval",
        description="Decide strict positivity on a rational interval.",
        tags=(
            "polynomial",
            "univariate",
            "interval",
            "positivity",
            "sturm",
            "exact-decision",
            "inequality",
            "exceeds-bound",
            "rational-derivative-bound",
        ),
        request_type=PolynomialIntervalPositivityRequest,
        result_type=PolynomialIntervalPositivityOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.interval.positivity.verify",
        version="1",
        family="polynomial",
        title="Verify a polynomial interval positivity decision",
        description="Verify a polynomial interval positivity decision.",
        tags=(
            "polynomial",
            "univariate",
            "interval",
            "positivity",
            "sturm",
            "verification",
            "verify",
            "inequality",
            "exceeds-bound",
            "rational-derivative-bound",
        ),
        request_type=PolynomialIntervalPositivityVerifyRequest,
        result_type=PolynomialIntervalPositivityVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.jacobian_degree_slice.system.materialize",
        version="1",
        family="polynomial",
        title="Materialize the normalized Jacobian degree-(2,3) slice",
        description="Materialize the exact 12-chart QQ polynomial system for normalized bivariate constant-Jacobian maps.",
        tags=("polynomial", "jacobian", "degree-slice", "rabinowitsch", "exact"),
        request_type=JacobianDegreeSliceMaterializeRequest,
        result_type=JacobianDegreeSliceMaterializeOutput,
        produced_artifact_types=(_NULLSTELLENSATZ_SYSTEM_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.collision.search",
        version="1",
        family="polynomial",
        title="Search a bounded rational grid for a collision",
        description="Search a bounded rational grid for a collision.",
        tags=("polynomial", "map", "collision", "bounded-search"),
        request_type=PolynomialCollisionSearchRequest,
        result_type=PolynomialCollisionSearchOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.collision.verify",
        version="1",
        family="polynomial",
        title="Verify a polynomial-map collision",
        description="Verify a polynomial-map collision.",
        tags=("polynomial", "map", "collision", "verification"),
        request_type=PolynomialCollisionVerifyRequest,
        result_type=PolynomialCollisionVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.collision_evidence.verify",
        version="1",
        family="polynomial",
        title="Verify stored polynomial-map collision evidence",
        description="Independently replay one exact stored collision witness against its bound map and injectivity claim.",
        tags=("polynomial", "map", "collision", "verification"),
        request_type=WitnessReplayRequest,
        result_type=VerificationResult,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.collision_witness",
        version="1",
        family="polynomial",
        title="Construct a polynomial-map collision witness",
        description="Construct a polynomial-map collision witness.",
        tags=("polynomial", "map", "collision", "witness", "artifact-composition"),
        request_type=PolynomialCollisionRequest,
        result_type=PolynomialCollisionOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.compute_jacobian",
        version="1",
        family="polynomial",
        title="Compute a polynomial-map Jacobian",
        description="Compute a polynomial-map Jacobian.",
        tags=("polynomial", "map", "jacobian", "determinant", "exact-computation"),
        request_type=PolynomialJacobianRequest,
        result_type=PolynomialJacobianOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.evaluate",
        version="1",
        family="polynomial",
        title="Evaluate a rational polynomial map",
        description="Evaluate a rational polynomial map.",
        tags=("polynomial", "map", "evaluation", "exact-computation"),
        request_type=PolynomialEvaluationRequest,
        result_type=PolynomialEvaluationOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.inverse.candidate_synthesize",
        version="1",
        family="polynomial",
        title="Synthesize a bounded polynomial-map inverse candidate",
        description="Synthesize a bounded polynomial-map inverse candidate.",
        tags=("polynomial", "map", "inverse", "synthesis", "exact-rational"),
        request_type=PolynomialMapInverseSynthesisRequest,
        result_type=PolynomialMapInverseSynthesisOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.inverse.refute_by_collision",
        version="1",
        family="polynomial",
        title="Refute a polynomial-map inverse by collision",
        description="Refute a polynomial-map inverse by collision.",
        tags=("polynomial", "map", "inverse", "collision", "verification"),
        request_type=PolynomialMapInverseCollisionVerifyRequest,
        result_type=PolynomialMapInverseCollisionVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.inverse.verify",
        version="1",
        family="polynomial",
        title="Verify a two-sided polynomial-map inverse",
        description="Verify a two-sided polynomial-map inverse.",
        tags=("polynomial", "map", "inverse", "verification", "exact-rational"),
        request_type=PolynomialMapInverseVerifyRequest,
        result_type=PolynomialMapInverseVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.map.keller_condition.verify",
        version="1",
        family="polynomial",
        title="Verify a polynomial-map Keller condition",
        description="Verify a polynomial-map Keller condition.",
        tags=("polynomial", "map", "jacobian", "Keller", "verification"),
        request_type=PolynomialKellerConditionVerifyRequest,
        result_type=PolynomialKellerConditionVerifyOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.nullstellensatz.infeasibility_certificate.compute",
        version="1",
        family="polynomial",
        title="Compute a bounded Nullstellensatz infeasibility certificate",
        description="Use pinned Singular lift computations to produce one exact certificate for each chart.",
        tags=("polynomial", "nullstellensatz", "singular", "certificate", "bounded"),
        request_type=NullstellensatzCertificateRequest,
        result_type=NullstellensatzCertificateOutput,
        accepted_input_kinds=(OperationInputKind.TYPED_ARTIFACT,),
        accepted_artifact_types=(_NULLSTELLENSATZ_SYSTEM_SCHEMA_URI,),
        produced_artifact_types=(_NULLSTELLENSATZ_BUNDLE_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="polynomial.nullstellensatz.infeasibility_certificate.verify",
        version="1",
        family="polynomial",
        title="Verify a chart-cover Nullstellensatz certificate",
        description="Independently multiply and sum every bounded QQ certificate identity for the degree slice.",
        tags=("polynomial", "nullstellensatz", "certificate", "verification", "exact"),
        request_type=NullstellensatzVerificationRequest,
        result_type=NullstellensatzVerificationOutput,
        accepted_input_kinds=(OperationInputKind.TYPED_ARTIFACT,),
        accepted_artifact_types=(
            _NULLSTELLENSATZ_SYSTEM_SCHEMA_URI,
            _NULLSTELLENSATZ_BUNDLE_SCHEMA_URI,
        ),
    ),
    FamilyIndexSpec(
        operation_id="polynomial.rational_function.identity.verify",
        version="1",
        family="polynomial",
        title="Verify an exact rational-function identity",
        description="Verify an exact rational-function identity.",
        tags=("polynomial", "rational-function", "identity", "verification"),
        request_type=RationalFunctionIdentityRequest,
        result_type=RationalFunctionIdentityOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.system.rational_solution.search",
        version="1",
        family="polynomial",
        title="Search a bounded rational grid for a polynomial-system solution",
        description="Search a bounded rational grid for a polynomial-system solution.",
        tags=("polynomial", "system", "solution", "bounded-search"),
        request_type=PolynomialSystemRationalSearchRequest,
        result_type=PolynomialSystemRationalSearchOutput,
    ),
    FamilyIndexSpec(
        operation_id="polynomial.system.solution.verify",
        version="1",
        family="polynomial",
        title="Verify an exact polynomial-system solution",
        description="Verify an exact polynomial-system solution.",
        tags=("polynomial", "system", "solution", "verification"),
        request_type=PolynomialSystemSolutionRequest,
        result_type=PolynomialSystemSolutionOutput,
    ),
    FamilyIndexSpec(
        operation_id="polytope.separate",
        version="1",
        family="core",
        title="Separate a rational point from a convex hull",
        description="Compute exact membership evidence or a separator; replay is separate.",
        tags=("polytope", "exact"),
        request_type=PolytopeSeparateRequest,
        result_type=PolytopeSeparateResult,
    ),
    FamilyIndexSpec(
        operation_id="sat.cnf.materialize",
        version="1",
        family="sat-smt",
        title="Materialize a canonical SAT CNF",
        description="Canonicalize and store the exact CNF consumed by SAT model and UNSAT search operations.",
        tags=(
            "sat",
            "cnf",
            "canonical-cnf",
            "materialization",
            "model",
            "unsat",
            "proof",
            "boolean-encoding",
            "finite-coloring",
            "forbidden-configurations",
            "exact-finite-existence",
            "certified-exhaustive-search",
        ),
        request_type=SatCnfMaterializationRequest,
        result_type=SatCnfMaterializationOutput,
        examples=(
            OperationExample(
                name="finite-coloring-cnf",
                description=(
                    "Encode two items with exactly one of two colors and forbid "
                    "them from sharing a color."
                ),
                input={
                    "variable_names": [
                        "item1_red",
                        "item1_blue",
                        "item2_red",
                        "item2_blue",
                    ],
                    "clauses": [
                        [1, 2],
                        [-1, -2],
                        [3, 4],
                        [-3, -4],
                        [-1, -3],
                        [-2, -4],
                    ],
                },
            ),
        ),
    ),
    FamilyIndexSpec(
        operation_id="sat.lrat.verify",
        version="2",
        family="sat-smt",
        title="Replay and verify an LRAT UNSAT proof",
        description="Replay and verify an LRAT UNSAT proof.",
        tags=(
            "sat",
            "cnf",
            "lrat",
            "unsat",
            "certificate",
            "verification",
            "proof-replay",
            "invalid-step",
            "rejection-witness",
        ),
        request_type=SatLratVerificationRequest,
        result_type=SatLratVerificationOutput,
    ),
    FamilyIndexSpec(
        operation_id="sat.model.find",
        version="1",
        family="sat-smt",
        title="Find a SAT assignment",
        description="Find a SAT assignment.",
        tags=(
            "sat",
            "cnf",
            "assignment",
            "exploration",
            "cadical",
            "boolean-encoding",
            "finite-coloring",
            "exact-finite-existence",
            "named-assignment",
        ),
        request_type=SatExplorationRequest,
        result_type=SatModelFindOutput,
    ),
    FamilyIndexSpec(
        operation_id="sat.model.verify",
        version="1",
        family="sat-smt",
        title="Verify a SAT assignment",
        description="Verify a SAT assignment.",
        tags=(
            "sat",
            "cnf",
            "assignment",
            "verification",
            "finite-coloring",
            "exact-finite-existence",
            "named-assignment",
        ),
        request_type=SatAssignmentVerificationRequest,
        result_type=SatAssignmentVerificationOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_SAT_ASSIGNMENT_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="sat.unsat_proof.find",
        version="1",
        family="sat-smt",
        title="Find a SAT UNSAT proof",
        description="Find a SAT UNSAT proof.",
        tags=(
            "sat",
            "cnf",
            "unsat",
            "proof",
            "exploration",
            "cadical",
            "boolean-encoding",
            "finite-coloring",
            "forbidden-configurations",
            "certified-exhaustive-search",
        ),
        request_type=SatExplorationRequest,
        result_type=SatUnsatProofFindOutput,
    ),
    FamilyIndexSpec(
        operation_id="sat.unsat_proof.verify",
        version="1",
        family="sat-smt",
        title="Verify a SAT UNSAT proof",
        description="Verify a SAT UNSAT proof.",
        tags=(
            "sat",
            "cnf",
            "unsat",
            "proof",
            "verification",
            "drat",
            "finite-coloring",
            "forbidden-configurations",
            "certified-exhaustive-search",
        ),
        request_type=SatUnsatProofVerificationRequest,
        result_type=SatUnsatProofVerificationOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_SAT_PROOF_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="smt.unsat_proof.find",
        version="1",
        family="sat-smt",
        title="Find a quantifier-free SMT UNSAT proof",
        description="Find a quantifier-free SMT UNSAT proof.",
        tags=(
            "smt",
            "unsat",
            "proof",
            "alethe",
            "exploration",
            "cvc5",
            "qf-uf",
            "qf-lia",
            "qf-lra",
        ),
        request_type=SmtUnsatProofFindRequest,
        result_type=SmtUnsatProofFindOutput,
    ),
    FamilyIndexSpec(
        operation_id="smt.unsat_proof.verify",
        version="1",
        family="sat-smt",
        title="Verify a compatible SMT UNSAT proof",
        description="Verify a compatible SMT UNSAT proof.",
        tags=("smt", "qf-uf", "unsat", "proof", "verification", "alethe", "carcara"),
        request_type=SmtUnsatProofVerificationRequest,
        result_type=SmtUnsatProofVerificationOutput,
        accepted_input_kinds=(
            OperationInputKind.STRUCTURED_REQUEST,
            OperationInputKind.TYPED_ARTIFACT,
        ),
        accepted_artifact_types=(_SMT_PROOF_SCHEMA_URI,),
    ),
    FamilyIndexSpec(
        operation_id="universal_algebra.evaluate_laws",
        version="1",
        family="core",
        title="Evaluate laws on a finite magma",
        description="Evaluate each equational law exactly on a finite binary-operation table.",
        tags=("universal-algebra", "finite-model", "law-evaluation", "counterexample"),
        request_type=UniversalAlgebraEvaluationRequest,
        result_type=UniversalAlgebraEvaluationOutput,
    ),
    FamilyIndexSpec(
        operation_id="universal_algebra.law_evaluation.verify",
        version="1",
        family="core",
        title="Verify a finite-magma law evaluation",
        description="Independently replay one exhaustive finite-magma law evaluation certificate.",
        tags=("universal-algebra", "finite-magma", "law-evaluation", "verification"),
        request_type=CertificateReplayRequest,
        result_type=VerificationResult,
    ),
    FamilyIndexSpec(
        operation_id="universal_algebra.search.countermodel",
        version="1",
        family="core",
        title="Search for a fixed-order finite magma countermodel",
        description="Search for a fixed-order finite magma countermodel.",
        tags=(
            "universal-algebra",
            "finite-model",
            "countermodel",
            "counterexample",
            "bounded-search",
        ),
        request_type=UniversalAlgebraCountermodelSearchRequest,
        result_type=UniversalAlgebraCountermodelSearchOutput,
    ),
)


def family_index_payloads() -> tuple[dict[str, Any], ...]:
    """Compile family discovery cards from public request and result models."""

    return tuple(_family_index_payload(spec) for spec in FAMILY_INDEX_SPECS)


def _family_index_payload(spec: FamilyIndexSpec) -> dict[str, Any]:
    input_schema = SCHEMA_COMPILER.compile_model(spec.request_type).definition()
    if spec.operation_id == "graph.compute.properties":
        input_schema = dict(input_schema)
        input_schema["x-supported-invariants"] = list(
            _GRAPH_COMPUTE_PROPERTIES_INVARIANTS
        )
    return {
        "operation_id": spec.operation_id,
        "version": spec.version,
        "family": spec.family,
        "title": spec.title,
        "description": spec.description,
        "tags": spec.tags,
        "examples": [
            {
                "name": example.name,
                "description": example.description,
                "input": dict(example.input),
            }
            for example in spec.examples
        ],
        "input_schema": input_schema,
        "output_schema": SCHEMA_COMPILER.compile_model(spec.result_type).definition(),
        "read_only": spec.read_only,
        "accepted_input_kinds": [kind.value for kind in spec.accepted_input_kinds],
        "accepted_artifact_types": list(spec.accepted_artifact_types),
        "produced_artifact_types": list(spec.produced_artifact_types),
    }


__all__ = [
    "FAMILY_INDEX_SPECS",
    "FamilyIndexSpec",
    "family_index_payloads",
]
