# Audit: validator-only public contract anti-patterns in Jacobian

**Date:** 2026-08-16
**Auditor:** Linzumi Codex session (f3d961c4-a0a6-4014-9abb-7fae29996655)
**Scope:** `src/jacobian/contracts/*.py` (request models) and `tests/` (test suite)

## Anti-pattern: "validator-only public contract"

> The API accepts a structured value whose decisive rules exist only in runtime
> validation, while the schema and examples do not teach the caller how to
> form it.

In Jacobian this manifests as: a `@model_validator(mode="after")` on a
Pydantic request model enforces a cross-field rule (e.g. "must be square",
"must be a bijection", "must be antisymmetric and acyclic"), while:

1. The Pydantic field types are permissive (`tuple[int, ...]`, `str`,
   `tuple[tuple[int, int], ...]`), so the JSON schema the agent sees does
   not express the constraint.
2. The operation's declared `examples=` tuple only shows happy-path
   values — values that happen to satisfy the validator. No example
   shows a value that would fail without the validator, so the example
   does not *teach* the rule.

The caller (an agent using `math.run`) forms a request from the schema +
examples, hits the validator, and gets a runtime rejection — with no
prior guidance.

## Findings — request models with validator-only public contracts

### Most clear-cut instances (permissive types + non-trivial cross-field rule + zero example coverage)

| Request model | File:line | Rule enforced only by validator | Example teaches rule? |
|---|---|---|---|
| `PermutationGroupRequest` | `src/jacobian/contracts/permutation_group.py:15` | each generator is a bijection of `0..degree-1` | No |
| `PermutationGroupOrbitRequest` | `src/jacobian/contracts/permutation_group.py:36` | same bijection rule | No |
| `ChineseRemainderRequest` | `src/jacobian/contracts/number_theory.py:304` | `residues`/`moduli` equal length; moduli in `[2,10000]`; residues canonical | No |
| `FinitePosetRequest` | `src/jacobian/contracts/posets.py:205` | antisymmetry + acyclicity + interpretation semantics | No |
| `ConvexPolygonTriangulationRequest` | `src/jacobian/contracts/geometry.py:362` | strict CCW convexity + complete diagonal weights | No |
| `MatrixDeterminantRequest` | `src/jacobian/contracts/matrix_operations.py:107` | matrix must be square | No |
| `SquareIntegerMatrixRequest` | `src/jacobian/contracts/matrix_operations.py:139` | matrix must be square | No |
| `RationalLinearSolveRequest` | `src/jacobian/contracts/matrix_operations.py:158` | square matrix + matching rhs length | No |
| `RationalMatrixProductRequest` | `src/jacobian/contracts/matrix_operations.py:46` | left columns == right rows (shape compatibility) | No |
| `BooleanTruthTableRequest` | `src/jacobian/contracts/boolean.py:15` | length must be a power of two | No |
| `UnivariatePolynomialRequest` | `src/jacobian/contracts/root_isolation.py:13` | leading coefficient must be nonzero | No |
| `ProjectiveLineArrangementRequest` | `src/jacobian/contracts/projective_geometry.py:59` | unique labels; no projective duplicates | No |
| `GraphConnectionProbabilityRequest` | `src/jacobian/contracts/probability.py:276` | edge probabilities cover graph edges; terminals distinct & declared | No |
| `ModularPolynomialResidueImageRequest` | `src/jacobian/contracts/number_theory.py:259` | variables unique; terms lexicographic; residues < modulus | No |
| `JacobiSymbolRequest` | `src/jacobian/contracts/number_theory.py:324` | `n` must be odd (schema allows even) | No |
| `LegendreSymbolRequest` | `src/jacobian/contracts/number_theory.py:173` | `prime` must be odd (schema allows even) | No |
| `DiscreteLogarithmRequest` | `src/jacobian/contracts/number_theory.py:337` | `base`, `target` must be `< modulus` | No |
| `FiniteEventRequest` | `src/jacobian/contracts/probability.py:499` | event values strictly increasing, subset of support | No |
| `FinitePushforwardRequest` | `src/jacobian/contracts/probability.py:649` | mapping sources exactly cover distribution atoms in canonical order | No |
| `FiniteConvolutionRequest` | `src/jacobian/contracts/probability.py:747` | pair product <= 4096; aggregated atoms <= 256 | No |
| `LineRequest` | `src/jacobian/contracts/geometry.py:120` | two points must be distinct | No |
| `PointSetRequest` | `src/jacobian/contracts/geometry.py:151` | points must be unique | No |
| `SimplePolygonPointRequest` | `src/jacobian/contracts/geometry.py:262` | polygon must be simple (non-self-intersecting) | No |
| `FlowGraph` (embedded in `MaxFlowRequest`/`MinCutRequest`) | `src/jacobian/contracts/graph_flow.py:20` | edge endpoints in `[0, vertex_count)` | No |
| `MaxFlowRequest`/`MinCutRequest` | `src/jacobian/contracts/graph_flow.py:38` | `source`/`sink` in `[0, vertex_count)` and distinct (enforced in operation body, not schema) | No |
| `FiniteIntegerSet` (embedded) | `src/jacobian/contracts/finite_sets.py:18` | elements must be unique | No |
| `MobiusFunctionRequest` | `src/jacobian/contracts/posets.py:504` | scope-vs-intervals rules | No |
| `LinearExtensionRequest` | `src/jacobian/contracts/posets.py:481` | `len(poset.elements) <= 14` (tighter than the schema's 64) | No |

### Note: `graph.invariant.chromatic_number.compute` declares no `examples=` at all

The `graph.invariant.chromatic_number.compute` operation
(`src/jacobian/domains/graph_optimization/chromatic_number.py`) declares
**no examples**, so the only way to learn how to form the
`GraphChromaticNumberRequest` is to read the schema — which does not
express the uniqueness/self-loop/orientation rules enforced by the
`ChromaticGraph` validator.

## Findings — test suite anti-patterns

### 1. Tests that encode undocumented cross-field validator rules

These tests assert that a request/result model rejects input because of a
`@model_validator` cross-field rule that is only discoverable by reading the
validator or the test — not the Pydantic schema or the operation's declared
examples. (This is the *test-side* symptom of the validator-only public
contract: the test is the only place that teaches the rule.)

Representative cases:

| Test | File:line | Rule the test encodes |
|---|---|---|
| `test_graph_symmetry_request_rejects_incomplete_permutation` | `tests/unit/contracts/test_graph_symmetry_contracts.py:39` | every generator must be a total vertex permutation |
| `test_graph_symmetry_request_rejects_color_breaking_generator` | `tests/unit/contracts/test_graph_symmetry_contracts.py:47` | generators must preserve declared vertex colors |
| `test_cover_relation_rejects_cycles_and_redundant_edges` | `tests/unit/contracts/test_poset_contracts.py:31` | antisymmetric + redundant-edge rules |
| `test_comparable_pairs_require_complete_transitive_relation` | `tests/unit/contracts/test_poset_contracts.py:53` | complete strict order |
| `test_sidon_request_rejects_duplicate_integer_elements` | `tests/unit/contracts/test_additive_combinatorics_contracts.py:18` | elements must be unique |
| `test_linear_system_requires_exact_matching_dimensions` | `tests/unit/contracts/test_linear_contracts.py:24` | rhs and variable dimensions must match |
| `test_facet_request_rejects_duplicates_nonmaximal_faces_and_hidden_isolates` | `tests/unit/contracts/test_topology_contracts.py:22` | facets distinct, maximal, no hidden isolates |
| `test_chinese_remainder_rejects_noncanonical_residues` | `tests/unit/contracts/test_number_theory_operation_contracts.py:18` | residues must be canonical |
| `test_chinese_remainder_rejects_invalid_system_bounds` | `tests/unit/contracts/test_number_theory_operation_contracts.py:31` | equal length + modulus bounds |
| `test_matrix_permanent_requires_square` | `tests/domain/matrix/test_permanent_kronecker_partial_trace.py:64` | matrix must be square |
| `test_symbolic_determinant_requires_square_matrix` | `tests/domain/symbolic_matrix/test_symbolic_matrix.py:92` | matrix must be square |
| `test_symbolic_matrix_rejects_non_rectangular` | `tests/domain/symbolic_matrix/test_symbolic_matrix.py:104` | rows same length |
| `test_syzygy_kernel_rejects_an_incomplete_linear_factor_request` | `tests/domain/polynomial/test_jacobian_syzygy_invariants.py:11` | linear-factor input completeness (uses `model_construct` to bypass validation and trigger a runtime rule) |

(Full list in the audit findings: ~40 tests across graph symmetry, posets,
additive combinatorics, linear, topology, certified SNF, recurrence, number
theory, probability, operation catalog, mutual information, and matrix
domains.)

### 2. Tests over-specifying internal implementation details

- `tests/domain/matrix/test_inverse_unimodular.py:5` — imports from private
  `jacobian.domains.certified_snf._kernel`; all tests exercise internal
  helper functions rather than the public `MathTool` operation contract.
- `tests/unit/domains/test_finite_field_bundle.py:35` — asserts on
  `request_type.model_fields["..."].annotation is <Type>` for 10+ fields,
  coupling the test to private Pydantic field annotations rather than the
  public request/result contract.
- `tests/component/providers/public_api/test_graphs.py:40` — asserts
  `SimpleUndirectedGraph.__module__ == "jacobian.math.graphs.values"`,
  coupling the test to the internal module path of the class.
- `tests/unit/contracts/test_topology_contracts.py:14` — imports four
  underscore-prefixed private helpers (`_canonical_complex`,
  `_canonicalize`, `_chain_result`, `_homology`) from
  `jacobian.domains.topology.operations` and calls them directly.

### 3. Weak or missing assertions

- `tests/unit/test_operation_dispatcher.py:22` —
  `test_invoke_operation_runs_determinant_directly` asserts
  `result.runtime_ms >= 0` and `result.output["determinant"] == ...` but
  never asserts `result.output is not None` before indexing or checks
  `is_error`.
- `tests/unit/test_serving_catalog.py:22` —
  `test_invoke_operation_runs_determinant_without_state` — same pattern.
- `tests/boundary/mcp/test_mcp_sdk_2_conformance.py:76` — the
  `invalid_request` branch asserts only `is_error is True` without
  checking the error content or code.
- `tests/boundary/mcp/test_mcp_invocation_journey.py:57` — the `invalid`
  branch asserts `is_error is True` and `structured_content is None` but
  never validates which validation error was triggered (no `match=` on
  the error).

### 4. Tests depending on example data living in another module

- `tests/composition/catalog/test_builtin_invocation_examples.py:33` —
  imports the operation bundle, takes `operation.examples[0]`, validates
  it, and runs it — round-trips the declared example without teaching
  what the contract requires; if the example input changes, the test
  silently passes.
- `tests/boundary/mcp/test_mcp_sdk_2_conformance.py:96` — fetches
  `matrix.determinant.compute` via `math.find`, takes
  `contract["operation"]["examples"][0]["input"]`, and feeds it into
  `math.run`; only asserts `isinstance(result.structured_content, dict)`
  rather than the determinant value.
- `tests/boundary/mcp/test_mcp_invocation_journey.py:108` — fetches
  `graph.invariant.maximum_matching.compute` via `math.find` and asserts
  `matching_contract["operation"]["examples"][0]["name"] ==
  "triangle_with_tail"`, hard-coupling the test to the example's declared
  name in another module.

## Recommendation

### For the validator-only public contract

1. **Express constraints in the schema where feasible.** For fields where
   the constraint is a simple bound on a single field (e.g. "must be odd",
   "must be nonzero", "must be < modulus"), tighten the `Field` constraint
   or use a `Literal`/`Annotated` type so the JSON schema teaches the
   rule.
2. **Add a negative example for each cross-field rule.** For each request
   model with a `@model_validator`, add at least one operation `example`
   whose description explicitly states the rule and shows a value that
   would fail without the validator (or, if the example contract requires
   valid inputs, document the rule in the operation `description`).
3. **Document the rule in the field/docstring.** The
   `OperationExample.description` field is capped at 256 characters but
   can carry a one-line rule statement; the request model's docstring can
   carry a longer one.
4. **Add `examples=` to `graph.invariant.chromatic_number.compute`** —
   it currently declares none, so the contract is undisoverable from the
   catalog alone.

### For the test suite

1. **Promote undocumented rules from tests to examples/docs.** Each test
   that encodes a cross-field rule is evidence the rule is only taught by
   the test; the rule should also be discoverable from the schema or an
   example.
2. **Decouple over-specified tests from internals.** Replace private
   `_kernel` imports, `model_fields[...].annotation is ...` assertions,
   and `__module__` checks with calls through the public `MathTool`
   operation surface.
3. **Strengthen weak assertions.** Add `is_error` checks and `match=`
   on error content where a test currently only checks `is_error is True`.
4. **Make example-data-coupled tests assert on the contract result**, not
   just `isinstance(result, dict)`, and avoid hard-coupling to example
   `name`s in other modules.
