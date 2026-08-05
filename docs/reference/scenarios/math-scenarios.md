# Mathematical scenario catalog

[Documentation home](../../index.md)

- Status: Active scenario catalog; supported contracts are defined by the
  applicable specification and conformance document
- Purpose: Exact public fixtures, reference-plugin workloads, and held-out
  model evaluations

## Decision

Jacobian should be tested first on small mathematical obligations, not on a
leaderboard of famous conjectures.

A whole conjecture entangles statement formalization, representation choice,
search, verification, and explanation. When it fails, we learn very little
about which capability is missing. A component scenario isolates one operation:

```text
validate a quantified claim
find and replay one counter-witness
close a finite semantic family
check an exhaustive certificate
shrink while preserving a predicate
verify a representation relation
enumerate a declared finite scope
separate a rational point from a convex hull
```

The initial corpus therefore has four layers:

1. **Public exact micro-scenarios.** Tiny, hand-auditable fixtures used in
   contract and conformance tests. Their answers are intentionally public.
2. **Reference-plugin workflows.** Several micro-scenarios combined through
   one domain plugin and replayed in a clean process.
3. **Held-out structural variants.** Generated or manually transformed cases
   with hidden oracles, used for model-in-the-loop evaluation.
4. **Imported and historical corpora.** Pinned external statements or
   evaluator programs used for later compatibility, formal-proof, and
   rediscovery experiments.

Passing layer 1 is necessary but says nothing about model creativity. Passing
layer 3 says something about tool-assisted behavior but cannot replace layer 1
checker conformance.

Capability IDs below name candidate agent-facing contracts for the scenario.
They do not imply that a capability is installed; agents must discover
availability through `capability://catalog` and inspect the exact contract with
`math.find`.

## Scenario record

Every scenario receives an immutable manifest containing:

```text
scenario ID and version
capability and contract version under test
public claim and starting artifacts
declared finite scope
expected operational state
expected mathematical conclusion
required assurance mechanism
accepted witnesses or certificate class
hidden or public oracle digest
adversarial mutations
scale parameters
source and derivation
license and redistribution status
knowledge cutoff and contamination class
checker and environment requirements
```

Public fixtures may store the complete oracle beside the input. Held-out
fixtures store only an oracle digest and evaluator-visible metadata in the
candidate workspace.

An imported dataset row is never its own oracle. A `proved`, `valid`, or
`answer` field is provenance to investigate; Jacobian must replay the relevant
checker under a pinned environment.

## Public exact micro-scenarios

These scenarios are deliberately small enough to verify with standard-library
Python and exact integers or rationals.

### QNT-CYCLE-001 — One finite quantifier alternation

**Claim**

Let \(U = \mathbb Z/3\mathbb Z\). Verify

\[
\forall x \in U\;\exists y \in U:\quad y \equiv x+1 \pmod 3.
\]

**Public oracle**

```text
x = 0 -> y = 1
x = 1 -> y = 2
x = 2 -> y = 0
```

**Capabilities**

- `finite_logic.validate.quantified_claim` checks finite domains, bound
  variables, quantifier order, and the modular-arithmetic semantics reference.
- `finite_logic.evaluate.witness_table` may score a proposed witness table.
- `finite_logic.verify.exhaustive_table` replays the complete three-row table.

**Adversarial variants**

- leave `y` unbound;
- refer to modulus zero;
- omit the row for `x = 2` while claiming exhaustive coverage;
- bind a valid table to modulus 4.

This is a schema and quantifier scenario, not a test of whether a model can
recall a theorem.

### INT-FACTOR-001 — Direct counter-witness versus finite nonexistence

**Claim**

For a bounded positive integer candidate \(n\), determine whether \(n\) is
prime.

**Instances**

- \(n=91\): false, with direct witness \(91=7\cdot13\).
- \(n=97\): true in the declared divisor domain, with a complete table showing
  that no \(d\in\{2,\ldots,96\}\) divides 97.

**Capabilities**

- `integer.search.factor` returns a factor for 91.
- `integer.verify.factor` checks bounds, nontriviality, and exact
  multiplication.
- `integer.verify.primality_table` checks the complete finite divisor table for
  97.

**Adversarial variants**

- factors 1 and 91;
- a factor of a different candidate;
- a table stopping at 9 but lacking a checked square-root argument;
- a timeout relabeled as “prime.”

The search side may use trial division only through \(\lfloor\sqrt n\rfloor\).
The first independent checker should deliberately use the simpler full bounded
domain.

### BOOL-MP-001 — Counterassignment and exhaustive truth table

**Claims**

1. \((p \land (p\rightarrow q))\rightarrow q\) is a tautology.
2. \(p\lor q\) is a tautology.

**Public oracle**

- The first claim has four satisfying truth-table rows.
- The second is false under \(p=\mathrm{false},q=\mathrm{false}\).

**Capabilities**

- `boolean.search.counterassignment` returns a counterassignment for the false
  claim.
- `boolean.verify.assignment` directly evaluates the formula under that
  assignment.
- `boolean.verify.truth_table` checks all four rows for the true claim.

**Adversarial variants**

- omit one truth-table row;
- change variable ordering without changing the encoding digest;
- provide an assignment with an unknown variable;
- copy the tautology certificate to another formula.

### BOOL-MUS-001 — Checked shrinking of an unsatisfiable core

**Candidate**

\[
(x)\land(\neg x)\land(y\lor z).
\]

**Public oracle**

The candidate is unsatisfiable. Under clause deletion, the one-step-minimal
core is:

\[
(x)\land(\neg x).
\]

Removing either remaining clause makes the formula satisfiable.

**Capabilities**

- `boolean.evaluate.cnf` reports an unverified unsatisfiable result or
  objective.
- `boolean.minimize.unsat_core` proposes clause deletions.
- the preservation checker rejects any step that makes the formula satisfiable;
- `boolean.verify.unsat_certificate` replays a complete table or a registered
  SAT proof format.

**Adversarial variants**

- a reducer claims minimality without checking both final deletions;
- variable renaming is treated as clause deletion;
- a cached UNSAT result survives a changed clause set.

### PATH-CLOSURE-001 — Intended paths are not the path language

**Structure**

Vertices:

```text
s, a, b, x, t1, t2
```

Arcs:

```text
s -> a
a -> x
s -> b
b -> x
x -> t1
x -> t2
```

The designer lists only:

```text
s -> a -> x -> t1
s -> b -> x -> t2
```

**Public oracle**

The actual graph induces four simple source-terminal paths, including:

```text
s -> a -> x -> t2
s -> b -> x -> t1
```

**Capabilities**

- `graph.enumerate.simple_paths` enumerates the complete bounded path family;
- `graph.search.omitted_path` returns one omitted path;
- `graph.verify.path` checks every arc and the source/terminal role;
- `graph.minimize.counterexample` may minimize the responsible subgraph.

**Adversarial variants**

- one terminal is renamed without updating distinguished roles;
- an intended-path list is mislabeled exhaustive;
- duplicate arcs or a path with a missing arc;
- a depth or path-count limit is reached but coverage remains exhaustive.

This tiny synthetic graph replaces a full routing conjecture as the first
semantic-closure fixture.

### GRAPH-BIP-001 — Shrink a structural counterexample

**Candidate**

A triangle together with three isolated vertices.

**Claim**

The graph is bipartite.

**Public oracle**

The triangle is an odd-cycle witness. Deleting the three isolated vertices
preserves non-bipartiteness; deleting any triangle edge does not.

**Capabilities**

- `graph.search.odd_cycle` returns an odd cycle;
- `graph.verify.odd_cycle` checks adjacency, closure, odd length, and distinct
  interior vertices;
- `graph.minimize.counterexample` reduces the candidate to the triangle and
  reports a checked local reduction. A one-step-minimality claim needs separate
  neighborhood-completeness evidence.

**Adversarial variants**

- an even cycle;
- a repeated interior vertex;
- an edge from another graph;
- a “smaller” graph that lost the odd cycle.

### MAT-KERNEL-001 — Exact linear-algebra witness

**Candidate**

\[
A=\begin{pmatrix}2&4\\1&2\end{pmatrix}.
\]

**Claim**

\(A\) is nonsingular over the rationals.

**Public oracle**

\[
v=\begin{pmatrix}2\\-1\end{pmatrix}\neq0,\qquad Av=0.
\]

**Capabilities**

- `matrix.search.kernel_vector` proposes a kernel vector;
- `matrix.verify.kernel_vector` checks dimensions, nonzeroness, and exact
  multiplication;
- `matrix.compute.determinant` may report \(\det A=0\), but it cannot certify
  its own result.

**Adversarial variants**

- the zero vector;
- a vector of the wrong dimension;
- a valid vector rebound to a nearby matrix;
- floating residuals treated as exact zero.

### MAT-MAXDET3-001 — Tiny solved optimization question

**Question**

Among all \(3\times3\) matrices with entries in \(\{-1,1\}\), maximize
\(|\det A|\).

**Public oracle**

The exact optimum is 4. One maximizer is:

\[
\begin{pmatrix}
-1&-1&-1\\
-1&-1&1\\
-1&1&-1
\end{pmatrix}.
\]

There are \(2^9=512\) labeled candidates. A complete labeled enumeration finds
192 maximizers.

**Capabilities**

- `matrix.compute.determinant` computes exact determinant objectives;
- a candidate matrix is a lower-bound witness;
- `matrix.verify.exhaustive_determinant_table` checks all 512 determinant rows
  for the upper bound;
- `matrix.minimize.determinant_certificate` may simplify a certificate but must
  not alter its scope.

**Adversarial variants**

- enumerate only normalized first-row matrices while claiming labeled scope;
- omit determinant signs but claim the signed optimum;
- substitute a \(3\times2\) matrix;
- use floating determinant rounding.

This is the initial non-graph reference-plugin optimization workload.

### ERDOS-STRAUS-001 — Bounded unit-fraction verification

**Bounded claim**

For every integer \(n\) in the exact closed interval \(2\leq n\leq1000\),
there are positive integers \(x,y,z\) such that

\[
\frac4n=\frac1x+\frac1y+\frac1z.
\]

This is a finite executable instance of the Erdős-Straus conjecture. It is not
a proof of the unbounded conjecture.

**Evidence**

The search plugin proposes one row \((n,x,y,z)\) for every integer in the
interval. The independent checker requires exactly the declared set of
\(n\)-values, rejects duplicates and nonpositive denominators, and checks each
row using only the integer identity

\[
4xyz=n(xy+xz+yz).
\]

**Capabilities**

- `number_theory.search.erdos_straus_table` proposes a complete decomposition
  table using exact bounded search;
- `number_theory.verify.erdos_straus_table` independently replays every row and
  the exact interval.

**Adversarial variants**

- omit one value of \(n\) while claiming exhaustive coverage;
- duplicate one valid row to preserve the row count;
- use a valid table for a nearby upper bound;
- accept zero or negative denominators;
- promote timeout or failure to find a row into a counterexample;
- report the finite result as a proof of the full conjecture.

### MAGMA-IMPL-001 — A finite countermodel to a false implication

**Claim**

Every commutative magma is associative.

**Countermodel**

On \(\{0,1\}\), use the operation table:

| \(\diamond\) | 0 | 1 |
| --- | ---: | ---: |
| 0 | 1 | 0 |
| 1 | 0 | 0 |

The table is commutative. Associativity fails for \((0,0,1)\):

```text
(0 ◇ 0) ◇ 1 = 0
0 ◇ (0 ◇ 1) = 1
```

**Capabilities**

- the candidate is a finite operation table;
- the witness contains a model and a violating variable assignment;
- `finite_magma.verify.countermodel` checks closure, every commutativity row,
  and the displayed associativity failure;
- a complete certificate may enumerate all eight triples.

**Adversarial variants**

- an out-of-domain table entry;
- a table that violates the hypothesis as well as the conclusion;
- an assignment from another table;
- an evaluator checks only the displayed commutative pairs.

This scenario is inspired by the finite-countermodel workflow of the
[Equational Theories project](https://github.com/teorth/equational_theories),
but the two-element table and checker are independently generated for Jacobian.

### JAC-DEG-023 — Normalized bivariate degree-(2,3) infeasibility

Let a characteristic-zero bivariate polynomial map have exact component
degrees `(2,3)` and nonzero constant Jacobian. If `A=JF(0)`, then
`F(A^{-1}z)-F(0)` preserves the component degrees and has constant Jacobian
one with value `I` at the origin. Over `QQ` the normalized coefficient slice is

```text
P=x+a20*x^2+a11*x*y+a02*y^2
Q=y+b20*x^2+b11*x*y+b02*y^2+b30*x^3+b21*x^2*y+b12*x*y^2+b03*y^3.
```

The original characteristic-zero claim, this rational normalized slice, the
generated coefficient system, and infeasibility of that system are distinct
claims. This scenario certifies only the declared `QQ` slice.

Exact degree is

```text
(a20 != 0 or a11 != 0 or a02 != 0)
and (b30 != 0 or b21 != 0 or b12 != 0 or b03 != 0).
```

It is represented by twelve charts. Chart `(a_i,b_j)` adds a fresh `t` and
`t*a_i*b_j-1`; the charts cover the disjunction because any pair of nonzero
coordinates selects one chart. A global alternative would introduce
`u20,u11,u02,v30,v21,v12,v03` and equations
`u20*a20+u11*a11+u02*a02-1` and
`v30*b30+v21*b21+v12*b12+v03*b03-1`.

Each chart certificate gives exact rational multipliers with
`sum(h_i*f_i)=1`. Singular may produce them, but only the separate sparse-QQ
replay checker can promote the bound infeasibility claim to `VERIFIED`.

**Adversarial variants**

- require every top coefficient to be nonzero instead of encoding the two
  disjunctions;
- omit one of the twelve charts while claiming complete scope;
- reuse multipliers after reordering variables or changing a generator;
- bind a certificate to a different system artifact;
- treat Singular success, timeout, or an oversized partial output as product
  verification;
- generalize the rational slice result to every characteristic-zero
  coefficient field without a separately justified transfer argument.

## Reference-plugin decision

Maintain reference scenarios across structurally different domains. Finite
logic exercises schemas and checker dispatch; directed graphs exercise semantic
closure; integer matrices exercise exact arithmetic; finite magmas exercise a
model-plus-assignment witness with a natural route to theorem-prover
integration. These are portfolio coverage targets, not a mandatory
implementation sequence.

The full routing counterexample episode is not a prerequisite for exposing the
atomic path capabilities. It becomes an end-to-end regression workload after
the tiny path-language fixture has established the semantic-closure boundary.

## Discovery and transformation scenarios

### XFORM-AMO-001 — Equivalence versus relaxation

**Source predicate**

At most one of \(a,b,c\) is true.

**Correct CNF**

\[
(\neg a\lor\neg b)\land
(\neg a\lor\neg c)\land
(\neg b\lor\neg c).
\]

**Public oracle**

The full encoding is equivalent over all eight assignments. If
\((\neg a\lor\neg c)\) is omitted, the result is only a relaxation:

```text
a = true
b = false
c = true
```

satisfies the incomplete CNF but violates the source predicate.

**Capabilities**

- `boolean.transform.at_most_one_cnf` emits the encoding and variable map;
- `boolean.verify.encoding_relation` checks the declared relation by complete
  enumeration;
- a direct assignment refutes an incorrect equivalence claim;
- `boolean.minimize.encoding_counterexample` minimizes the omitted-clause
  explanation.

### POLY-SEP-001 — Exact membership and separation

**Generator set**

\[
S=\{000,100,010,001\}.
\]

**Points**

- \((1/4,1/4,1/4)\) lies in \(\operatorname{conv}(S)\).
- \((1/2,1/2,1/2)\) lies outside it.

**Public oracle**

The second point violates:

\[
z_1+z_2+z_3\le1
\]

by exactly \(1/2\). The first has convex weights
\((1/4,1/4,1/4,1/4)\).

**Capabilities**

- exact convex-hull membership;
- exact separator construction and replay;
- sparse primitive-normal normalization;
- distinction between a membership witness and a separation certificate.

An additional variant compares the integral hull with the pairwise relaxation
\(z_i+z_j\le1\), which admits \((1/2,1/2,1/2)\).

### ENUM-NQUEENS-001 — Parameterized finite enumeration

**Question**

Count labeled \(n\)-queens placements for \(1\le n\le5\).

**Public oracle**

```text
n:      1  2  3  4   5
count:  1  0  0  2  10
```

The two \(n=4\) row vectors by column are:

```text
(1, 3, 0, 2)
(2, 0, 3, 1)
```

**Capabilities**

- `constraint.enumerate.n_queens` declares the full row-vector scope;
- a solution is a direct witness;
- zero counts require a complete certificate;
- later symmetry reduction must distinguish labeled solutions from orbits.

The formulation follows
[CSPLib problem 054](https://www.csplib.org/Problems/prob054/). Jacobian
recomputes its small oracle independently and does not vendor CSPLib files
until their redistribution terms are explicitly resolved.

### ENUM-RAMSEY-001 — Alternating quantifiers at a larger finite scale

**Questions**

- Exhibit a red/blue edge coloring of \(K_5\) with no monochromatic triangle.
- Verify that every red/blue edge coloring of \(K_6\) has a monochromatic
  triangle.

**Public oracle**

- A 5-cycle in red with its complement in blue is a \(K_5\) witness.
- The \(K_6\) exhaustive scope contains \(2^{15}=32768\) colorings and none
  avoids a monochromatic triangle.

**Capabilities**

- direct model witness for the existential side;
- complete enumeration for the universal side;
- symmetry-aware enumeration may optimize search but must bind its orbit
  certificate.

This is public conformance, not a hidden test of whether a model remembers
\(R(3,3)=6\).

### ENUM-CAPSET-001 — Small construction and optimum

**Question**

Find the largest subset of \(\mathbb F_3^2\) containing no three distinct
points whose sum is zero.

**Public oracle**

The optimum is 4. One maximizer is:

```text
(1,1), (1,2), (2,1), (2,2)
```

The complete subset scope has \(2^9=512\) candidates.

**Capabilities**

- finite-set candidate validation;
- forbidden-triple witnesses;
- exhaustive upper-bound certificate;
- construction search and later program-search scaling.

### ENUM-MAGMA-001 — Complete finite model space

There are \(2^{2^2}=16\) binary operation tables on a labeled two-element
carrier. Enumerate them and classify each under selected equational laws such
as commutativity and associativity.

This scenario checks candidate enumeration, operation-table canonical bytes,
law evaluation, counterassignments, and honest distinction between labeled
tables and isomorphism classes.

## Search and refinement scenarios

These scenarios compose earlier exact operations rather than inventing new
truth semantics:

| ID | Workflow | Exact promotion gate |
| --- | --- | --- |
| `REFINE-AMO-001` | Start from an incomplete at-most-one encoding; refine it with each verified defeating assignment | `boolean.verify.encoding_relation` proves equivalence over all eight assignments |
| `SEARCH-DET3-001` | Explore \(\{-1,1\}\) matrices under a plugin-defined proposal strategy | Candidate determinant replay plus the existing 512-row optimum certificate |
| `SEARCH-NQUEENS-001` | Explore partial queen placements under a plugin-defined search strategy | Direct solution checker or complete bounded search certificate |
| `SEARCH-CAPSET-001` | Search for a constructor producing cap sets over small finite fields | Every materialized set receives forbidden-triple replay |
| `RESUME-SEARCH-001` | Interrupt and resume any of the above after a fixed number of evaluations | Identical lineage and promoted verified artifacts after clean replay |

Search score, novelty, or model confidence never replaces the promotion gate.
The scenario harness may use exact enumeration, refinement, tree search,
evolutionary search, or another strategy without changing runtime records or
trust semantics.

## Claim-transformation and repair scenarios

These scenarios exercise agent-facing claim-transformation operations without
requiring a shared corpus:

- mutate a true bounded theorem by dropping one necessary hypothesis, then
  generate and verify a counterexample;
- repair the bounded statement “\(n\)-queens has a solution for every
  \(1\le n\le8\)” to the exact surviving parameter set;
- infer candidate determinant or cap-set patterns from small values while
  keeping the extrapolation explicitly hypothetical.

Symbolic hypothesis deletion is particularly valuable: recent formal
counterexample work generates false statements by removing necessary
hypotheses from verified theorems, then asks for formal counterexamples. Jacobian
can use the same idea on small pure-data domains before Lean integration.

## Knowledge-retrieval scenarios

These scenarios exercise optional corpus-assisted workflows:

- cluster omitted-path failures separately from malformed-path failures;
- cluster magma countermodels by the smallest violating assignment;
- retrieve a verified witness without upgrading neighboring unverified
  experiments;
- enforce a publication-date cutoff during historical retrieval;
- run a claim-repair task with the provider unavailable and report global
  novelty as unknown rather than failing or inventing a result.

## Held-out variants

Public exact fixtures are unsuitable for measuring model reasoning because the
answers are visible. Each public template therefore has a private generator:

- relabel carrier elements, vertices, variables, rows, and columns;
- add irrelevant structure that a shrinker must remove;
- move the decisive omitted clause, path, or certificate row;
- change constants while preserving the same quantifier skeleton;
- generate both true and false neighboring instances;
- insert one tempting witness that is valid for a nearby artifact;
- vary whether the search completes, times out, or reaches a declared bound.

The generator seed is recorded for reproducibility but hidden during a run. The
oracle is generated by a separate exhaustive implementation, frozen, and then
removed from the candidate workspace.

Public exact fixtures can support prescribed-tool conformance cases. Held-out
variants can support autonomous-portfolio evaluations that let agents discover
and compose capabilities; the two case types measure different things and must
not be aggregated together.

Do not rely on random generation alone. Every family contains hand-designed
boundary cases, and every generated instance is checked for:

- a unique intended discriminator or documented alternate routes;
- nontriviality under the baseline condition;
- absence of answer leakage in names and metadata;
- bounded oracle runtime;
- replayability under the declared checker.

## External source decisions

### Use now as design references or import seeds

| Source | Useful artifacts | Decision |
| --- | --- | --- |
| [TPTP/TSTP](https://tptp.org/TPTP/) | Stable problem identifiers, explicit statuses, parameterized problem generators, standard solution records | Adopt its status/reproducibility discipline; import only pinned small cases later |
| [CSPLib](https://github.com/csplib/csplib) | Structured finite constraint specifications, models, and known small result tables | Use formulations as inspiration; independently derive oracles; do not copy files until licensing is clear |
| [Equational Theories](https://github.com/teorth/equational_theories) | Equation lists, finite magma tables, implication/countermodel workflows, Lean-verified outcomes | Strong finite-algebra source; pin an Apache-2.0 release and replay models |
| [AlphaEvolve problem repository](https://github.com/google-deepmind/alphaevolve_repository_of_problems) | Prompts, verification code, initial programs, and evolved programs in notebooks | Later program-search regression corpus; answers are public, so not held out |
| [SMT-LIB](https://smt-lib.org/) | Standard theory semantics, input/output formats, and solver benchmark instances | Later typed-backend compatibility and parser corpus, not public `solver.solve` semantics |

### Use later for formal integration, not public micro-scenario oracles

| Source | Reason |
| --- | --- |
| [miniF2F](https://github.com/openai/miniF2F) | Useful multilingual formal statements, but the repository is archived and public solutions make it a contaminated hidden benchmark |
| [PutnamBench](https://github.com/trishullab/PutnamBench) | Broad formal mathematics and multilingual coverage, but it is intentionally treated as an evaluation set and asks users not to publish proofs |
| [Formal Conjectures](https://github.com/google-deepmind/formal-conjectures) | Excellent claim-ingestion and provenance corpus; open statements lack ground-truth proofs and the maintainers explicitly warn about misformalization |

### Hugging Face datasets

The Dataset Viewer investigation on 2026-07-23 produced these decisions:

| Dataset | Useful fields or metadata found | Decision |
| --- | --- | --- |
| [`Tonic/MiniF2F`](https://huggingface.co/datasets/Tonic/MiniF2F) | 488 rows; names, declared split, informal statement, formal statement, goal, header; MIT-tagged snapshot | Good parser and later Lean compatibility corpus; never a hidden reasoning oracle |
| [`internlm/Lean-Workbook`](https://huggingface.co/datasets/internlm/Lean-Workbook) | 25,214 Viewer rows; NL statement, answer, formal statement, tactic, proof states; Apache-2.0 | Useful proof-state and ingestion stress corpus; pin and replay against its declared Lean environment |
| [`formalanon/semantic-lean-errors`](https://huggingface.co/datasets/formalanon/semantic-lean-errors) | 92 expert-annotated NL/formalization mismatch cases; 24 labeled primary specification errors | Excellent design source for domain-specific claim-validation and transformation traps, but no license was declared; write independent analogues unless permission is obtained |
| [`AgenticCommons/formal-math-autoformalization`](https://huggingface.co/datasets/AgenticCommons/formal-math-autoformalization) | The dataset card declares NL/Lean/proof pairs with per-row provenance, toolchain, mathlib revision, axioms, and a CC0 license | Promising later statement/proof import source; the Dataset Viewer returned server errors during this review, so those fields still require row-level confirmation from a pinned commit |
| [`charliemeyer2000/ai4math-lean`](https://huggingface.co/datasets/charliemeyer2000/ai4math-lean) | Aggregates millions of rows with source, proof, toolchain, and structured verification labels; Apache-2.0 | Useful v1 compatibility corpus; too large and answer-rich for initial or hidden fixtures |
| [`AllenGrahamHart/formal-conjectures-gold`](https://huggingface.co/datasets/AllenGrahamHart/formal-conjectures-gold) | Pinned source/mathlib/toolchain fields, per-row licensing and redistribution status, bundled-oracle metadata | Use its manifest design as precedent; do not expose bundled gold solutions to evaluated agents |

The observations above were made at these Hub revisions:

```text
Tonic/MiniF2F@3a5dceb842b916345a4d7bb7dc4c4c1dbd4b98aa
internlm/Lean-Workbook@2e066e310b2c6d2c27616927ae131f82901c8f1c
formalanon/semantic-lean-errors@16a65901fc7c79ad0f93fc94187c0fcb13ea5cd2
AgenticCommons/formal-math-autoformalization@6ad33573f0f333b42afb51882d4384c939b0c0b3
charliemeyer2000/ai4math-lean@6735c2403f2c57bcd5e9b7aab572872d8265d7d9
AllenGrahamHart/formal-conjectures-gold@aa2ea46926454b2e8ab23103dbccebc5f2f9fc81
```

Every imported Hugging Face fixture records:

```text
dataset ID
Hub commit SHA
config and split
stable row ID and row index
source dataset and original identifier
license and redistribution status
Lean/mathlib versions when applicable
retrieved row digest
local transformed-artifact digest
```

Use Dataset Viewer pagination and filters to select small rows instead of
downloading an entire corpus. If a viewer is unavailable, record that fact and
use a pinned raw file or parquet shard only after its license and digest are
known.

## Suggested repository layout

Current schemas exist. Any future fixture layout may evolve, but public inputs
and hidden oracles must remain separate:

```text
scenarios/
    manifests/
        QNT-CYCLE-001.yaml
        INT-FACTOR-001.yaml
        ...
    public/
        QNT-CYCLE-001/
        ...
    generators/
        path_closure.py
        matrix_kernel.py
        ...
    imported/
        sources.lock

oracle/
    public/
        ...
    private/
        ...
```

The private oracle directory must not be packaged into candidate workspaces,
MCP resources, public container layers, or agent trace exports.

## Portfolio development goals

Develop scenarios across several domains and operation types in parallel.
Prioritize gaps revealed by held-out evaluations, real transcripts, checker
coverage, and backend availability rather than imposing a fixed sequence.
Useful near-term coverage includes:

- direct witnesses and exhaustive certificates in finite logic;
- exact arithmetic and linear-algebra witnesses;
- semantic closure and minimization in graph workloads;
- finite optimization and model enumeration;
- checked claim transformations and representation relations;
- retrieval and formal-proof operations where measured tasks benefit.

Experimental capabilities may be exposed before this catalog is complete.
Scenario results guide discovery, examples, routing, consolidation, and
retirement; they do not create verification authority or a universal tool
taxonomy.
