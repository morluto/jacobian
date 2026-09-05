# Scale and backend investigation

Treat a timeout, slow trace, or static complexity finding as a candidate, not
proof that an operation or backend should change. Start from a representative
exact workload supplied by an agent trajectory, primary-source obligation,
benchmark, or deterministic boundary case. Do not infer mathematical-kernel
performance from a repository-wide static scan alone.

First separate the stable mathematical postcondition from the release's
admitted execution envelope and complete the execution-envelope review in
[public operation admission](../../../../docs/reference/public-operation-admission.md#execution-envelope-review). Keep the semantic domain broad. Before
retaining a small fixed cap, compare result-sensitive admission, compact exact
representations, maintained specialist backends, and exact algorithms suited
to different input regimes. A backend restriction is not a mathematical domain
restriction, and large scalar inputs should remain admissible when predicted
work, intermediates, memory, and exact output are small. When no sharper safe
envelope is established, classify and document the fixed cap as a conservative
fallback and state what evidence could raise it.

Freeze the workload before comparing implementations. Record the operation ID,
repository revision, canonical request or input digest, coefficient domain and
parent, relevant mathematical size quantities, current backend and version,
candidate backend and version, execution environment, and the defining
invariant or independent oracle used to compare results.

Derive the operation's mathematical budget before measuring wall time. Depending
on the operation, relevant quantities include candidate tuples, search nodes,
terms, degree, coefficient height, matrix dimensions, witness count, and
worst-case serialized output. Keep these quantities separate unless a documented
formula proves that one aggregate conservatively bounds them. A wall-time limit
is an execution safety net, not the mathematical work bound.

Compare current and candidate implementations on identical canonical inputs,
mathematical options, completion criteria, and resource limits. Report these
evidence classes separately:

- static observations from source or backend capabilities;
- mathematical estimates derived from stated assumptions;
- measurements tied to the frozen workload and environment;
- behavioral or invariant checks establishing comparable exact results; and
- unknowns, timeouts, unavailable backends, and incomplete coverage.

Exercise accepted and rejected boundaries, relevant algorithm or
representation crossover points, and realistic source-backed scale cases.
Check every regime with the same defining invariant or independent oracle; a
fast path that returns a weaker or lossy value does not widen the operation.

Classify the outcome before recommending work:

- **admission:** the admitted execution envelope uses an unproved, excessively
  coarse, or misleading work or output proxy;
- **representation/output:** computation succeeds but canonical conversion,
  repeated source context, or output shape dominates the useful bound;
- **scale/backend:** a maintained engine materially improves completion or cost
  without changing the public mathematical contract;
- **operation:** the stable mathematical postcondition itself is missing or
  incorrectly scoped;
- **composition/reasoning:** factor selection, partitioning, saturation,
  branch exploration, or stopping policy belongs to the caller; or
- **no gap:** the evidence does not support a useful contract or implementation
  change.

Prefer a private backend replacement or owner-local algorithm improvement when
the public contract stays the same. Do not turn a difficult benchmark into a
problem-specific public solver, branch checker, skill, or workflow operation.
Retain only a small exact regression or benchmark fixture when it represents a
recurrent boundary and has a stable reconstruction or correctness invariant.

For a durable performance finding, keep a compact coverage ledger containing:
operation and revision, workload digest and provenance, mathematical size
quantities, backend versions, declared resource limits, completion status,
exact output size, timing environment, and invariant-check outcome. The ledger
prevents a single dramatic trace or convenient fixture from being reported as
general coverage.
