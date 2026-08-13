# Paired evaluation and scoring

## Model-call gate

Require all items before running model arms:

- exact source and artifact are new and unreserved;
- source gate passes;
- oracle matches the source and is input-hashed;
- current-main producer/verifier replay is complete;
- the comparison asks a question not answered deterministically;
- no issue or PR already owns the proposed root cause;
- control and treatment isolation is real;
- model, reasoning, temperature, output limit, timeout, execution budget, environment, revision, prompt hash, and suite hash are frozen;
- checker availability is captured before expensive materialization;
- JSONL, elapsed-time, token, and visible-byte accounting work;
- the operator authorized the applicable cost boundary.

## Arm isolation

The control sees no Jacobian MCP server, skill, catalog, tool description,
routing hint, or artifact. The treatment sees exactly the frozen Jacobian
surface. Keep unrelated tools and settings identical. Use fresh workspaces and
contexts.

## Observable scoring

Score each arm independently:

1. final mathematical correctness;
2. evidence or certificate validity;
3. exact input binding;
4. source fidelity;
5. scope calibration and completeness;
6. assurance calibration;
7. discovery and contract inspection;
8. useful executions;
9. invalid, cancelled, redundant, and post-terminal calls;
10. recovery, stopping, and use of evidence in the final answer;
11. elapsed time, total and uncached tokens, and model-visible bytes.

## Assurance rules

- A producer result is not independently verified.
- A local calculation can be correct without being Jacobian-verified.
- A checker record must bind the input, result, operation version, and checker authority.
- A correct scalar with a missing required certificate is incomplete.
- Failure to find evidence is not evidence of nonexistence.
- Infrastructure-invalid runs do not count as mathematical evidence.

Accept alternative valid strategies. Do not prescribe one tool sequence or
score hidden reasoning.

## Evidence retention

Save raw discovery and invocation responses as JSON or JSONL, including safe
failures. Record hashes for the frozen input, probe, raw output, and report. A
prose reconstruction of terminal output is not independently auditable.
