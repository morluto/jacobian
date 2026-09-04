---
name: Bug report
about: Report incorrect mathematics, rejected useful inputs, or runtime and tooling failures
title: "[Bug]: "
labels: ["bug", "status: needs-triage"]
assignees: []
---

**Priority**
<!-- Select one: P0 (critical), P1 (high), P2 (medium), P3 (low) -->

**Area**
<!-- Which area is affected? math, native API, cli, mcp, docs, ci, deps, security -->

**Component**
<!-- Which part of Jacobian is affected? (e.g., math domain, CLI, MCP adapter, docs) -->

**Observed behavior**
<!-- What happened? Include exact error messages or unexpected output. -->

**Expected behavior**
<!-- What should have happened instead? -->

**Failure stage and owner**
<!-- Optional diagnosis: check the earliest confirmed failing stage and name
the owner if known. Leave unknown rather than guessing from the error alone. -->
- [ ] request parsing
- [ ] request bounds/admission
- [ ] backend or kernel execution
- [ ] result construction
- [ ] transport projection
- Owner:

**Typed outcome**
<!-- What typed result or error should have been returned? -->

**Reproduction**
<!-- Paste the smallest useful exact math.run request or native API call, its
result/error, and a nearby supported case if known. State the violated identity
or expected mathematical property. For performance or scale, include a useful
input, observed work/time, budgets, and whether serialization is included. A
timeout does not establish nonexistence. Do not include credentials. -->

**Environment**
- Jacobian package version or source revision:
- Installed MCP/catalog identity, if different or unverified:
- Python version:
- OS:

**Additional context**
<!-- Screenshots, logs, or other relevant information. -->
