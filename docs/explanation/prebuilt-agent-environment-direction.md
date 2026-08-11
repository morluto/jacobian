# Digest-pinned prebuilt agent environments (issue #504)

Status: implementation direction for Harbor agent evaluations.

## Problem

Observation jobs still set `environment.force_build = true` and many agent
images install Codex or shared tools during trial setup. That spreads
environment construction across task bundles, slows evaluation, and pressures
hosts toward ad-hoc caching outside Harbor's identity model.

## Desired boundary

```text
GitHub Actions (or operator CI)
  ├─ builds and publishes a shared Codex agent image by content digest
  └─ publishes Jacobian MCP image by revision/digest

Harbor trial
  ├─ starts the digest-pinned prebuilt agent image (force_build=false when pinned)
  ├─ uploads task-visible files into /app
  └─ attaches Jacobian only for treatment via native MCP configuration

Jacobian tooling
  └─ records agent-image digest and MCP-server identity in observation evidence
```

Formal-provider tasks use the same identity discipline but a different trust
boundary: the agent image may carry an exploratory provider, while a separate
digest-pinned checker image replays the submitted source artifact. A successful
provider trace alone remains diagnostic evidence, not independent verification.
The maintained Lean bases and their publication evidence are documented in
[benchmark contracts](../reference/evaluations/benchmark-contracts.md#reusable-evaluation-images).

## Repository hooks (current)

- Task environments resolve through
  [`benchmarks/environment-profiles.toml`](../../benchmarks/environment-profiles.toml).
- Profile `core-python-codex` already documents baking Codex into the task image
  at build time so trials need no package network for the agent runtime.
- Observation jobs under `benchmarks/datasets/*/jobs/jacobian-observation*.json`
  still set `force_build: true` today for reproducibility of local trees.

## Implementation plan

1. **Image contract**
   - Define a published image name such as
     `ghcr.io/<org>/jacobian-codex-agent` with an immutable digest pin.
   - Bake: Python 3.12, Codex CLI (pinned version), common tools (ripgrep,
     git, curl as required), no task-specific files.
2. **Profile**
   - Add `core-python-codex-prebuilt` (or promote `core-python-codex`) with
     `agent_image = "...@sha256:..."` and `allow_apt = false`.
   - Keep verifier images separate and digest-pinned.
3. **Jobs**
   - Observation jobs may set `force_build: false` only when the prebuilt
     digest is recorded in the job config and observation evidence schema.
   - Fail closed if the running image digest does not match the pinned digest.
4. **Evidence**
   - Extend observation evidence to require `agent_image_digest` and
     `jacobian_image_digest` on every trial.
5. **CI**
   - Workflow builds/pushes the shared agent image on a controlled path
     (main or release), never from untrusted PR heads without review.

## Non-goals

- Replacing Harbor's runner or inventing a second task format.
- Baking Jacobian into the agent image (treatment attaches MCP separately).
- Weakening offline `no-network` task policy.

## Acceptance

- A treatment observation trial starts without `apt-get` or Codex install in
  the task Dockerfile when using the prebuilt profile.
- Trial evidence binds exact agent and Jacobian image digests.
- Control/treatment pairs share the same agent image digest.
