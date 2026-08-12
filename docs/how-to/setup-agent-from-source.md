# Configure an agent from a source checkout

Use the source bootstrap when an agent must run the exact code in a Jacobian
clone rather than the latest published Python package. From the repository
root, run:

```sh
./scripts/setup-agent --client codex --profile core --yes
```

Replace `codex` with `claude`, `cursor`, `gemini`, or `opencode`; repeat
`--client` to configure several clients, or use `--all`. The generated client
entry starts the server directly from the absolute clone and state paths:

```text
uv run --project /absolute/path/to/jacobian --locked --no-sync \
  jacobian-mcp --state-dir /absolute/path/to/jacobian/.jacobian
```

The bootstrap requires the uv version in [`.uv-version`](../../.uv-version),
CPython 3.12 or 3.13, Git, and Node.js 18 or newer. The tested binary-install
platform is glibc Linux x86-64; other platforms require a compatible wheel for
every mandatory Python backend. It requires a clean checkout,
performs a locked sync, initializes the state directory, and audits the exact
Git revision, package version, catalog digest, and provider runtimes before it
writes any client configuration. The audit is saved as
`.jacobian/bootstrap-doctor.json` unless `--state-dir` selects another path.
An alternate state directory inside the checkout must already be ignored by
Git; the bootstrap rejects a path that would make its clean source identity
dirty during initialization. A state directory outside the checkout is also
supported.
Bootstrap uses the provider-bearing base `PATH` to discover Lean and external
proof tools. It resolves each accepted tool to an absolute path and records its
provider identity; product execution does not repeat ambient `PATH` discovery.
Immediately before an invocation, Jacobian remeasures the recorded provider and
rejects an identity mismatch. A provider that legitimately launches nested
tools receives a constructed `PATH` containing only its authorized toolchain
directories.

The generated launcher preserves the bootstrap environment needed to reproduce
doctor's discovery. uv prepends the project virtual environment to that base
path for doctor and MCP startup, but child-process policy still controls the
environment used for provider execution. If bootstrap inherits a custom
`UV_PROJECT_ENVIRONMENT`, it resolves that path absolutely and records it in
the launcher; a checkout-local custom environment must already be ignored by
Git. Relative entries in the inherited `PATH` are resolved against the
bootstrap working directory so later GUI working directories cannot change
bootstrap discovery. The `lean` profile likewise records a nondefault
`ELAN_HOME` and any `JACOBIAN_LEAN_RUNTIME` override, so a GUI restart uses the
toolchain home and mathlib checkout that doctor audited.

The bootstrap configures the selected client's MCP entry only. Jacobian does
not install prompts, skills, or client-specific mathematical workflows.

## Profiles

| Profile | Installed or checked surface |
| --- | --- |
| `core` | Complete locked Python backend stack: SymPy, NetworkX, Z3, Python-FLINT, and cvc5 |
| `lean` | `core` plus a build of the pinned `lean/lean-toolchain` project; `elan`/`lake` must already be on `PATH` |
| `external-proof` | `core` plus fail-closed availability checks for pinned CaDiCaL, DRAT-trim, and Carcara executables |

The `external-proof` profile does not download or trust native executables on
the user's behalf. Their exact version and provenance contracts are defined in
the [SAT artifact reference](../reference/capabilities/sat-smt/sat-artifacts.md) and
[SMT artifact reference](../reference/capabilities/sat-smt/smt-artifacts.md). If a required provider
is absent or has the wrong identity, doctor fails before client configuration.
The `lean` profile likewise uses the repository's pinned toolchain and manifest
instead of a floating Lean installation.

These profiles configure an agent client against a source checkout. For
ordinary contributor work that only needs to run the test suite, the
`make setup PROFILE=core` quick path installs the same complete Python backend
stack without writing client configuration; see
[CONTRIBUTING.md](../../CONTRIBUTING.md). Optional native backend installation is described in
[Install native and formal providers](install-native-and-formal-providers.md).

Add `--dev` when the clone also needs the locked development group. Use
`--dry-run` to inspect every sync, init, configuration, and doctor command
without changing the environment, state directory, or client files. Repeating
the same command is idempotent. Client edits are one transaction: if any write
fails, changed earlier files from that run are restored. No-op clients are not
part of rollback, and a concurrently changed config is left untouched instead
of being overwritten. Declining the confirmation exits without printing the
ready message. `jacobian remove` removes only the Jacobian entry and preserves
unrelated client settings. Client config symlinks are rejected rather than
replaced; update the link target directly or temporarily use a regular config
file.

After pulling a new commit, rerun the bootstrap. The client already points at
the same source path, while the locked sync and doctor refresh its environment
and recorded identity before the next agent session.
