# Deploy the remote MCP server

[Documentation home](../index.md)

Use STDIO for a single local Codex process. Use Streamable HTTP when ChatGPT or
another remote MCP client must reach Jacobian.

This guide and the checked-in files under `deploy/` are the reproducible
deployment source of truth. Keep host-specific copies, smoke output, and
last-deployed notes outside source control; do not install configuration from
operator scratch space or treat it as current.

The server exposes `math.find` and `math.run`.
Clients may read installed descriptors from `operation://catalog` and inspect
exact contracts before invoking mathematical operations, which remain behind
namespaced operation IDs.

The tool names and schemas do not vary with deployment configuration.
Evaluation harnesses may observe MCP traffic, but they do not add fields to
`math.run` or impose a production research workflow.

## Install from a clone

On a systemd host, the maintained installer turns the committed checkout into
an immutable release, renders the selected ingress, validates systemd and
Caddy, starts the services, and runs the read-only MCP smoke:

```sh
git clone https://github.com/morluto/jacobian.git
cd jacobian
sudo ./deploy/install.sh --mode domain --domain math.example.org
```

Choose one deployment mode:

| Mode | Command | Connector |
| --- | --- | --- |
| Localhost | `sudo ./deploy/install.sh` | `http://127.0.0.1:8765/mcp` |
| Public domain | `sudo ./deploy/install.sh --mode domain --domain math.example.org` | `https://math.example.org/mcp` |
| Tailscale Funnel | `sudo ./deploy/install.sh --mode tailscale` | `https://<tailnet-dns-name>/mcp` |

The host must already provide:

- `uv`, Python 3, Git, and systemd for every mode;
- Caddy for `domain` and `tailscale`;
- a connected Tailscale installation whose tailnet permits Funnel for
  `tailscale`; and
- `elan` when using `--with-lean`. The installer reads the committed
  `lean/lean-toolchain`, installs that exact toolchain through elan, restores
  the manifest-pinned Mathlib cache, and builds the checked-in Lean runtime.
  Under `sudo`, it resolves elan from the root execution path, standard system
  locations, or the invoking user's account-local `~/.elan/bin/elan`.
  The service-readable elan home defaults to `/opt/jacobian/lean/elan`; it does
  not depend on the invoking operator's home directory.

The installer does not pipe remote installation scripts into a shell. Install
those host dependencies through a reviewed package or the upstream documented
procedure. For `domain`, point the domain's DNS at the host and allow inbound
TCP 80 and 443 before deployment so Caddy can obtain and renew its certificate.

Authentication is the default. The first authenticated run creates
`/etc/jacobian-mcp/tokens.json` with mode `0600` and uses that file for the
smoke without printing the credential. Retrieve it explicitly with privileged
access or import it into a secret manager. A subsequent run reuses the secret.
Supply a reviewed multi-tenant file with `--auth-tokens-file PATH` instead.
Anonymous operation requires `--allow-anonymous`; a public anonymous endpoint
additionally requires `--confirm-public-anonymous`, because every reachable
caller shares its operator-chosen tenant and state.

Inspect a complete plan without root or host mutation:

```sh
./deploy/install.sh \
  --mode domain \
  --domain math.example.org \
  --dry-run
```

To serve the pinned Lean CORE and MATHLIB portfolio, select the Lean release
profile explicitly:

```sh
sudo ./deploy/install.sh \
  --mode domain \
  --domain math.example.org \
  --with-lean
```

All code and toolchain paths derive from one installation root. The default is
`/opt/jacobian`; on a new VPS or a host with a separate application volume,
select another absolute path without editing the unit templates:

```sh
sudo ./deploy/install.sh \
  --install-root /srv/math/jacobian \
  --mode domain \
  --domain math.example.org \
  --with-lean
```

This places `releases`, `current`, managed Python runtimes, and the shared elan
home below `/srv/math/jacobian`, and renders both the authenticated unit and the
anonymous override with those exact paths. Reuse the same option on every
upgrade. State under `/var/lib/jacobian-mcp` and secrets under
`/etc/jacobian-mcp` remain separate host data: copy and validate them explicitly
during a VPS migration rather than treating an application release as a backup.
The selected root must be durable across reboots and remain visible through the
hardened systemd unit. The installer rejects `/run`, `/dev/shm`, and their
descendants because they are volatile, and rejects `/home`, `/root`, `/tmp`,
`/var/tmp`, and their descendants because `ProtectHome=true` and
`PrivateTmp=true` hide those host paths from the service. Use an application
path such as `/opt/jacobian` or `/srv/math/jacobian` instead.

With the default installation root, core releases remain under
`/opt/jacobian/releases/<git-sha>`; Lean-enabled releases use
`/opt/jacobian/releases/<git-sha>-lean`. This prevents a core-only
release at one revision from being mistaken for a later Lean-enabled build of
the same source. The Lean build must complete before the release marker or
`/opt/jacobian/current` activation is written.

The installer archives committed `HEAD` to
`/opt/jacobian/releases/<git-sha>`, syncs its locked non-development
environment, and atomically selects it through `/opt/jacobian/current`.
Tracked or staged changes fail closed; commit them or deploy a clean checkout.
Untracked files are not archived. Re-running the same revision is idempotent.
After reviewing and pulling a new revision, run the same command to upgrade.
Before activation, the candidate release performs a read-only compatibility
check against every tenant selected by the configured token file, or against
the selected anonymous tenant. It first records the existing service state,
arms rollback, and stops any existing MCP unit without relying on its active
state, including a unit waiting for an automatic restart. The SQLite/CAS scan
therefore observes a quiescent snapshot. After stopping the writer, the
installer remeasures both the state and the rollback filesystem before scanning
or copying state, so writes received during a long release build cannot stale
the earlier capacity plan. Rollback records active, activating (including
automatic-restart delay), and reloading services as running services and
restarts them if candidate validation or activation fails.
Unsupported, corrupt, unreadable, or migration-incompatible state stops the
deployment without changing the active release and restores the prior service
state. A missing tenant store is valid and remains uncreated until first use.

After the deployment smoke and final permission audit succeed, the installer
keeps the active release and prioritizes the release that was active immediately
before the switch as its rollback. Additional retained slots use the newest
completed releases. Override the total retained count when the host has a
different rollback policy:

```sh
sudo ./deploy/install.sh --retain-releases 3
```

Only recognized release directories with complete revision and profile markers
are eligible for pruning. The active release, incomplete directories, and
operator-created paths are never pruned. A cleanup failure is reported as a
warning after the accepted deployment and does not roll back healthy traffic.
Use `--skip-smoke` only when the endpoint cannot yet be reached from the host,
then run [`deploy/smoke_remote.py`](../../deploy/smoke_remote.py) before
advertising it.

`--dry-run` is a host preflight as well as a plan. It resolves the required
`uv`, systemd, Caddy, Tailscale, and optional elan executables and checks free
space at the installation filesystem. A new Lean release requires at least 12
GiB free; a new core release requires at least 2 GiB. Existing complete release
directories do not require that build-space reserve. The rollback temporary
filesystem must also retain the full current state snapshot plus 64 MiB for
configuration and metadata; when it shares the installation filesystem, both
reserves are added before deployment work begins. A non-root dry-run that
cannot read an existing protected state directory reports rollback capacity as
unverified and tells the operator to repeat the preflight with `sudo`; the real
deployment never proceeds without the measurement.

## Migrate to another VPS

Application releases are reproducible; persistent tenant state and
authentication secrets are separate host data. Use this sequence when the new
VPS must preserve them. The maintained authenticated unit uses
`/var/lib/jacobian-mcp`; the anonymous test override uses
`/var/lib/jacobian-mcp-test`:

1. Install Git, Python, `uv`, systemd, the selected ingress dependencies, and
   elan for `--with-lean` on the destination. Check out the exact revision to be
   deployed and run the final installer command with `--dry-run`.
2. Securely transfer the existing token file when authentication is enabled.
   Do not place it in the repository, shell history, or an unencrypted archive.
3. Run the installer once on the destination with `--mode local`, the final
   install root, profile, tenant options, and `--skip-smoke`. For authenticated
   operation, pass the transferred file with `--auth-tokens-file`. Omit
   ingress-only options such as `--domain` until the final deployment. Local
   mode provisions the service users, immutable release, units, and destination
   directories while disabling Caddy and Funnel. Do not use `--mode domain` or
   `--mode tailscale` for this staging pass: `--skip-smoke` skips validation but
   still starts the selected services.
4. Stop `jacobian-mcp.service` on the destination. Stop it on the source, or
   quiesce all writers and take a storage-level snapshot, before the final copy.
   Copy the complete selected state root, including `metadata.sqlite3`, blobs,
   manifests, and tenant directories. Do not copy a live SQLite database and
   its blobs independently.
5. On the destination, restore state ownership to `jacobian:jacobian`. Keep
   `/etc/jacobian-mcp` owned by root with mode `0700` and its token file at mode
   `0600`. Rebuildable `cache/lean-declarations` data may be omitted. Restore
   tenant state as real directories and regular files, not symbolic links;
   links can resolve outside the filesystem view allowed by the hardened unit.
6. Choose the final activation sequence for the ingress mode while the source
   service remains stopped:

   - For `--mode domain`, run the final domain-mode command once with
     `--skip-smoke` after state restoration. Switch DNS to the destination, wait
     until the domain resolves to it from the destination host, then rerun the
     same command without `--skip-smoke`. This lets Caddy establish the
     destination endpoint before the revision-bound smoke follows the public
     hostname.
   - For `--mode tailscale`, run the final command without `--skip-smoke`, then
     update clients if the destination node has a different Tailscale DNS name.
   - For `--mode local`, rerun the final local command without `--skip-smoke`.

   The candidate-state preflight reads the token mapping while privileged,
   drops permanently to the `jacobian` service identity, and then checks every
   selected tenant. In that identity it verifies that SQLite and its runtime
   files are readable and writable, every persisted manifest and payload blob
   reference resolves to a local regular file whose SHA-256 matches its content
   address, and each manifest remains bound to its committed metadata, parents,
   descriptors, and mathematical object digest. Existing blob files must be
   readable and match their content addresses, and the persisted blob quota must
   match the full restored CAS tree without a pending recovery marker. SQLite
   integrity, foreign keys, required schema, and the state-format row must agree
   with the migration ledger. The tenant, staging, blob, and existing
   blob-prefix directories must be readable, writable, and traversable. It may
   report `MISSING` only when
   the selected tenant directory does not exist; an existing directory with
   missing or uninitialized metadata is a blocking partial restore. Every
   selected tenant must report `MISSING`, `COMPATIBLE`, or a supported
   `MIGRATION_PENDING` state before activation. Do not bypass an `UNSUPPORTED`,
   `INCOMPATIBLE`, `CORRUPT`, or `UNREADABLE` result; an access failure means
   destination ownership or mode restoration is incomplete.
7. Verify `deployment://identity`, the two-tool surface, catalog policy,
   required capabilities, and any provider-specific smoke before updating
   remaining client configuration. Keep the source VPS and its unchanged state
   available, but stopped, until the destination has passed these checks.

Current Jacobian writes state revision 12 and accepts revision 11 for an
in-place migration. Earlier revisions have no direct import bridge; retain the
old state unchanged with a compatible checkout and start fresh state on current
Jacobian as described in
[Persistent state format](../reference/state-format.md).

Migration reminders:

- `/etc/jacobian-mcp/tokens.json` is not part of an immutable release. Copying
  only `/opt/jacobian` will leave existing authenticated clients unable to
  connect.
- An anonymous tenant's directory key is derived from its tenant ID. Reuse the
  exact `--anonymous-tenant-id` to preserve copied state, or deliberately choose
  a new ID for disposable fresh state.
- A Tailscale Funnel connector is derived from the destination node's Tailscale
  DNS name and may differ after a VPS move. Update clients explicitly, or use a
  stable operator-owned domain and perform a DNS cutover.
- Never allow the source and destination to accept writes against independently
  copied state after the final snapshot. Jacobian does not merge divergent
  stores.

## Create the auth secret

Create a JSON file outside the repository:

```json
{
  "tokens": [
    {
      "tenant_id": "research-team-a",
      "token": "replace-with-at-least-32-random-characters",
      "scopes": ["jacobian:use"]
    }
  ]
}
```

Treat this as a secret. Rotate a token by replacing the file and restarting the
server. Do not put tokens in prompts, source control, command-line arguments,
or Jacobian artifacts.

## Start Streamable HTTP

```sh
uv run jacobian-remote-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --path /mcp \
  --state-dir /var/lib/jacobian \
  --max-tenant-runtimes 32 \
  --tenant-idle-timeout-seconds 900 \
  --auth-tokens-file /run/secrets/jacobian-tokens.json \
  --public-base-url https://math-tools.example.org
```

Put a TLS-terminating reverse proxy in front of `127.0.0.1:8000`. The public
URL must route `/mcp` without stripping the path. Each authenticated subject is
mapped to a separate hashed directory below
`/var/lib/jacobian/tenants/`. The server retains at most 32 tenant runtimes by
default. At the limit, it evicts the least-recently-used inactive runtime; it
never evicts a runtime with an active request, and returns a bounded admission
error when every slot is active. Inactive runtimes expire after 900 seconds by
default. Expiry is checked on the next tenant acquisition rather than by a
background reaper. Set `--max-tenant-runtimes` and
`--tenant-idle-timeout-seconds` to match the instance's memory budget.

For a disposable local transport test only:

```sh
uv run jacobian-remote-mcp \
  --transport streamable-http \
  --max-tenant-runtimes 32 \
  --tenant-idle-timeout-seconds 900 \
  --allow-anonymous \
  --anonymous-tenant-id local-smoke-2026-07
```

`--anonymous-tenant-id` is fixed by the operator, never selected from a request.
Give every independently operated anonymous test endpoint a different value so
their artifacts do not share one state
directory. This is namespace isolation, not authentication: every caller that
can reach the same endpoint still shares that endpoint's tenant. Do not use
anonymous mode for a production or sensitive endpoint. A deliberately public,
time-bounded connector interoperability test must use disposable non-sensitive
state, external resource limits and monitoring, and a scheduled removal time.

## Use the maintained VPS topology

The checked-in VPS baseline separates the mathematical process from public
ingress:

```text
remote MCP client
  → Tailscale Funnel :443
  → Caddy 127.0.0.1:8766
  → jacobian-mcp 127.0.0.1:8765/mcp
  → persistent tenant state
```

The corresponding maintained files are:

| File | Purpose |
| --- | --- |
| [`deploy/install.sh`](../../deploy/install.sh) | Idempotent clone-to-systemd installer for localhost, public-domain, and Funnel modes |
| [`deploy/systemd/jacobian-mcp.service`](../../deploy/systemd/jacobian-mcp.service) | Authenticated backend baseline with persistent state and a versioned checkout path |
| [`deploy/systemd/jacobian-mcp-anonymous.conf`](../../deploy/systemd/jacobian-mcp-anonymous.conf) | Explicit test-only anonymous override with a separate state root |
| [`deploy/systemd/jacobian-caddy.service`](../../deploy/systemd/jacobian-caddy.service) | Local Caddy process and writable data directories |
| [`deploy/systemd/jacobian-funnel.service`](../../deploy/systemd/jacobian-funnel.service) | Restores Funnel after boot or a Tailscale restart |
| [`deploy/caddy/Caddyfile`](../../deploy/caddy/Caddyfile) | Path routing and credential-safe logging |
| [`deploy/smoke_remote.py`](../../deploy/smoke_remote.py) | Read-only handshake, version, tool, catalog, policy, and discovery gate |

The installer is the default clone-to-host path. The units remain reviewable
templates for operators who need to integrate with existing provisioning.
It builds each environment at its final
`/opt/jacobian/releases/<revision>/.venv` path so console scripts and local
project references never retain a temporary build path. uv-managed Python
runtimes live under `/opt/jacobian/python`, outside root's home directory. The
installer checks the final console-script shebang, resolved Python path, and
execution as the `jacobian` service user before atomically selecting the
release through `/opt/jacobian/current`. It disables Python bytecode writes in
its root-run probes and audits the release, managed Python, and optional Lean
toolchain as the service user both before activation and after smoke. This keeps
runtime readability independent of the operator's `umask` and makes a
post-build permission regression trigger rollback instead of accepting a
partially private release.
Before copying them manually, replace `math-tools.example.org`, verify the
service accounts, Caddy binary, and `/opt/jacobian/current` checkout, and decide
between the static-token baseline and the anonymous test override. Keep each
release in an immutable checkout and atomically move
`/opt/jacobian/current` to the selected release; do not point a long-running
service at a dirty developer worktree.

For manual provisioning, install reviewed copies and validate them before
enabling traffic:

```sh
sudo install -d -m 0700 /etc/jacobian-mcp
sudo install -m 0600 /path/to/jacobian-tokens.json \
  /etc/jacobian-mcp/tokens.json
sudo install -d -m 0755 /etc/caddy-jacobian
sudo install -m 0644 deploy/caddy/Caddyfile /etc/caddy-jacobian/Caddyfile
sudo install -m 0644 deploy/systemd/jacobian-mcp.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/jacobian-caddy.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/jacobian-funnel.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/jacobian-mcp.service
sudo systemd-analyze verify /etc/systemd/system/jacobian-caddy.service
sudo systemd-analyze verify /etc/systemd/system/jacobian-funnel.service
sudo caddy validate --config /etc/caddy-jacobian/Caddyfile --adapter caddyfile
sudo systemctl daemon-reload
```

For an isolated anonymous test endpoint, install the reviewed override only
after changing its tenant ID. If the test is intentionally paired with Funnel,
record that it is public and shared, keep its state non-sensitive, monitor it,
and remove the override when the test window closes:

```sh
sudo install -d -m 0755 /etc/systemd/system/jacobian-mcp.service.d
sudo install -m 0644 \
  deploy/systemd/jacobian-mcp-anonymous.conf \
  /etc/systemd/system/jacobian-mcp.service.d/anonymous.conf
sudo systemctl daemon-reload
```

Removing that drop-in does not authenticate existing anonymous state. Stop the
service, remove the override, configure the token credential, and deliberately
choose whether to retain or archive the separate test state root.

## Filter reverse-proxy logs

Access-log filtering does not cover a reverse proxy's own warning and error
logs. Those records can include a structured copy of request headers when an
upstream disconnects or times out. Configure both the proxy's global/runtime
logger and its access logger.

The checked-in [Caddyfile](../../deploy/caddy/Caddyfile) is the deployment
baseline for the localhost ports used by the hosted test service. It deletes
authorization, cookie, OpenAI session/subject, and Tailscale identity headers
from both log paths. `Traceparent` is reduced to Caddy's eight-hex-character
SHA-256 correlation digest; Jacobian emits the same digest in its bounded
`MCP operation attempt` record.

Validate and reload a copied configuration before changing live traffic:

```sh
caddy validate --config /etc/caddy-jacobian/Caddyfile --adapter caddyfile
caddy reload --config /etc/caddy-jacobian/Caddyfile --adapter caddyfile
```

Do not enable Caddy's `debug` or `log_credentials` options on a hosted
connector. Retention and downstream log aggregation must preserve the same
field-deletion boundary.

## Redeploy and prove what is running

Before a restart, record the currently selected release, package version,
service start time, catalog policy, and a passing smoke result. A catalog can
remain unchanged across code releases, so catalog membership alone is not a
deployment-version check.

Prepare and validate the new immutable checkout first:

```sh
uv sync --locked --dev
make check
uv run jacobian-remote-mcp --version
```

After moving `/opt/jacobian/current` to that checkout, restart only the backend
unless an ingress file also changed:

```sh
sudo systemctl restart jacobian-mcp.service
sudo systemctl is-active jacobian-mcp.service
sudo systemctl show jacobian-mcp.service \
  --property=MainPID,ExecMainStartTimestamp,FragmentPath,WorkingDirectory
```

There is intentionally no unauthenticated generic `/health` endpoint. Run the
read-only MCP smoke from the exact checkout being deployed; by default it
requires the remote handshake version to equal that checkout's installed
Python distribution version:

```sh
uv run python deploy/smoke_remote.py \
  https://math-tools.example.org/mcp \
  --expect-policy-profile DEFAULT \
  --require-operation graph.construct.explicit
```

For a token-protected endpoint, set `JACOBIAN_MCP_BEARER_TOKEN` in the smoke
process environment without placing it on the command line. The script does
not print the token and disables ambient proxy settings. It performs no state
writes or operation invocations.

Confirm the ingress route independently:

```sh
sudo systemctl is-active jacobian-caddy.service
sudo systemctl is-active jacobian-funnel.service
tailscale funnel status --json
```

Then inspect bounded logs for the new backend start and smoke requests:

```sh
sudo journalctl -u jacobian-mcp.service --since "10 minutes ago" --no-pager
sudo journalctl -u jacobian-caddy.service --since "10 minutes ago" --no-pager
```

Do not report a deployment complete until the service version, two-tool
surface, catalog policy, required operations, and bounded discovery response
all pass. Run deeper operation-specific smoke checks only for providers
changed by the release.

`--with-lean` adds `lean.check`, `lean.proof_state.apply_tactic`,
`lean.term.apply`, and `lean.retrieve.premises` to the required catalog set. It
then runs `deploy/smoke_lean.py`, which writes disposable tenant artifacts while
checking CORE and MATHLIB proof acceptance, a real MATHLIB declaration search,
plus accepted and rejected tactic transitions. The service also advertises the
root-owned `.git-revision` captured at process start through
`deployment://identity`; use that resource when an evaluation or operator
record must bind observations to the exact deployed checkout. This deeper smoke
is deliberately separate from the general read-only deployment smoke.

## Roll back without rewriting the repository

Keep the previous immutable checkout until the new deployment is accepted. If
the backend or smoke gate fails:

1. stop new rollout work and retain the failing logs;
2. move `/opt/jacobian/current` back to the prior checkout;
3. restart `jacobian-mcp.service`;
4. rerun the smoke with `--expect-version` set to the prior package version; and
5. confirm Caddy and Funnel still route to the restored backend.

The state root is persistent and is not rolled back with code. Back it up before
a release that changes stored artifact formats, and verify
backward compatibility before starting older code against newer state. Do not
use `git reset --hard` as a deployment or rollback mechanism.

Lean results are cached only for an exact content-addressed certificate and
the currently active checker implementation digest. The bounded in-memory cache holds 128
entries; a changed proof, statement, environment, checker, or authorization
state cannot reuse an entry. Before advertising a deployment, invoke a deployed
smoke check with `statement: "True"`, `proof: "by trivial"`, and
`environment: "MATHLIB"`.

Lean declaration discovery keeps its rebuildable catalog indexes below
`<state-root>/cache/lean-declarations`; the installer derives that location from
the configured artifact-store root, so no checkout, release, home-directory, or
VPS-specific path is embedded in the runtime. Index filenames bind the cache
format, Lean environment, and measured pinned content identity while excluding
the Mathlib checkout's absolute deployment root. Each index carries a complete
row count and content digest; a truncated or modified copy is ignored and
rebuilt. Exact typed search and inspect payloads also use a 128-entry, 32 MiB
in-memory LRU and are discarded with the runtime. Copy the state root during a
VPS migration. The declaration cache directory may instead be omitted or
removed while the service is stopped; Jacobian rebuilds it from the pinned
Lean/Mathlib environment.

## Container deployment

Build the repository image:

```sh
make container-image IMAGE=jacobian:local
```

Run it with a persistent state volume and read-only secret mount:

```sh
docker run --rm -p 127.0.0.1:8000:8000 \
  -v jacobian-state:/var/lib/jacobian \
  -v "$PWD/tokens.json:/run/secrets/jacobian-tokens.json:ro" \
  jacobian:local \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000 \
  --state-dir /var/lib/jacobian \
  --auth-tokens-file /run/secrets/jacobian-tokens.json \
  --public-base-url https://math-tools.example.org
```

The initial static-token verifier is suitable for controlled deployments. A
hosted service should integrate its OAuth/OIDC verifier and map the validated
subject to the same tenant-routing interface.

## Operational boundaries

- Back up the state volume; artifacts live there.
- Run one Jacobian process per state root until a lease model is implemented.
- Record the deployed git commit and the MCP-advertised package version. Keep
  host-local deployment notes outside source control; they supplement, rather
  than replace, this runbook and the checked-in templates.
- Apply CPU, memory, filesystem, and network policy outside Jacobian.
- Synchronous SAT and SMT solver requests are capped at 150 seconds so a
  structured fail-closed response precedes common remote-client deadlines.
  Client cancellation terminates the bound solver process group. Partition
  larger searches instead of holding one HTTP request open.
- Do not interpret HTTP success, solver completion, or an MCP response as a
  verified mathematical result.
- Use the one-line `MCP operation attempt` records for operational counts.
  The attempt record distinguishes `COMPLETED`, `TIMEOUT`, `CANCELLED`, and
  `ERROR` and retains only bounded status/provenance fields and argument
  digests.
