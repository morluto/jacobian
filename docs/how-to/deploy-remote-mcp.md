# Deploy the remote MCP server

[Documentation home](../index.md)

Use STDIO for a single local Codex process. Use Streamable HTTP when ChatGPT or
another remote MCP client must reach Jacobian.

The server exposes `capability.describe`, `capability.invoke`, and the three
direct `workspace.*` tools. Clients may read installed descriptors from
`capability://catalog` and inspect exact contracts before invoking mathematical
operations, which remain behind namespaced capability IDs. Workspace state is
subject-bound operational data and never becomes mathematical assurance.

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
uv run jacobian-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --path /mcp \
  --state-dir /var/lib/jacobian \
  --max-tenant-kernels 32 \
  --auth-tokens-file /run/secrets/jacobian-tokens.json \
  --public-base-url https://math-tools.example.org
```

Put a TLS-terminating reverse proxy in front of `127.0.0.1:8000`. The public
URL must route `/mcp` without stripping the path. Each authenticated subject is
mapped to a separate hashed directory below
`/var/lib/jacobian/tenants/`. The server retains at most 32 tenant kernels by
default. Existing tenants remain available at the limit; new tenants receive a
bounded admission error. Set `--max-tenant-kernels` to match the instance's
memory budget.

For a disposable local transport test only:

```sh
uv run jacobian-mcp \
  --transport streamable-http \
  --max-tenant-kernels 32 \
  --allow-anonymous \
  --anonymous-tenant-id local-smoke-2026-07
```

`--anonymous-tenant-id` is fixed by the operator, never selected from a request.
Give every independently operated anonymous test endpoint a different value so
their workspaces, research episodes, and artifacts do not share one state
directory. This is namespace isolation, not authentication: every caller that
can reach the same endpoint still shares that endpoint's tenant. Do not expose
anonymous mode to an untrusted network.

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
`MCP capability attempt` record.

Validate and reload a copied configuration before changing live traffic:

```sh
caddy validate --config /etc/caddy-jacobian/Caddyfile --adapter caddyfile
caddy reload --config /etc/caddy-jacobian/Caddyfile --adapter caddyfile
```

Do not enable Caddy's `debug` or `log_credentials` options on a hosted
connector. Retention and downstream log aggregation must preserve the same
field-deletion boundary.

## Warm the Mathlib profile when serving `lean.check`

Set `JACOBIAN_LEAN_WARMUP=1` on a host that serves `lean.check`. Jacobian then
checks a small pinned Mathlib theorem in the background when each tenant kernel
is first used. This warms Lean and filesystem caches without delaying MCP
startup.

Lean results are cached only for an exact content-addressed certificate and
the currently active checker digest. The bounded in-memory cache holds 128
entries; a changed proof, statement, environment, checker, or authorization
state cannot reuse an entry. `capability.describe` for `lean.check` reports the
cache policy and the MATHLIB warm-up state (`RUNNING`, `HEALTHY`, or
`UNHEALTHY`). Before advertising a deployment, wait for `HEALTHY` and invoke a
deployed smoke check with `statement: "True"`, `proof: "by trivial"`, and
`environment: "MATHLIB"`. An unhealthy deployment must not recommend the
MATHLIB profile.

## Container deployment

Build the repository image:

```sh
docker build -t jacobian:local .
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

- Back up the state volume; artifacts, workspaces, and research episodes live
  there.
- Run one Jacobian process per state root until a lease model is implemented.
- Apply CPU, memory, filesystem, and network policy outside Jacobian.
- Synchronous SAT and SMT solver requests are capped at 150 seconds so a
  structured fail-closed response precedes common remote-client deadlines.
  Client cancellation terminates the bound solver process group. Partition
  larger searches instead of holding one HTTP request open.
- Do not interpret HTTP success, solver completion, or an MCP response as a
  verified mathematical result.
- Use the one-line `MCP capability attempt` records for operational counts.
  Completed research episodes intentionally omit failed attempts, while the
  attempt record distinguishes `COMPLETED`, `TIMEOUT`, `CANCELLED`, and `ERROR`
  and retains only bounded status/provenance fields and argument digests.
