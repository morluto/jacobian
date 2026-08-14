# Deploy the stateless remote MCP server

[Documentation home](../index.md)

Deploy Jacobian as one immutable Python service artifact. The application owns
mathematical execution and health reporting; the deployment platform owns
provisioning, configuration, rollout, rollback, TLS, and process supervision.
The checked-in files under `deploy/` are examples for those platform boundaries,
not an installer or release transaction engine.

## Build an immutable artifact

Select an exact release or revision, install its locked Python environment, and
record the artifact digest in the deployment system. Do not update a live
checkout in place.

```sh
git clone https://github.com/morluto/jacobian.git
cd jacobian
git checkout <exact-release-or-revision>
uv sync --locked
```

The service command is `uv run jacobian-remote-mcp`. A container or another
immutable packaging system may wrap that command; rollout and rollback remain
the operator's responsibility.

## Configure authentication

Remote serving fails closed unless either `--auth-tokens-file` or the explicit
development-only `--allow-anonymous` option is supplied. Store the token file
outside the artifact and mount it as a secret:

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

The authenticated subject is request context only. It does not select a private
runtime, state directory, artifact store, or database.

## Start Streamable HTTP

```sh
uv run jacobian-remote-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --path /mcp \
  --auth-tokens-file /run/secrets/jacobian-tokens.json \
  --public-base-url https://math-tools.example.org
```

Put a TLS-terminating reverse proxy in front of the bound address. The public
URL must route `/mcp` without stripping the path. For a disposable local
transport test, use `--allow-anonymous`; never expose that mode as an
authenticated service.

## Install the example service files

[`deploy/systemd/jacobian-mcp.service`](../../deploy/systemd/jacobian-mcp.service)
and [`deploy/caddy/Caddyfile`](../../deploy/caddy/Caddyfile) are reviewable
templates. Adapt and install them through the deployment system rather than a
Jacobian-owned installer. Keep authentication secrets out of unit files and
source control.

## Health checks and rollout

Run [`deploy/smoke_remote.py`](../../deploy/smoke_remote.py) against the private
listener and public endpoint before directing traffic to the new artifact.
Where Lean is intentionally installed, also run
[`deploy/smoke_lean.py`](../../deploy/smoke_lean.py). A rollback selects the
previous immutable artifact; there is no application database migration or
state rollback step.

When moving hosts, provision the same pinned artifact, transfer only
operator-owned configuration and secrets, run the probes, and move traffic.
The stateless mathematical server has no tenant state tree to copy or merge.
