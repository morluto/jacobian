# Deploy the remote MCP server

[Documentation home](../index.md)

Deploy Jacobian as one immutable Python service artifact. The application owns
mathematical execution and health reporting; the deployment platform owns
provisioning, configuration, rollout, rollback, TLS, process supervision, and
persistence. The checked-in files under `deploy/` are examples for those
platform boundaries.

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

The authenticated subject is attached to the current request as authorization
context.

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

Run `uv run python -m deploy.smoke_remote <url>` against the private listener
and public endpoint before directing traffic to the new artifact. The probe
implementation lives in [`deploy/`](../../deploy/). Roll back by selecting the
previous immutable artifact and rerunning the probe.

When moving hosts, provision the same pinned artifact, transfer operator-owned
configuration and secrets, run the probes, and move traffic.
