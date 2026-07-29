#!/usr/bin/env bash
#
# Install the committed Jacobian checkout as an immutable systemd release.
#
# This installer deliberately does not curl remote installation scripts. The
# operator installs and reviews uv plus the ingress selected for this host
# (Caddy, and optionally Tailscale) before running it.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="local"
DOMAIN=""
AUTH_TOKENS_FILE=""
TENANT_ID="jacobian-user"
ANONYMOUS_TENANT_ID="jacobian-test"
ALLOW_ANONYMOUS=0
CONFIRM_PUBLIC_ANONYMOUS=0
SKIP_SMOKE=0
DRY_RUN=0

BACKEND_PORT=8765
INGRESS_PORT=8766
RELEASE_ROOT="/opt/jacobian/releases"
CURRENT_LINK="/opt/jacobian/current"
PYTHON_INSTALL_ROOT="/opt/jacobian/python"
CONFIG_ROOT="/etc/jacobian-mcp"
CADDY_CONFIG_ROOT="/etc/caddy-jacobian"
SYSTEMD_ROOT="/etc/systemd/system"
TOKEN_DESTINATION="${CONFIG_ROOT}/tokens.json"
TAILSCALE_STATUS=""
RENDER_ROOT=""
RELEASE_BUILD_DIR=""
RELEASE_WAS_BUILT=0

cleanup() {
    if [[ -n "${RELEASE_BUILD_DIR}" && -d "${RELEASE_BUILD_DIR}" ]]; then
        rm -rf -- "${RELEASE_BUILD_DIR}"
    fi
    if [[ -n "${RENDER_ROOT}" && -d "${RENDER_ROOT}" ]]; then
        rm -rf "${RENDER_ROOT}"
    fi
    if [[ -n "${TAILSCALE_STATUS}" && -f "${TAILSCALE_STATUS}" ]]; then
        rm -f "${TAILSCALE_STATUS}"
    fi
}

trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage:
  sudo ./deploy/install.sh [options]

Install the committed Jacobian checkout and start its MCP endpoint.

Modes:
  --mode local       Backend on http://127.0.0.1:8765/mcp (default).
  --mode domain      Public HTTPS through Caddy; requires --domain.
  --mode tailscale   Public HTTPS through Tailscale Funnel and local Caddy.

Configuration:
  --domain FQDN                 Public domain for --mode domain.
  --auth-tokens-file PATH       Install an existing static-token JSON secret.
  --tenant-id ID                Tenant for a newly generated token
                                (default: jacobian-user).
  --allow-anonymous             Explicitly disable authentication.
  --anonymous-tenant-id ID      Shared anonymous namespace
                                (default: jacobian-test).
  --confirm-public-anonymous    Also required when anonymous mode is public.
  --skip-smoke                  Do not run the read-only MCP deployment smoke.
  --dry-run                     Validate arguments and print the deployment plan.
  -h, --help                    Show this help.

Examples:
  sudo ./deploy/install.sh
  sudo ./deploy/install.sh --mode domain --domain math.example.org
  sudo ./deploy/install.sh --mode tailscale

The default is token authentication. On the first authenticated install, the
script creates /etc/jacobian-mcp/tokens.json and prints the generated token once.
Subsequent runs reuse that secret unless --auth-tokens-file is supplied.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '==> %s\n' "$*"
}

find_executable() {
    local name="$1"
    shift
    local selected
    selected="$(command -v "${name}" 2>/dev/null || true)"
    if [[ -n "${selected}" && -x "${selected}" ]]; then
        printf '%s\n' "${selected}"
        return
    fi
    for selected in "$@"; do
        if [[ -x "${selected}" ]]; then
            printf '%s\n' "${selected}"
            return
        fi
    done
    return 1
}

validate_tenant_id() {
    local value="$1"
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die \
        "tenant IDs must start with a letter or digit, use only letters, digits, '.', '_', or '-', and be at most 128 characters"
}

validate_domain() {
    local value="$1"
    [[ ${#value} -le 253 ]] || die "domain must be at most 253 characters"
    [[ "${value}" =~ ^([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$ ]] || die \
        "domain must be a fully qualified DNS name such as math.example.org"
}

validate_release_runtime() {
    local release_dir="$1"
    local entrypoint="${release_dir}/.venv/bin/jacobian-mcp"
    local expected_shebang="#!${release_dir}/.venv/bin/python"
    local python_target
    local shebang

    [[ -x "${entrypoint}" ]] || die \
        "release entrypoint is not executable: ${entrypoint}"
    shebang="$(head -n 1 "${entrypoint}")"
    [[ "${shebang}" == "${expected_shebang}" ]] || die \
        "release entrypoint is not bound to its final path: ${shebang}"
    python_target="$(
        readlink -f "${release_dir}/.venv/bin/python" 2>/dev/null || true
    )"
    [[ -n "${python_target}" && -x "${python_target}" ]] || die \
        "release Python is not executable: ${python_target:-unresolved}"
    case "${python_target}" in
        "${PYTHON_INSTALL_ROOT}"/*) ;;
        *) die \
            "release Python must resolve below ${PYTHON_INSTALL_ROOT}, got ${python_target}" ;;
    esac
    "${RUNUSER_BIN}" --user jacobian -- "${entrypoint}" --version >/dev/null \
        || die "release entrypoint is not executable by the jacobian service user"
}

while (($#)); do
    case "$1" in
        --mode)
            (($# >= 2)) || die "--mode requires a value"
            MODE="$2"
            shift 2
            ;;
        --domain)
            (($# >= 2)) || die "--domain requires a value"
            DOMAIN="$2"
            shift 2
            ;;
        --auth-tokens-file)
            (($# >= 2)) || die "--auth-tokens-file requires a path"
            AUTH_TOKENS_FILE="$2"
            shift 2
            ;;
        --tenant-id)
            (($# >= 2)) || die "--tenant-id requires a value"
            TENANT_ID="$2"
            shift 2
            ;;
        --allow-anonymous)
            ALLOW_ANONYMOUS=1
            shift
            ;;
        --anonymous-tenant-id)
            (($# >= 2)) || die "--anonymous-tenant-id requires a value"
            ANONYMOUS_TENANT_ID="$2"
            shift 2
            ;;
        --confirm-public-anonymous)
            CONFIRM_PUBLIC_ANONYMOUS=1
            shift
            ;;
        --skip-smoke)
            SKIP_SMOKE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

case "${MODE}" in
    local | domain | tailscale) ;;
    *) die "--mode must be local, domain, or tailscale" ;;
esac

validate_tenant_id "${TENANT_ID}"
validate_tenant_id "${ANONYMOUS_TENANT_ID}"

if [[ "${MODE}" == "domain" ]]; then
    [[ -n "${DOMAIN}" ]] || die "--mode domain requires --domain"
    validate_domain "${DOMAIN}"
elif [[ -n "${DOMAIN}" ]]; then
    die "--domain is only valid with --mode domain"
fi

if ((ALLOW_ANONYMOUS)) && [[ -n "${AUTH_TOKENS_FILE}" ]]; then
    die "--allow-anonymous and --auth-tokens-file are mutually exclusive"
fi
if ((ALLOW_ANONYMOUS)) && [[ "${MODE}" != "local" ]] \
    && ((!CONFIRM_PUBLIC_ANONYMOUS)); then
    die "public anonymous deployment also requires --confirm-public-anonymous"
fi
if [[ -n "${AUTH_TOKENS_FILE}" && ! -f "${AUTH_TOKENS_FILE}" ]]; then
    die "auth token file does not exist: ${AUTH_TOKENS_FILE}"
fi

GIT=(git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}")
"${GIT[@]}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "deploy/install.sh must be run from a Git clone"
REVISION="$("${GIT[@]}" rev-parse HEAD)"
SHORT_REVISION="$("${GIT[@]}" rev-parse --short=12 HEAD)"
RELEASE_DIR="${RELEASE_ROOT}/${SHORT_REVISION}"

PUBLIC_BASE_URL=""
CONNECTOR_URL=""
TAILSCALE_BIN=""
PYTHON_BIN="$(find_executable python3 /usr/local/bin/python3 /usr/bin/python3 || true)"
[[ -n "${PYTHON_BIN}" ]] || die "python3 is required"

case "${MODE}" in
    local)
        PUBLIC_BASE_URL="http://127.0.0.1:${BACKEND_PORT}"
        CONNECTOR_URL="${PUBLIC_BASE_URL}/mcp"
        ;;
    domain)
        PUBLIC_BASE_URL="https://${DOMAIN}"
        CONNECTOR_URL="${PUBLIC_BASE_URL}/mcp"
        ;;
    tailscale)
        TAILSCALE_BIN="$(find_executable tailscale /usr/bin/tailscale || true)"
        [[ -n "${TAILSCALE_BIN}" ]] || die \
            "tailscale is required for --mode tailscale"
        TAILSCALE_STATUS="$(mktemp)"
        "${TAILSCALE_BIN}" status --json >"${TAILSCALE_STATUS}" \
            || die "tailscale is not connected"
        DOMAIN="$("${PYTHON_BIN}" - "${TAILSCALE_STATUS}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
name = payload.get("Self", {}).get("DNSName", "").rstrip(".")
if not name:
    raise SystemExit("Tailscale did not report Self.DNSName")
print(name)
PY
)"
        validate_domain "${DOMAIN}"
        PUBLIC_BASE_URL="https://${DOMAIN}"
        CONNECTOR_URL="${PUBLIC_BASE_URL}/mcp"
        ;;
esac

AUTH_DESCRIPTION="generated or existing static bearer token"
if ((ALLOW_ANONYMOUS)); then
    AUTH_DESCRIPTION="anonymous shared tenant ${ANONYMOUS_TENANT_ID}"
elif [[ -n "${AUTH_TOKENS_FILE}" ]]; then
    AUTH_DESCRIPTION="static bearer tokens from ${AUTH_TOKENS_FILE}"
fi

if ((DRY_RUN)); then
    cat <<EOF
Jacobian deployment plan
  revision:    ${REVISION}
  release:     ${RELEASE_DIR}
  python:      ${PYTHON_INSTALL_ROOT}
  mode:        ${MODE}
  connector:   ${CONNECTOR_URL}
  auth:        ${AUTH_DESCRIPTION}
  backend:     jacobian-mcp.service on 127.0.0.1:${BACKEND_PORT}
  caddy:       $([[ "${MODE}" == "local" ]] && printf 'disabled' || printf 'enabled')
  funnel:      $([[ "${MODE}" == "tailscale" ]] && printf 'enabled' || printf 'disabled')
  smoke:       $(((SKIP_SMOKE)) && printf 'skipped' || printf 'required')
EOF
    exit 0
fi

((EUID == 0)) || die "run this installer with sudo"
"${GIT[@]}" diff --quiet --ignore-submodules -- \
    || die "tracked working-tree changes exist; commit or stash them before deployment"
"${GIT[@]}" diff --cached --quiet --ignore-submodules -- \
    || die "staged changes exist; commit or stash them before deployment"

UV_BIN="$(find_executable uv /usr/local/bin/uv /usr/bin/uv || true)"
if [[ -z "${UV_BIN}" && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    INVOKING_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    UV_BIN="$(find_executable uv "${INVOKING_HOME}/.local/bin/uv" || true)"
fi
[[ -n "${UV_BIN}" ]] || die \
    "uv is required; install it first or make it available on root's PATH"

SYSTEMCTL_BIN="$(find_executable systemctl /usr/bin/systemctl || true)"
SYSTEMD_ANALYZE_BIN="$(find_executable systemd-analyze /usr/bin/systemd-analyze || true)"
RUNUSER_BIN="$(find_executable runuser /usr/sbin/runuser /usr/bin/runuser || true)"
FLOCK_BIN="$(find_executable flock /usr/bin/flock || true)"
[[ -n "${SYSTEMCTL_BIN}" && -n "${SYSTEMD_ANALYZE_BIN}" \
    && -n "${RUNUSER_BIN}" && -n "${FLOCK_BIN}" ]] || die \
    "this installer requires a systemd host"

CADDY_BIN=""
if [[ "${MODE}" != "local" ]]; then
    CADDY_BIN="$(find_executable caddy /usr/local/bin/caddy /usr/bin/caddy || true)"
    [[ -n "${CADDY_BIN}" ]] || die \
        "caddy is required for public ingress; install it first"
fi

if ! getent group jacobian >/dev/null; then
    groupadd --system jacobian
fi
if ! id -u jacobian >/dev/null 2>&1; then
    useradd --system --gid jacobian --home-dir /nonexistent \
        --shell /usr/sbin/nologin jacobian
fi
if [[ "${MODE}" != "local" ]]; then
    if ! getent group jacobian-caddy >/dev/null; then
        groupadd --system jacobian-caddy
    fi
    if ! id -u jacobian-caddy >/dev/null 2>&1; then
        useradd --system --gid jacobian-caddy --home-dir /nonexistent \
            --shell /usr/sbin/nologin jacobian-caddy
    fi
fi

log "installing immutable release ${SHORT_REVISION}"
install -d -m 0755 "${RELEASE_ROOT}" "${PYTHON_INSTALL_ROOT}"
exec 9>"$(dirname -- "${RELEASE_ROOT}")/.install.lock"
"${FLOCK_BIN}" --nonblock 9 || die "another Jacobian deployment is in progress"
if [[ -e "${RELEASE_DIR}" && ! -d "${RELEASE_DIR}" ]]; then
    die "release path exists and is not a directory: ${RELEASE_DIR}"
fi
if [[ -d "${RELEASE_DIR}" && ! -f "${RELEASE_DIR}/.git-revision" ]]; then
    ACTIVE_RELEASE="$(readlink -f "${CURRENT_LINK}" 2>/dev/null || true)"
    [[ "${ACTIVE_RELEASE}" != "${RELEASE_DIR}" ]] || die \
        "active release is incomplete and will not be removed: ${RELEASE_DIR}"
    log "removing incomplete inactive release ${SHORT_REVISION}"
    rm -rf -- "${RELEASE_DIR}"
fi
if [[ ! -d "${RELEASE_DIR}" ]]; then
    RELEASE_BUILD_DIR="${RELEASE_DIR}"
    install -d -m 0755 "${RELEASE_BUILD_DIR}"
    "${GIT[@]}" archive --format=tar HEAD | tar -xf - -C "${RELEASE_BUILD_DIR}"
    (
        cd "${RELEASE_DIR}"
        UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_ROOT}" \
            "${UV_BIN}" sync \
            --locked \
            --no-dev \
            --all-extras \
            --managed-python \
            --link-mode copy
    )
    chown -R root:root "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"
    RELEASE_WAS_BUILT=1
elif [[ "$(cat "${RELEASE_DIR}/.git-revision" 2>/dev/null || true)" != "${REVISION}" ]]; then
    die "existing release directory is not bound to revision ${REVISION}"
fi
validate_release_runtime "${RELEASE_DIR}"
if ((RELEASE_WAS_BUILT)); then
    printf '%s\n' "${REVISION}" >"${RELEASE_DIR}/.git-revision"
    RELEASE_BUILD_DIR=""
fi

if [[ -e "${CURRENT_LINK}" && ! -L "${CURRENT_LINK}" ]]; then
    die "${CURRENT_LINK} exists and is not a symlink"
fi
install -d -m 0755 "$(dirname -- "${CURRENT_LINK}")"
ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"
mv -Tf "${CURRENT_LINK}.new" "${CURRENT_LINK}"

log "installing authentication configuration"
install -d -m 0700 "${CONFIG_ROOT}"
GENERATED_TOKEN=""
if ((ALLOW_ANONYMOUS)); then
    install -d -m 0755 \
        "${SYSTEMD_ROOT}/jacobian-mcp.service.d"
    sed \
        -e "s|replace-with-unique-test-id|${ANONYMOUS_TENANT_ID}|g" \
        "${REPO_ROOT}/deploy/systemd/jacobian-mcp-anonymous.conf" \
        >"${SYSTEMD_ROOT}/jacobian-mcp.service.d/anonymous.conf"
    chmod 0644 "${SYSTEMD_ROOT}/jacobian-mcp.service.d/anonymous.conf"
else
    rm -f "${SYSTEMD_ROOT}/jacobian-mcp.service.d/anonymous.conf"
    if [[ -n "${AUTH_TOKENS_FILE}" ]]; then
        "${RELEASE_DIR}/.venv/bin/python" - "${AUTH_TOKENS_FILE}" <<'PY'
import sys

from jacobian.adapters.mcp.remote import load_static_token_file

grants = load_static_token_file(sys.argv[1])
if not any("jacobian:use" in grant.scopes for grant in grants):
    raise SystemExit("token file has no grant with the required jacobian:use scope")
PY
        install -m 0600 "${AUTH_TOKENS_FILE}" "${TOKEN_DESTINATION}"
    elif [[ ! -f "${TOKEN_DESTINATION}" ]]; then
        GENERATED_TOKEN="$("${RELEASE_DIR}/.venv/bin/python" - <<'PY'
import secrets

print(secrets.token_urlsafe(48))
PY
)"
        umask 077
        printf '{\n  "tokens": [\n    {\n      "tenant_id": "%s",\n      "token": "%s",\n      "scopes": ["jacobian:use"]\n    }\n  ]\n}\n' \
            "${TENANT_ID}" "${GENERATED_TOKEN}" >"${TOKEN_DESTINATION}"
    fi
    "${RELEASE_DIR}/.venv/bin/python" - "${TOKEN_DESTINATION}" <<'PY'
import sys

from jacobian.adapters.mcp.remote import load_static_token_file

grants = load_static_token_file(sys.argv[1])
if not any("jacobian:use" in grant.scopes for grant in grants):
    raise SystemExit("token file has no grant with the required jacobian:use scope")
PY
fi

RENDER_ROOT="$(mktemp -d)"

sed "s|https://math-tools.example.org|${PUBLIC_BASE_URL}|g" \
    "${REPO_ROOT}/deploy/systemd/jacobian-mcp.service" \
    >"${RENDER_ROOT}/jacobian-mcp.service"
install -m 0644 "${RENDER_ROOT}/jacobian-mcp.service" \
    "${SYSTEMD_ROOT}/jacobian-mcp.service"

if [[ "${MODE}" != "local" ]]; then
    install -d -m 0755 "${CADDY_CONFIG_ROOT}"
    if [[ "${MODE}" == "domain" ]]; then
        sed \
            -e "s|http://:8766 {|${DOMAIN} {|g" \
            -e '/^[[:space:]]*bind 127\\.0\\.0\\.1[[:space:]]*$/d' \
            "${REPO_ROOT}/deploy/caddy/Caddyfile" \
            >"${RENDER_ROOT}/Caddyfile"
        "${CADDY_BIN}" fmt --overwrite "${RENDER_ROOT}/Caddyfile"
    else
        cp "${REPO_ROOT}/deploy/caddy/Caddyfile" "${RENDER_ROOT}/Caddyfile"
    fi
    install -m 0644 "${RENDER_ROOT}/Caddyfile" \
        "${CADDY_CONFIG_ROOT}/Caddyfile"
    sed "s|/usr/local/bin/caddy|${CADDY_BIN}|g" \
        "${REPO_ROOT}/deploy/systemd/jacobian-caddy.service" \
        >"${RENDER_ROOT}/jacobian-caddy.service"
    install -m 0644 "${RENDER_ROOT}/jacobian-caddy.service" \
        "${SYSTEMD_ROOT}/jacobian-caddy.service"
    "${CADDY_BIN}" validate --config "${CADDY_CONFIG_ROOT}/Caddyfile" \
        --adapter caddyfile
fi

if [[ "${MODE}" == "tailscale" ]]; then
    sed "s|/usr/bin/tailscale|${TAILSCALE_BIN}|g" \
        "${REPO_ROOT}/deploy/systemd/jacobian-funnel.service" \
        >"${RENDER_ROOT}/jacobian-funnel.service"
    install -m 0644 "${RENDER_ROOT}/jacobian-funnel.service" \
        "${SYSTEMD_ROOT}/jacobian-funnel.service"
fi

log "validating and starting systemd services"
"${SYSTEMD_ANALYZE_BIN}" verify "${SYSTEMD_ROOT}/jacobian-mcp.service"
if [[ "${MODE}" != "local" ]]; then
    "${SYSTEMD_ANALYZE_BIN}" verify "${SYSTEMD_ROOT}/jacobian-caddy.service"
fi
if [[ "${MODE}" == "tailscale" ]]; then
    "${SYSTEMD_ANALYZE_BIN}" verify "${SYSTEMD_ROOT}/jacobian-funnel.service"
fi
"${SYSTEMCTL_BIN}" daemon-reload
"${SYSTEMCTL_BIN}" enable jacobian-mcp.service
"${SYSTEMCTL_BIN}" restart jacobian-mcp.service

case "${MODE}" in
    local)
        "${SYSTEMCTL_BIN}" disable --now jacobian-funnel.service \
            >/dev/null 2>&1 || true
        "${SYSTEMCTL_BIN}" disable --now jacobian-caddy.service \
            >/dev/null 2>&1 || true
        ;;
    domain)
        "${SYSTEMCTL_BIN}" disable --now jacobian-funnel.service \
            >/dev/null 2>&1 || true
        "${SYSTEMCTL_BIN}" enable jacobian-caddy.service
        "${SYSTEMCTL_BIN}" restart jacobian-caddy.service
        ;;
    tailscale)
        "${SYSTEMCTL_BIN}" enable jacobian-caddy.service
        "${SYSTEMCTL_BIN}" restart jacobian-caddy.service
        "${SYSTEMCTL_BIN}" enable jacobian-funnel.service
        "${SYSTEMCTL_BIN}" restart jacobian-funnel.service
        ;;
esac

"${SYSTEMCTL_BIN}" is-active --quiet jacobian-mcp.service \
    || die "jacobian-mcp.service did not become active"
if [[ "${MODE}" != "local" ]]; then
    "${SYSTEMCTL_BIN}" is-active --quiet jacobian-caddy.service \
        || die "jacobian-caddy.service did not become active"
fi
if [[ "${MODE}" == "tailscale" ]]; then
    "${SYSTEMCTL_BIN}" is-active --quiet jacobian-funnel.service \
        || die "jacobian-funnel.service did not become active"
fi

if ((!SKIP_SMOKE)); then
    log "running the read-only deployment smoke"
    SMOKE_TOKEN=""
    if ((!ALLOW_ANONYMOUS)); then
        SMOKE_TOKEN="$("${RELEASE_DIR}/.venv/bin/python" - \
            "${TOKEN_DESTINATION}" <<'PY'
import sys

from jacobian.adapters.mcp.remote import load_static_token_file

grants = load_static_token_file(sys.argv[1])
print(next(grant.token for grant in grants if "jacobian:use" in grant.scopes))
PY
)"
    fi
    SMOKE_SUCCEEDED=0
    for attempt in {1..12}; do
        if JACOBIAN_MCP_BEARER_TOKEN="${SMOKE_TOKEN}" \
            "${RELEASE_DIR}/.venv/bin/python" \
            "${RELEASE_DIR}/deploy/smoke_remote.py" \
            "${CONNECTOR_URL}" \
            --expect-policy-profile DEFAULT \
            --require-capability graph.construct.explicit; then
            SMOKE_SUCCEEDED=1
            break
        fi
        if ((attempt < 12)); then
            sleep 5
        fi
    done
    ((SMOKE_SUCCEEDED)) || die \
        "deployment smoke failed; inspect jacobian-mcp and ingress journals"
fi

cat <<EOF

Jacobian MCP deployment is active.
  revision:  ${REVISION}
  mode:      ${MODE}
  connector: ${CONNECTOR_URL}
  auth file: $([[ "${ALLOW_ANONYMOUS}" == 1 ]] && printf 'anonymous mode' || printf '%s' "${TOKEN_DESTINATION}")
EOF
if [[ -n "${GENERATED_TOKEN}" ]]; then
    cat <<EOF

Generated bearer token (shown once; store it in your client secret manager):
${GENERATED_TOKEN}
EOF
fi
