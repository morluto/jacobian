#!/usr/bin/env bash
#
# Install the committed Jacobian checkout as an immutable systemd release.
#
# This installer deliberately does not curl remote installation scripts. The
# operator installs and reviews uv plus the ingress selected for this host
# (Caddy, and optionally Tailscale) before running it.

set -Eeuo pipefail

# Root-run validation and smoke probes must not mutate an immutable release by
# creating bytecode caches that inherit the operator's (possibly restrictive)
# umask. The long-running service has its own systemd environment.
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/service_state.sh"

MODE="local"
DOMAIN=""
AUTH_TOKENS_FILE=""
TENANT_ID="jacobian-user"
ANONYMOUS_TENANT_ID="jacobian-test"
ALLOW_ANONYMOUS=0
CONFIRM_PUBLIC_ANONYMOUS=0
SKIP_SMOKE=0
DRY_RUN=0
WITH_LEAN=0

BACKEND_PORT=8765
INGRESS_PORT=8766
INSTALL_ROOT="/opt/jacobian"
CONFIG_ROOT="/etc/jacobian-mcp"
CADDY_CONFIG_ROOT="/etc/caddy-jacobian"
SYSTEMD_ROOT="/etc/systemd/system"
DEPLOY_LOCK_PATH="/run/lock/jacobian-mcp-install.lock"
TAILSCALE_STATUS=""
RENDER_ROOT=""
RELEASE_BUILD_DIR=""
RELEASE_WAS_BUILT=0
ROLLBACK_ROOT=""
ROLLBACK_ARMED=0
DEPLOYMENT_ACCEPTED=0
PREVIOUS_RELEASE=""
VALIDATED_INSTALL_ROOT=""

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
    if [[ -n "${ROLLBACK_ROOT}" && -d "${ROLLBACK_ROOT}" ]]; then
        rm -rf -- "${ROLLBACK_ROOT}"
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
  --install-root PATH           Durable root for immutable releases, managed
                                Python, and Lean (default: /opt/jacobian).
  --with-lean                   Build and require the pinned Lean CORE + MATHLIB
                                provider and checker portfolio.
  --skip-smoke                  Do not run the read-only MCP deployment smoke.
  --dry-run                     Validate arguments and print the deployment plan.
  -h, --help                    Show this help.

Examples:
  sudo ./deploy/install.sh
  sudo ./deploy/install.sh --mode domain --domain math.example.org
  sudo ./deploy/install.sh --mode tailscale

The default is token authentication. On the first authenticated install, the
script creates /etc/jacobian-mcp/tokens.json with mode 0600. It prints only that
path; retrieve it explicitly with privileged access or import it into a secret
manager. Subsequent runs reuse it unless --auth-tokens-file is supplied.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    if ((ROLLBACK_ARMED)) && ((!DEPLOYMENT_ACCEPTED)); then
        rollback_deployment 1 || true
    fi
    exit 1
}

log() {
    printf '==> %s\n' "$*"
}

snapshot_file() {
    local name="$1"
    local path="$2"
    if [[ -e "${path}" || -L "${path}" ]]; then
        cp -a -- "${path}" "${ROLLBACK_ROOT}/${name}"
        : >"${ROLLBACK_ROOT}/${name}.present"
    else
        : >"${ROLLBACK_ROOT}/${name}.absent"
    fi
}

restore_file() {
    local name="$1"
    local path="$2"
    if [[ -f "${ROLLBACK_ROOT}/${name}.present" ]]; then
        install -d -m 0755 "$(dirname -- "${path}")"
        rm -rf -- "${path}"
        cp -a -- "${ROLLBACK_ROOT}/${name}" "${path}"
    else
        rm -rf -- "${path}"
    fi
}

rollback_deployment() {
    local original_status="$1"
    local rollback_failed=0
    trap - ERR
    set +e
    printf 'error: deployment failed; restoring the previous activation\n' >&2
    restore_file token "${TOKEN_DESTINATION}" || rollback_failed=1
    restore_file mcp-service "${SYSTEMD_ROOT}/jacobian-mcp.service" \
        || rollback_failed=1
    restore_file anonymous \
        "${SYSTEMD_ROOT}/jacobian-mcp.service.d/anonymous.conf" \
        || rollback_failed=1
    restore_file caddy-config "${CADDY_CONFIG_ROOT}/Caddyfile" \
        || rollback_failed=1
    restore_file caddy-service "${SYSTEMD_ROOT}/jacobian-caddy.service" \
        || rollback_failed=1
    restore_file funnel-service "${SYSTEMD_ROOT}/jacobian-funnel.service" \
        || rollback_failed=1
    if [[ -n "${PREVIOUS_RELEASE}" ]]; then
        ln -sfn "${PREVIOUS_RELEASE}" "${CURRENT_LINK}.rollback" \
            && mv -Tf "${CURRENT_LINK}.rollback" "${CURRENT_LINK}" \
            || rollback_failed=1
    else
        rm -f -- "${CURRENT_LINK}" || rollback_failed=1
    fi
    "${SYSTEMCTL_BIN}" daemon-reload || rollback_failed=1
    restore_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
        jacobian-mcp.service || rollback_failed=1
    restore_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
        jacobian-caddy.service || rollback_failed=1
    restore_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
        jacobian-funnel.service || rollback_failed=1
    ROLLBACK_ARMED=0
    if ((rollback_failed)); then
        printf 'error: rollback encountered additional failures; inspect systemd and the preserved release %s\n' \
            "${RELEASE_DIR}" >&2
    else
        printf 'error: previous deployment activation restored; failed release preserved at %s\n' \
            "${RELEASE_DIR}" >&2
    fi
    set -e
    return "${original_status}"
}

on_error() {
    local status="$?"
    if ((ROLLBACK_ARMED)) && ((!DEPLOYMENT_ACCEPTED)); then
        rollback_deployment "${status}" || true
    fi
    exit "${status}"
}

trap on_error ERR

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

resolve_effective_path() {
    local candidate="$1"
    local unresolved=""
    local resolved

    # `readlink -f` cannot resolve a nonexistent leaf on every supported host.
    # Resolve the nearest existing ancestor, including a symlink ancestor, then
    # append the still-nonexistent suffix without changing its validated shape.
    while [[ ! -e "${candidate}" && ! -L "${candidate}" ]]; do
        [[ "${candidate}" != "/" ]] || break
        unresolved="/${candidate##*/}${unresolved}"
        candidate="${candidate%/*}"
        [[ -n "${candidate}" ]] || candidate="/"
    done
    resolved="$(readlink -f -- "${candidate}" 2>/dev/null || true)"
    [[ -n "${resolved}" ]] || return 1
    printf '%s%s\n' "${resolved%/}" "${unresolved}"
}

validate_install_root() {
    local value="$1"
    local effective
    [[ "${value}" != "/" && "${value}" =~ ^/[A-Za-z0-9._/-]+$ ]] || die \
        "--install-root must be a non-root absolute path without spaces"
    [[ "${value}" != *"//"* && "${value}" != *"/./"* \
        && "${value}" != */. && "${value}" != *"/../"* \
        && "${value}" != */.. ]] || die \
        "--install-root must not contain empty, '.' or '..' path segments"
    case "${value}/" in
        /home/* | /root/* | /tmp/* | /var/tmp/*)
            die "--install-root must remain visible through the systemd sandbox; do not place it below /home, /root, /tmp, or /var/tmp"
            ;;
        /run/* | /dev/shm/*)
            die "--install-root must be durable across reboots; do not place it below /run or /dev/shm"
            ;;
    esac
    effective="$(resolve_effective_path "${value}")" || die \
        "--install-root has an unresolved or broken symlink ancestor"
    [[ "${effective}" != "/" && "${effective}" =~ ^/[A-Za-z0-9._/-]+$ ]] || die \
        "--install-root resolves to a non-root path with unsupported characters"
    case "${effective}/" in
        /home/* | /root/* | /tmp/* | /var/tmp/*)
            die "--install-root resolves below a path hidden by the systemd sandbox: ${effective}"
            ;;
        /run/* | /dev/shm/*)
            die "--install-root resolves below a volatile runtime hierarchy: ${effective}"
            ;;
    esac
    VALIDATED_INSTALL_ROOT="${effective}"
}

validate_release_runtime() {
    local release_dir="$1"
    local entrypoint="${release_dir}/.venv/bin/jacobian-remote-mcp"
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

validate_service_readability() {
    local issue
    local runtime_root

    for runtime_root in "$@"; do
        [[ -d "${runtime_root}" ]] || die \
            "runtime input root is unavailable: ${runtime_root}"
        if ! issue="$(
            "${RUNUSER_BIN}" --user jacobian -- \
                find "${runtime_root}" \
                \( \
                    \( -type d \( ! -readable -o ! -executable \) \) -o \
                    \( -type f ! -readable \) \
                \) \
                -print -quit 2>&1
        )"; then
            die "jacobian service user could not inspect ${runtime_root}: ${issue}"
        fi
        [[ -z "${issue}" ]] || die \
            "runtime input is not readable by the jacobian service user: ${issue}"
    done
}

validate_lean_release_runtime() {
    local release_dir="$1"
    local required
    local required_paths=(
        "lean/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"
        "lean/.lake/build/lib/lean/JacobianLeanRuntime.olean"
        "lean/.lake/build/lib/lean/JacobianLeanProofState.olean"
        "lean/.lake/build/bin/jacobian_lean_proof_state"
    )

    for required in "${required_paths[@]}"; do
        [[ -f "${release_dir}/${required}" ]] || die \
            "pinned Lean release component is unavailable: ${required}"
    done
    (
        cd "${release_dir}"
        "${RUNUSER_BIN}" --user jacobian -- \
            env \
            "ELAN_HOME=${LEAN_ELAN_HOME}" \
            "PATH=${LEAN_SERVICE_PATH}" \
            "${release_dir}/.venv/bin/python" - <<'PY'
from jacobian_checkers import lean4
from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.providers.lean_runtime import lean_provider_runtime

executable, mathlib_runtime = lean4.inspect_runtime(require_mathlib=True)
if not executable.is_file() or mathlib_runtime is None or not mathlib_runtime.is_dir():
    raise SystemExit("pinned Lean provider is unavailable")
runtime = lean_provider_runtime(
    profiles={
        "CORE": {},
        "MATHLIB": {"mathlib_commit": lean4.MATHLIB_COMMIT},
    },
    checker_ids=(),
)
if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
    raise SystemExit(runtime.diagnostic or "pinned Lean provider is unavailable")
PY
    ) || die "pinned Lean provider failed its release readiness probe"
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
        --install-root)
            (($# >= 2)) || die "--install-root requires a path"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --with-lean)
            WITH_LEAN=1
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

INSTALL_ROOT="${INSTALL_ROOT%/}"
validate_install_root "${INSTALL_ROOT}"
INSTALL_ROOT="${VALIDATED_INSTALL_ROOT}"
RELEASE_ROOT="${INSTALL_ROOT}/releases"
CURRENT_LINK="${INSTALL_ROOT}/current"
PYTHON_INSTALL_ROOT="${INSTALL_ROOT}/python"
LEAN_ELAN_HOME="${INSTALL_ROOT}/lean/elan"
LEAN_SERVICE_PATH="${LEAN_ELAN_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOKEN_DESTINATION="${CONFIG_ROOT}/tokens.json"

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
RELEASE_PROFILE="core"
if ((WITH_LEAN)); then
    RELEASE_PROFILE="lean"
fi
RELEASE_SUFFIX=""
if [[ "${RELEASE_PROFILE}" != "core" ]]; then
    RELEASE_SUFFIX="-${RELEASE_PROFILE}"
fi
RELEASE_DIR="${RELEASE_ROOT}/${SHORT_REVISION}${RELEASE_SUFFIX}"

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
  install:     ${INSTALL_ROOT}
  release:     ${RELEASE_DIR}
  python:      ${PYTHON_INSTALL_ROOT}
  mode:        ${MODE}
  connector:   ${CONNECTOR_URL}
  auth:        ${AUTH_DESCRIPTION}
  backend:     jacobian-mcp.service on 127.0.0.1:${BACKEND_PORT}
  caddy:       $([[ "${MODE}" == "local" ]] && printf 'disabled' || printf 'enabled')
  funnel:      $([[ "${MODE}" == "tailscale" ]] && printf 'enabled' || printf 'disabled')
  lean:        $(((WITH_LEAN)) && printf 'pinned CORE + MATHLIB runtime' || printf 'disabled')
  smoke:       $(((SKIP_SMOKE)) && printf 'skipped' || printf 'required')
EOF
    exit 0
fi

((EUID == 0)) || die "run this installer with sudo"
"${GIT[@]}" diff --quiet --ignore-submodules -- \
    || die "tracked working-tree changes exist; commit or stash them before deployment"
"${GIT[@]}" diff --cached --quiet --ignore-submodules -- \
    || die "staged changes exist; commit or stash them before deployment"

INVOKING_HOME=""
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    INVOKING_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6 || true)"
fi

UV_BIN="$(find_executable uv /usr/local/bin/uv /usr/bin/uv || true)"
if [[ -z "${UV_BIN}" && -n "${INVOKING_HOME}" ]]; then
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

ELAN_BIN=""
LEAN_TOOLCHAIN=""
if ((WITH_LEAN)); then
    ELAN_FALLBACKS=(/usr/local/bin/elan /usr/bin/elan)
    if [[ -n "${INVOKING_HOME}" ]]; then
        ELAN_FALLBACKS+=("${INVOKING_HOME}/.elan/bin/elan")
    fi
    ELAN_BIN="$(find_executable elan "${ELAN_FALLBACKS[@]}" || true)"
    [[ -n "${ELAN_BIN}" ]] || die \
        "--with-lean requires an operator-installed elan launcher"
    LEAN_TOOLCHAIN="$(tr -d '\r\n' <"${REPO_ROOT}/lean/lean-toolchain")"
    [[ -n "${LEAN_TOOLCHAIN}" ]] || die "lean/lean-toolchain is empty"
fi

CADDY_BIN=""
if [[ "${MODE}" != "local" ]]; then
    CADDY_BIN="$(find_executable caddy /usr/local/bin/caddy /usr/bin/caddy || true)"
    [[ -n "${CADDY_BIN}" ]] || die \
        "caddy is required for public ingress; install it first"
fi

install -d -m 0755 "$(dirname -- "${DEPLOY_LOCK_PATH}")"
exec 9>"${DEPLOY_LOCK_PATH}"
"${FLOCK_BIN}" --nonblock 9 || die "another Jacobian deployment is in progress"

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
            --managed-python \
            --link-mode copy
    )
    if ((WITH_LEAN)); then
        log "building pinned Lean and Mathlib release runtime"
        install -d -m 0755 "${LEAN_ELAN_HOME}/bin"
        install -m 0755 "$(readlink -f "${ELAN_BIN}")" \
            "${LEAN_ELAN_HOME}/bin/elan"
        INSTALLED_LEAN_TOOLCHAINS="$({
            ELAN_HOME="${LEAN_ELAN_HOME}" \
                "${LEAN_ELAN_HOME}/bin/elan" toolchain list
        })" || die "could not inspect installed Lean toolchains"
        LEAN_TOOLCHAIN_INSTALLED=0
        while IFS= read -r installed_toolchain; do
            if [[ "${installed_toolchain}" == "${LEAN_TOOLCHAIN}" ]]; then
                LEAN_TOOLCHAIN_INSTALLED=1
                break
            fi
        done <<<"${INSTALLED_LEAN_TOOLCHAINS}"
        if ((!LEAN_TOOLCHAIN_INSTALLED)); then
            ELAN_HOME="${LEAN_ELAN_HOME}" \
                "${LEAN_ELAN_HOME}/bin/elan" toolchain install "${LEAN_TOOLCHAIN}"
        fi
        (
            cd "${RELEASE_DIR}/lean"
            ELAN_HOME="${LEAN_ELAN_HOME}" \
                "${LEAN_ELAN_HOME}/bin/elan" run "${LEAN_TOOLCHAIN}" \
                lake exe cache get
            ELAN_HOME="${LEAN_ELAN_HOME}" \
                "${LEAN_ELAN_HOME}/bin/elan" run "${LEAN_TOOLCHAIN}" \
                lake build repl JacobianLeanRuntime jacobian_lean_proof_state
        )
    fi
    chown -R root:root "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"
    # uv and backend caches honor the invoking root umask and may create
    # owner-only directories or executables. The immutable release contains no
    # secrets, so normalize every runtime input before service-user validation.
    chmod -R a+rX "${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}"
    RELEASE_WAS_BUILT=1
elif [[ "$(cat "${RELEASE_DIR}/.git-revision" 2>/dev/null || true)" != "${REVISION}" ]]; then
    die "existing release directory is not bound to revision ${REVISION}"
elif [[ "$(cat "${RELEASE_DIR}/.release-profile" 2>/dev/null || printf 'core')" \
    != "${RELEASE_PROFILE}" ]]; then
    die "existing release directory does not match profile ${RELEASE_PROFILE}"
fi
validate_release_runtime "${RELEASE_DIR}"
if ((WITH_LEAN)); then
    # Elan's shared toolchain is outside the immutable release. Normalize it on
    # every deployment so service readability does not depend on root's umask
    # or on whether this invocation reused an existing release.
    chmod -R a+rX "${LEAN_ELAN_HOME}"
    validate_lean_release_runtime "${RELEASE_DIR}"
fi
if ((RELEASE_WAS_BUILT)); then
    printf '%s\n' "${RELEASE_PROFILE}" >"${RELEASE_DIR}/.release-profile"
    printf '%s\n' "${REVISION}" >"${RELEASE_DIR}/.git-revision"
    chmod 0644 \
        "${RELEASE_DIR}/.release-profile" \
        "${RELEASE_DIR}/.git-revision"
fi

RUNTIME_READ_ROOTS=("${RELEASE_DIR}" "${PYTHON_INSTALL_ROOT}")
if ((WITH_LEAN)); then
    RUNTIME_READ_ROOTS+=("${LEAN_ELAN_HOME}")
fi
validate_service_readability "${RUNTIME_READ_ROOTS[@]}"
if ((RELEASE_WAS_BUILT)); then
    RELEASE_BUILD_DIR=""
fi

if [[ -e "${CURRENT_LINK}" && ! -L "${CURRENT_LINK}" ]]; then
    die "${CURRENT_LINK} exists and is not a symlink"
fi
ROLLBACK_ROOT="$(mktemp -d)"
PREVIOUS_RELEASE="$(readlink "${CURRENT_LINK}" 2>/dev/null || true)"
snapshot_file token "${TOKEN_DESTINATION}"
snapshot_file mcp-service "${SYSTEMD_ROOT}/jacobian-mcp.service"
snapshot_file anonymous "${SYSTEMD_ROOT}/jacobian-mcp.service.d/anonymous.conf"
snapshot_file caddy-config "${CADDY_CONFIG_ROOT}/Caddyfile"
snapshot_file caddy-service "${SYSTEMD_ROOT}/jacobian-caddy.service"
snapshot_file funnel-service "${SYSTEMD_ROOT}/jacobian-funnel.service"
snapshot_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
    jacobian-mcp.service
snapshot_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
    jacobian-caddy.service
snapshot_systemd_service_state "${SYSTEMCTL_BIN}" "${ROLLBACK_ROOT}" \
    jacobian-funnel.service
ROLLBACK_ARMED=1

install -d -m 0755 "$(dirname -- "${CURRENT_LINK}")"
ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"
mv -Tf "${CURRENT_LINK}.new" "${CURRENT_LINK}"

log "installing authentication configuration"
install -d -m 0700 "${CONFIG_ROOT}"
GENERATED_TOKEN_FILE=0
if ((ALLOW_ANONYMOUS)); then
    install -d -m 0755 \
        "${SYSTEMD_ROOT}/jacobian-mcp.service.d"
    sed \
        -e "s|replace-with-unique-test-id|${ANONYMOUS_TENANT_ID}|g" \
        -e "s|/opt/jacobian/current|${CURRENT_LINK}|g" \
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
        "${RELEASE_DIR}/.venv/bin/python" - \
            "${TOKEN_DESTINATION}" "${TENANT_ID}" <<'PY'
import json
import os
import secrets
import sys

destination, tenant_id = sys.argv[1:]
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "tokens": [
                {
                    "tenant_id": tenant_id,
                    "token": secrets.token_urlsafe(48),
                    "scopes": ["jacobian:use"],
                }
            ]
        },
        stream,
        indent=2,
    )
    stream.write("\n")
PY
        GENERATED_TOKEN_FILE=1
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

sed \
    -e "s|https://math-tools.example.org|${PUBLIC_BASE_URL}|g" \
    -e "s|/opt/jacobian/current|${CURRENT_LINK}|g" \
    -e "s|/opt/jacobian/lean/elan|${LEAN_ELAN_HOME}|g" \
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
    SMOKE_TOKEN_FILE=""
    if ((!ALLOW_ANONYMOUS)); then
        SMOKE_TOKEN_FILE="${TOKEN_DESTINATION}"
    fi
    SMOKE_SUCCEEDED=0
    SMOKE_REQUIREMENTS=(
        --require-capability graph.construct.explicit
    )
    if ((WITH_LEAN)); then
        SMOKE_REQUIREMENTS+=(
            --require-capability lean.check
            --require-capability lean.proof_state.apply_tactic
            --require-capability lean.term.apply
            --require-capability lean.retrieve.premises
        )
    fi
    for attempt in {1..12}; do
        if JACOBIAN_MCP_AUTH_TOKENS_FILE="${SMOKE_TOKEN_FILE}" \
            "${RELEASE_DIR}/.venv/bin/python" \
            "${RELEASE_DIR}/deploy/smoke_remote.py" \
            "${CONNECTOR_URL}" \
            --expect-revision "${REVISION}" \
            --expect-policy-profile DEFAULT \
            "${SMOKE_REQUIREMENTS[@]}"; then
            if ((WITH_LEAN)) && ! \
                JACOBIAN_MCP_AUTH_TOKENS_FILE="${SMOKE_TOKEN_FILE}" \
                "${RELEASE_DIR}/.venv/bin/python" \
                "${RELEASE_DIR}/deploy/smoke_lean.py" \
                "${CONNECTOR_URL}"; then
                if ((attempt < 12)); then
                    sleep 5
                    continue
                fi
                break
            fi
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

# Recheck after root-run authentication and smoke probes. With rollback still
# armed, any post-build mutation that makes runtime input private fails closed.
validate_service_readability "${RUNTIME_READ_ROOTS[@]}"

DEPLOYMENT_ACCEPTED=1
ROLLBACK_ARMED=0

cat <<EOF

Jacobian MCP deployment is active.
  revision:  ${REVISION}
  mode:      ${MODE}
  connector: ${CONNECTOR_URL}
  auth file: $([[ "${ALLOW_ANONYMOUS}" == 1 ]] && printf 'anonymous mode' || printf '%s' "${TOKEN_DESTINATION}")
EOF
if ((GENERATED_TOKEN_FILE)); then
    cat <<EOF

A bearer token was generated in ${TOKEN_DESTINATION}.
Retrieve it explicitly with privileged access or import that root-readable file
into your secret manager; the installer never prints credential values.
EOF
fi
