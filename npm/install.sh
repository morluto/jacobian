#!/bin/sh

set -eu

package_name="jacobian"
release_selector="${JACOBIAN_RELEASE:-latest}"
uv_version="0.11.28"
uv_installer_sha256="b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b"

plain=0
assume_yes=0
dry_run=0
defer_runtime=0
install_uv=-1
select_all=0
select_claude=0
select_codex=0
select_cursor=0
select_gemini=0
select_opencode=0
selected_clients=0

candidate_dir=""
temporary_dir=""
lock_dir=""
lock_held=0
activation_changed=0
binding_complete=0
previous_link=""
activated_target=""

usage() {
    cat <<'EOF'
Install Jacobian for one or more coding agents.

Usage:
  curl -fsSL https://raw.githubusercontent.com/morluto/jacobian/main/npm/install.sh | sh
  curl -fsSL https://raw.githubusercontent.com/morluto/jacobian/main/npm/install.sh | \
    sh -s -- --client codex --yes

Options:
  --client ID          Configure claude, codex, cursor, gemini, or opencode.
                       Repeat the option to configure multiple clients.
  --all                Configure every supported client.
  --yes, -y            Skip confirmations; requires --client or --all.
  --release VERSION    Install an exact version, latest, or alpha.
  --defer-runtime      Do not install and verify the local math runtime now.
  --install-uv         Install the pinned uv release if uv is not on PATH.
  --no-install-uv      Fail instead of offering to install uv.
  --dry-run            Print the plan without downloading or changing files.
  --plain              Disable terminal colors.
  --help, -h           Show this help.

Environment:
  JACOBIAN_DATA_DIR    Guided installer release data root.
  JACOBIAN_BIN_DIR     Directory for the stable jacobian command.
  JACOBIAN_RELEASE     Default release selector.
  NO_COLOR             Disable terminal colors.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_tty() {
    [ -t 1 ] && [ -r /dev/tty ] && [ -w /dev/tty ]
}

init_style() {
    if [ "$plain" -eq 1 ] || [ "${NO_COLOR+x}" = x ] || ! is_tty; then
        bold=""
        cyan=""
        amber=""
        green=""
        reset=""
        return
    fi
    esc=$(printf '\033')
    bold="${esc}[1m"
    cyan="${esc}[38;2;25;145;170m"
    amber="${esc}[38;2;211;132;43m"
    green="${esc}[38;2;67;160;71m"
    reset="${esc}[0m"
}

banner() {
    printf '\n%s%s  J A C O B I A N%s\n' "$bold" "$amber" "$reset"
    printf '%s  exact math  ->  inspectable evidence  ->  your agent%s\n\n' \
        "$cyan" "$reset"
}

step() {
    printf '%s[%s/%s]%s %s\n' "$cyan" "$1" "$2" "$reset" "$3"
}

success() {
    printf '%s%s%s\n' "$green" "$1" "$reset"
}

add_client() {
    case "$1" in
        claude)
            [ "$select_claude" -eq 1 ] || selected_clients=$((selected_clients + 1))
            select_claude=1
            ;;
        codex)
            [ "$select_codex" -eq 1 ] || selected_clients=$((selected_clients + 1))
            select_codex=1
            ;;
        cursor)
            [ "$select_cursor" -eq 1 ] || selected_clients=$((selected_clients + 1))
            select_cursor=1
            ;;
        gemini)
            [ "$select_gemini" -eq 1 ] || selected_clients=$((selected_clients + 1))
            select_gemini=1
            ;;
        opencode)
            [ "$select_opencode" -eq 1 ] || selected_clients=$((selected_clients + 1))
            select_opencode=1
            ;;
        *) die "unknown client '$1'; expected claude, codex, cursor, gemini, or opencode" ;;
    esac
}

main() {
while [ "$#" -gt 0 ]; do
    case "$1" in
        --client)
            [ "$#" -ge 2 ] || die "--client requires a value"
            add_client "$2"
            shift 2
            ;;
        --client=*)
            add_client "${1#--client=}"
            shift
            ;;
        --all)
            select_all=1
            shift
            ;;
        --yes|-y)
            assume_yes=1
            shift
            ;;
        --release)
            [ "$#" -ge 2 ] || die "--release requires a value"
            release_selector="$2"
            shift 2
            ;;
        --release=*)
            release_selector="${1#--release=}"
            shift
            ;;
        --defer-runtime)
            defer_runtime=1
            shift
            ;;
        --install-uv)
            install_uv=1
            shift
            ;;
        --no-install-uv)
            install_uv=0
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --plain)
            plain=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *) die "unknown argument '$1'" ;;
    esac
done

case "$release_selector" in
    latest|alpha) ;;
    *[!0-9A-Za-z.-]*|.*|*..*|*-|"")
        die "unsafe release selector '$release_selector'"
        ;;
esac

if [ "$assume_yes" -eq 1 ] && [ "$select_all" -eq 0 ] && [ "$selected_clients" -eq 0 ]; then
    die "--yes requires --client or --all; client detection is not consent"
fi
if [ "$select_all" -eq 1 ] && [ "$selected_clients" -gt 0 ]; then
    die "--all cannot be combined with --client"
fi

init_style
banner

data_dir="${JACOBIAN_DATA_DIR:-${XDG_DATA_HOME:-${HOME:-}/.local/share}/jacobian}"
bin_dir="${JACOBIAN_BIN_DIR:-${HOME:-}/.local/bin}"
release_root="$data_dir/npm-releases"
command_path="$bin_dir/jacobian"

if [ "$dry_run" -eq 1 ]; then
    cat <<EOF
Install plan
  release:       $release_selector (resolved and pinned through npm)
  launcher:      $release_root/<version>
  command:       $command_path
  clients:       $([ "$select_all" -eq 1 ] && printf all || { [ "$selected_clients" -gt 0 ] && printf '%s selected' "$selected_clients" || printf interactive; })
  math runtime:  $([ "$defer_runtime" -eq 1 ] && printf deferred || printf 'installed and verified after setup (~160 MB; +~110 MB if Python 3.12 is needed)')
  changes:       none (dry-run)
EOF
    exit 0
fi

[ -n "${HOME:-}" ] || die "HOME must be set"
[ "$(id -u)" -ne 0 ] || die "run the local installer as your normal user, not root"
if [ "$selected_clients" -eq 0 ] && [ "$select_all" -eq 0 ] && ! is_tty; then
    die "non-interactive setup requires --client or --all"
fi

node_bin="${JACOBIAN_NODE_BIN:-$(command -v node || true)}"
npm_bin="${JACOBIAN_NPM_BIN:-$(command -v npm || true)}"
[ -n "$node_bin" ] || die "Node.js 18 or newer is required"
[ -n "$npm_bin" ] || die "npm is required"

node_major=$("$node_bin" -p 'Number(process.versions.node.split(".")[0])') \
    || die "could not inspect Node.js"
case "$node_major" in *[!0-9]*|"") die "could not inspect Node.js" ;; esac
[ "$node_major" -ge 18 ] || die "Node.js 18 or newer is required; found major version $node_major"
"$npm_bin" --version >/dev/null 2>&1 || die "npm is not usable"

step 1 4 "Resolve an immutable launcher release"
release_json=$("$npm_bin" view "$package_name@$release_selector" version --json) \
    || die "npm could not resolve jacobian@$release_selector"
resolved_version=$(printf '%s' "$release_json" | "$node_bin" -e '
let value;
try { value = JSON.parse(require("node:fs").readFileSync(0, "utf8")); }
catch { process.exit(1); }
if (typeof value !== "string" ||
    !/^\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$/.test(value)) process.exit(1);
process.stdout.write(value);
') || die "npm returned an unsupported Jacobian version"
printf '      jacobian@%s\n' "$resolved_version"

cleanup() {
    status=$?
    trap - 0
    if [ "$status" -ne 0 ] && [ "$activation_changed" -eq 1 ] && \
        [ "$binding_complete" -eq 0 ] && \
        [ -L "$command_path" ] && [ "$(readlink "$command_path")" = "$activated_target" ]; then
        if [ -n "$previous_link" ]; then
            rollback_link="$bin_dir/.jacobian.rollback.$$"
            ln -s "$previous_link" "$rollback_link" && mv -f "$rollback_link" "$command_path"
        else
            rm -f "$command_path"
        fi
    fi
    [ -z "$candidate_dir" ] || rm -rf "$candidate_dir"
    [ -z "$temporary_dir" ] || rm -rf "$temporary_dir"
    if [ "$lock_held" -eq 1 ]; then
        rm -rf "$lock_dir"
    fi
    if [ "$status" -ne 0 ] && [ "$activation_changed" -eq 1 ] && \
        [ "$binding_complete" -eq 0 ]; then
        printf 'Jacobian launcher activation was rolled back.\n' >&2
    fi
    exit "$status"
}
trap cleanup 0
trap 'exit 130' INT HUP TERM

mkdir -p "$data_dir" "$release_root" "$bin_dir"
lock_dir="$data_dir/install.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    die "another Jacobian install is active, or $lock_dir is stale"
fi
lock_held=1

if [ -e "$command_path" ] && [ ! -L "$command_path" ]; then
    die "$command_path exists and is not a managed symlink"
fi
if [ -L "$command_path" ]; then
    previous_link=$(readlink "$command_path")
    case "$previous_link" in
        "$release_root"/*/bin/jacobian) ;;
        *) die "$command_path is not managed by this installer" ;;
    esac
fi

if [ "$defer_runtime" -eq 0 ]; then
    uv_bin="${JACOBIAN_UV_BIN:-$(command -v uv || true)}"
    if [ -z "$uv_bin" ]; then
        if [ "$install_uv" -eq -1 ]; then
            if [ "$assume_yes" -eq 1 ]; then
                install_uv=1
            elif is_tty; then
                printf 'uv is required for the local math runtime. Install uv %s? [Y/n] ' \
                    "$uv_version" >/dev/tty
                IFS= read -r answer </dev/tty || answer=""
                case "$answer" in n|N|no|NO|No) install_uv=0 ;; *) install_uv=1 ;; esac
            else
                install_uv=0
            fi
        fi
        [ "$install_uv" -eq 1 ] || die "uv is required; install it or pass --install-uv"
        command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
        temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/jacobian-install.XXXXXX") \
            || die "could not create a temporary directory"
        uv_script="$temporary_dir/uv-install.sh"
        uv_url="https://astral.sh/uv/$uv_version/install.sh"
        printf '      downloading pinned uv installer %s\n' "$uv_version"
        curl -fsSL "$uv_url" -o "$uv_script"
        if command -v sha256sum >/dev/null 2>&1; then
            printf '%s  %s\n' "$uv_installer_sha256" "$uv_script" | sha256sum -c - >/dev/null
        elif command -v shasum >/dev/null 2>&1; then
            actual=$(shasum -a 256 "$uv_script" | awk '{print $1}')
            [ "$actual" = "$uv_installer_sha256" ] || die "uv installer checksum mismatch"
        else
            die "sha256sum or shasum is required to verify the uv installer"
        fi
        UV_NO_MODIFY_PATH=1 sh "$uv_script"
        PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        export PATH
        uv_bin=$(command -v uv || true)
        [ -n "$uv_bin" ] || die "uv installed but was not found in the expected user directories"
    fi
    "$uv_bin" --version >/dev/null 2>&1 || die "uv is not usable"
fi

step 2 4 "Install the small launcher without lifecycle scripts"
release_dir="$release_root/$resolved_version"
if [ -d "$release_dir" ]; then
    installed_version=$("$release_dir/bin/jacobian" --version 2>/dev/null || true)
    [ "$installed_version" = "jacobian $resolved_version" ] \
        || die "existing release $release_dir failed validation"
    printf '      reusing %s\n' "$release_dir"
else
    if [ -e "$release_dir" ] || [ -L "$release_dir" ]; then
        die "$release_dir exists and is not a valid release directory"
    fi
    candidate_dir=$(mktemp -d "$release_root/.${resolved_version}.XXXXXX") \
        || die "could not create a release candidate"
    "$npm_bin" install --global --prefix "$candidate_dir" --ignore-scripts \
        --omit=dev --no-audit --no-fund "$package_name@$resolved_version"
    installed_version=$("$candidate_dir/bin/jacobian" --version 2>/dev/null || true)
    [ "$installed_version" = "jacobian $resolved_version" ] \
        || die "installed launcher failed its version check"
    mv "$candidate_dir" "$release_dir"
    candidate_dir=""
fi

activated_target="$release_dir/bin/jacobian"
new_link="$bin_dir/.jacobian.new.$$"
ln -s "$activated_target" "$new_link"
mv -f "$new_link" "$command_path"
activation_changed=1

step 3 4 "Bind Jacobian to coding agents"
set -- setup
[ "$select_all" -eq 0 ] || set -- "$@" --all
[ "$select_claude" -eq 0 ] || set -- "$@" --client claude
[ "$select_codex" -eq 0 ] || set -- "$@" --client codex
[ "$select_cursor" -eq 0 ] || set -- "$@" --client cursor
[ "$select_gemini" -eq 0 ] || set -- "$@" --client gemini
[ "$select_opencode" -eq 0 ] || set -- "$@" --client opencode
[ "$assume_yes" -eq 0 ] || set -- "$@" --yes
if is_tty && [ "$assume_yes" -eq 0 ]; then
    "$command_path" "$@" </dev/tty
else
    "$command_path" "$@"
fi
binding_complete=1

if [ "$defer_runtime" -eq 1 ]; then
    step 4 4 "Defer the local math runtime"
    printf '      first use installs ~160 MB; add ~110 MB if Python 3.12 is needed\n'
else
    step 4 4 "Install the math runtime and verify the MCP handshake"
    if [ -z "$temporary_dir" ]; then
        temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/jacobian-install.XXXXXX") \
            || die "could not create a temporary directory"
    fi
    mkdir -p "$temporary_dir/doctor-state"
    JACOBIAN_STATE_DIR="$temporary_dir/doctor-state" "$command_path" doctor
fi

printf '\n'
success "Jacobian $resolved_version is ready."
if ! command -v jacobian >/dev/null 2>&1; then
    printf 'Add the command to this shell with:\n\n  export PATH="%s:\$PATH"\n' "$bin_dir"
fi
if [ "$defer_runtime" -eq 1 ]; then
    printf 'Run %s doctor when you are ready to install and verify the math runtime.\n' \
        "$command_path"
fi
}

main "$@"
