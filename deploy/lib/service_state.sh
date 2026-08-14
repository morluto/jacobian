#!/usr/bin/env bash

snapshot_systemd_service_state() {
    local systemctl_bin="$1"
    local snapshot_root="$2"
    local unit="$3"
    local active_state
    if "${systemctl_bin}" is-enabled --quiet "${unit}"; then
        : >"${snapshot_root}/${unit}.enabled"
    else
        : >"${snapshot_root}/${unit}.disabled"
    fi
    active_state="$(
        "${systemctl_bin}" show --property=ActiveState --value "${unit}" \
            2>/dev/null || true
    )"
    case "${active_state}" in
        active | activating | reloading)
            : >"${snapshot_root}/${unit}.active"
            ;;
        *)
            : >"${snapshot_root}/${unit}.inactive"
            ;;
    esac
}

restore_systemd_service_state() {
    local systemctl_bin="$1"
    local snapshot_root="$2"
    local unit="$3"
    if [[ -f "${snapshot_root}/${unit}.enabled" ]]; then
        "${systemctl_bin}" enable "${unit}" || return 1
    else
        "${systemctl_bin}" disable "${unit}" >/dev/null 2>&1 || true
        if "${systemctl_bin}" is-enabled --quiet "${unit}"; then
            return 1
        fi
    fi
    if [[ -f "${snapshot_root}/${unit}.active" ]]; then
        "${systemctl_bin}" restart "${unit}" || return 1
    else
        "${systemctl_bin}" stop "${unit}" >/dev/null 2>&1 || true
        if "${systemctl_bin}" is-active --quiet "${unit}"; then
            return 1
        fi
    fi
}
