#!/usr/bin/env bash

snapshot_systemd_service_state() {
    local systemctl_bin="$1"
    local snapshot_root="$2"
    local unit="$3"
    if "${systemctl_bin}" is-enabled --quiet "${unit}"; then
        : >"${snapshot_root}/${unit}.enabled"
    else
        : >"${snapshot_root}/${unit}.disabled"
    fi
    if "${systemctl_bin}" is-active --quiet "${unit}"; then
        : >"${snapshot_root}/${unit}.active"
    else
        : >"${snapshot_root}/${unit}.inactive"
    fi
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
