#!/usr/bin/env bash
set -Eeuo pipefail

readonly workspace_root="${K2LAB_WORKSPACE_ROOT:-/workspace/k2lab}"

if [[ -z "${K2LAB_AGENT_SESSION_TOKEN:-}" ]]; then
    printf '%s\n' "K2LAB_AGENT_SESSION_TOKEN is required."
    exit 64
fi
if [[ -z "${K2LAB_WORKSPACE_ID:-}" ]]; then
    printf '%s\n' "K2LAB_WORKSPACE_ID is required."
    exit 64
fi
if [[ -z "${K2LAB_IMAGE_VERSION:-}" ]]; then
    printf '%s\n' "K2LAB_IMAGE_VERSION is required."
    exit 64
fi
if [[ ! -d /workspace ]]; then
    printf '%s\n' "/workspace is not mounted."
    exit 72
fi

mkdir -p "${workspace_root}"
if [[ ! -w "${workspace_root}" ]]; then
    printf '%s\n' "${workspace_root} is not writable."
    exit 73
fi

exec /opt/k2lab-venv/bin/k2lab-agent --host 0.0.0.0 --port 8080
