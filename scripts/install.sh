#!/usr/bin/env bash
set -euo pipefail

K2LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$K2LAB_ROOT"

K2LAB_DEV_INSTALL=false
if [[ "${1:-}" == "--dev" ]]; then
    K2LAB_DEV_INSTALL=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dev]" >&2
    exit 2
fi

# A relocated checkout may be launched from a shell that still has its previous
# virtual environment activated. Always target this checkout's own .venv.
unset VIRTUAL_ENV

K2LAB_VENV="$K2LAB_ROOT/.venv"
if [[ -f "$K2LAB_VENV/bin/activate" ]] \
    && ! grep -Fq "VIRTUAL_ENV='$K2LAB_VENV'" "$K2LAB_VENV/bin/activate"; then
    echo "Recreating relocated virtual environment at $K2LAB_VENV"
    rm -rf -- "$K2LAB_VENV"
fi

if command -v uv >/dev/null 2>&1; then
    K2LAB_UV_CACHE="${UV_CACHE_DIR:-$K2LAB_ROOT/.uv-cache}"
    K2LAB_UV_ARGS=(sync --frozen)
    if [[ "$K2LAB_DEV_INSTALL" == true ]]; then
        K2LAB_UV_ARGS+=(--extra dev)
    fi
    UV_CACHE_DIR="$K2LAB_UV_CACHE" uv "${K2LAB_UV_ARGS[@]}"
else
    K2LAB_PYTHON="${K2LAB_PYTHON:-python3.12}"
    if ! command -v "$K2LAB_PYTHON" >/dev/null 2>&1; then
        echo "Python 3.12 is required. Install it or set K2LAB_PYTHON." >&2
        exit 1
    fi
    if [[ ! -x "$K2LAB_VENV/bin/python" ]]; then
        "$K2LAB_PYTHON" -m venv "$K2LAB_VENV"
    fi
    if ! "$K2LAB_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        "$K2LAB_VENV/bin/python" -m ensurepip --upgrade
    fi
    "$K2LAB_VENV/bin/python" -m pip install --upgrade pip
    K2LAB_PIP_TARGET="."
    if [[ "$K2LAB_DEV_INSTALL" == true ]]; then
        K2LAB_PIP_TARGET=".[dev]"
    fi
    "$K2LAB_VENV/bin/python" -m pip install --editable "$K2LAB_PIP_TARGET"
fi

"$K2LAB_VENV/bin/python" -c \
    "import k2core, k2_region_lab; print(f'Installed K2Lab with k2core {k2core.__version__}')"

echo
echo "Installation complete."
echo "Run: $K2LAB_ROOT/.venv/bin/k2lab"
