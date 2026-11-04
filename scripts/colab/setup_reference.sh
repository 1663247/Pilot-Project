#!/usr/bin/env bash
# Current RAGEN/verl reference stack. Does not support Qwen3.5.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_DIR=${ENV_DIR:-/content/ragen-reference}
UV=${UV:-uv}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Colab Linux x86_64 required'; exit 2; }
[[ -f "$ROOT/verl/verl/__init__.py" ]] || { echo 'Upload initialized verl submodule first'; exit 2; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$UV" venv --python 3.12 --seed "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$UV" pip install --python "$PY" -r "$ROOT/scripts/colab/requirements-reference.txt"
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/verl" -e "$ROOT"
ABI=$("$PY" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
"$UV" pip install --python "$PY" "$WHEEL_URL"
"$PY" -m pip check
"$PY" -m pip freeze > "$ENV_DIR/freeze.txt"
