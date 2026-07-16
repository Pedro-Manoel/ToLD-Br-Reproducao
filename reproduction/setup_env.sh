#!/usr/bin/env bash
# Cria o venv e instala as dependencias (Linux/macOS). Espelha setup_env.ps1.
# Uso:
#   ./setup_env.sh            # torch com CUDA 12.4 (GPU NVIDIA)
#   ./setup_env.sh --cpu      # torch CPU-only
#   PY=python3.12 ./setup_env.sh
set -euo pipefail
PY="${PY:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.cache/toldbr-repro-venv}"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
if [ "${1:-}" = "--cpu" ]; then TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Criando venv em $VENV_DIR ..."
  "$PY" -m venv "$VENV_DIR"
fi
VPY="$VENV_DIR/bin/python"

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install torch --index-url "$TORCH_INDEX"
"$VPY" -m pip install -r "$(dirname "$0")/requirements.txt"
"$VPY" -m ipykernel install --user --name toldbr-repro

echo "----------------------------------------------------"
echo "VENV pronto em: $VENV_DIR"
echo "Python do venv: $VPY"
echo 'Kernel Jupyter registrado: toldbr-repro'
"$VPY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
