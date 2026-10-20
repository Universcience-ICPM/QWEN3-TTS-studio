#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_REPO="$PROJECT_DIR/.stable-audio-3"
ENGINE_DIR="$ENGINE_REPO/optimized/mlx"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Cette installation MLX exige un Mac Apple Silicon." >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Git est requis pour recuperer le moteur Stable Audio officiel." >&2
    exit 1
fi

if [[ -d "$ENGINE_REPO/.git" ]]; then
    echo "Mise a jour du moteur Stable Audio 3..."
    git -C "$ENGINE_REPO" pull --ff-only
elif [[ -e "$ENGINE_REPO" ]]; then
    echo "$ENGINE_REPO existe mais n'est pas un depot Git valide." >&2
    exit 1
else
    echo "Telechargement du moteur Stable Audio 3 officiel..."
    git clone --depth=1 https://github.com/Stability-AI/stable-audio-3.git "$ENGINE_REPO"
fi

echo "Installation du runtime MLX et du modele Small-SFX..."
cd "$ENGINE_DIR"
./install.sh -y --python 3.11 --download sm-sfx

METALLIB="$ENGINE_DIR/.venv/lib/python3.11/site-packages/mlx/lib/mlx.metallib"
if [[ ! -s "$METALLIB" ]] || [[ "$(du -k "$METALLIB" | awk '{print $1}')" -eq 0 ]]; then
    echo "Bibliotheque Metal MLX incomplete : reinstallation de MLX..."
    VIRTUAL_ENV="$ENGINE_DIR/.venv" uv pip install --reinstall "mlx>=0.30"
fi
if [[ ! -s "$METALLIB" ]] || [[ "$(du -k "$METALLIB" | awk '{print $1}')" -eq 0 ]]; then
    echo "MLX reste incomplet. Liberez de l'espace disque puis relancez ce script." >&2
    exit 1
fi

echo
echo "Installation terminee. Lancez : ./start_sfx_macos.sh"
