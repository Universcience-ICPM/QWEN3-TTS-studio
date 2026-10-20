#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$PROJECT_DIR/.stable-audio-3/optimized/mlx/sa3" ]]; then
    echo "Moteur SFX absent. Lancez d'abord ./install_sfx_macos.sh" >&2
    exit 1
fi

for candidate in "$PROJECT_DIR/env_qwen/bin/python" "$PROJECT_DIR/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
        exec "$candidate" "$PROJECT_DIR/sfx_app.py"
    fi
done

echo "Environnement botQwen absent. Installez d'abord requirements.txt." >&2
exit 1
