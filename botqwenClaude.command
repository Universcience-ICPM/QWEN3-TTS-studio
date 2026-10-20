#!/bin/bash
# Voix (botqwenClaude) — sur ce Mac seulement.
# Rien n'est téléchargé : Qwen et la reconnaissance vocale sont déjà sur le Mac.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
./env_qwen/bin/python botqwenClaude.py
