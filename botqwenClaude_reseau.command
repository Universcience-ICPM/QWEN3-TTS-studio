#!/bin/bash
# Voix (botqwenClaude) — ouverte au réseau local : l'interface et l'API répondent aux autres machines
# du même réseau (Raspberry Pi, autre ordinateur de l'installation…). L'adresse s'affiche au lancement
# et dans l'onglet « À distance ».
# Pour demander un mot de passe (identifiant : voix), enlève le # de la ligne suivante et choisis-le :
# export BOTQWEN_MOT_DE_PASSE="change-moi"
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
export BOTQWEN_RESEAU=1
./env_qwen/bin/python botqwenClaude.py
