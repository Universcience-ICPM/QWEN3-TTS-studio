# Prototype Stable Audio 3 Small-SFX

Ce prototype genere des bruitages et des ambiances dans des fichiers WAV
separes des voix produites par Qwen3-TTS.

Il utilise les runtimes optimises officiels de Stable Audio 3 :

- MLX sur Mac Apple Silicon ;
- TFLite/LiteRT sur le processeur sous Windows x64.

Le moteur Stable Audio, ses dependances et ses modeles sont installes dans
`.stable-audio-3/`. Ils ne modifient ni `env_qwen`, ni `.venv`, ni
`.venv-windows`.

## Fonctions du prototype

- pistes de type bruitage ou ambiance ;
- duree de 1 a 120 secondes ;
- une a quatre variantes par lancement ;
- seed aleatoire ou reproductible ;
- controle du nombre d'etapes et du CFG ;
- description negative facultative ;
- ecoute directe et telechargement des WAV ;
- manifeste JSON conserve avec chaque campagne.

Le second onglet, `Scene de roman`, ajoute :

- lecture d'un document `.txt` ou `.docx` ;
- separation uniquement aux titres de chapitre explicites ;
- analyse locale de la narration avec un modele Ollama ;
- exclusion des dialogues, voix, pensees, narration vocale et musique ;
- plan sonore chronologique editable avant generation ;
- rapport comparant le prompt brut du LLM au prompt exact de Stable Audio ;
- generation et assemblage automatiques des sequences ;
- une piste SFX finale distincte pour chaque chapitre ;
- synchronisation facultative sur les pistes voix des chapitres.

Les fichiers sont ecrits dans :

```text
outputs/sfx/<date>_<description>/
├── sfx_01_seed-123.wav
├── sfx_02_seed-124.wav
└── manifest.json
```

Pour une ambiance, les fichiers commencent par `ambience_`.

Les adaptations de romans sont ecrites dans :

```text
outputs/sfx_scenes/<date>_<document>/
├── source.txt
├── manifest.json
├── stable_audio_prompts.md
├── chapter_001_<titre>/
│   ├── clips/
│   └── chapter_001_<titre>.wav
└── chapter_002_<titre>/
    ├── clips/
    └── chapter_002_<titre>.wav
```

## Installation sur Mac Apple Silicon

Depuis le dossier `QWEN` :

```bash
chmod +x install_sfx_macos.sh start_sfx_macos.sh
./install_sfx_macos.sh
```

Le script telecharge le depot officiel dans `.stable-audio-3/`, cree son
environnement MLX independant et telecharge le bundle `sm-sfx`.

Lancer ensuite l'interface :

```bash
./start_sfx_macos.sh
```

## Installation sous Windows

Installer d'abord botQwen avec `install_windows_nvidia.ps1`, puis lancer dans
PowerShell :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_sfx_windows.ps1
.\start_sfx_windows.ps1
```

Le runtime SFX Windows utilise le processeur. Le GPU NVIDIA reste ainsi
disponible pour Qwen3-TTS. Aucun environnement PyTorch Stable Audio n'est
ajoute a `.venv-windows`.

## Ouverture de l'interface

Le Studio SFX s'ouvre sur :

```text
http://127.0.0.1:7861
```

Si ce port est deja utilise par une autre instance, le studio choisit
automatiquement le suivant (`7862`, puis `7863`, etc.). L'adresse exacte est
affichee dans le terminal et ouverte dans le navigateur.

Le port peut etre modifie sous Windows :

```powershell
.\start_sfx_windows.ps1 -Port 7862
```

Sur Mac :

```bash
BOTQWEN_SFX_PORT=7862 ./start_sfx_macos.sh
```

## Ecriture des descriptions

Le modele a ete entraine principalement avec des descriptions anglaises. Une
description anglaise precise donne donc generalement un meilleur resultat.

Dans l'onglet `Scene de roman`, un chapitre peut contenir jusqu'a 120
sequences sonores. La duree indicative d'une sequence est reglable de 2 a 120
secondes. Ces valeurs sont des maximums : le LLM conserve uniquement les
ambiances et evenements audibles qu'il juge utiles au recit.

Sur Mac, l'installateur verifie aussi que la bibliotheque Metal de MLX n'est
pas vide. Si une installation incomplete est detectee, MLX est reinstalle sans
retoucher aux poids Stable Audio deja telecharges.

Exemples de bruitages :

```text
Heavy wooden door slamming in a stone corridor, close perspective
Slow footsteps on wet gravel at night, no background activity
Small ceramic cup placed on a wooden table, intimate room sound
```

Exemples d'ambiances :

```text
Quiet apartment interior during heavy rain, distant city traffic
Large empty museum hall at night, subtle ventilation and room tone
Dense summer forest before a storm, leaves moving in the wind
```

Le prototype ajoute automatiquement une consigne excluant la parole, les voix
et la musique. Le texte saisi et la consigne effective sont tous deux inscrits
dans `manifest.json`.

## Adapter une scene de roman

1. Lancer Ollama et verifier qu'un modele local est installe avec `ollama list`.
2. Ouvrir l'onglet `Scene de roman`.
3. Charger un fichier `.txt` ou `.docx`, ou coller directement une scene.
4. Choisir le nombre maximal de sequences sonores par chapitre.
5. Cliquer sur `Analyser seulement` pour examiner le plan, ou sur
   `Analyser et generer` pour lancer toute la chaine.

Apres l'analyse, la section `Document transmis a Stable Audio` affiche et
permet de telecharger `stable_audio_prompts.md`. Pour chaque sequence, ce
rapport contient :

- l'extrait narratif retenu par Ollama ;
- le prompt anglais brut produit par Ollama ;
- le prompt positif exact apres suppression des contenus vocaux ;
- le negative prompt exact ;
- le type de piste, la duree et la position dans le chapitre.

Une copie identique est conservee dans le dossier final de generation avant le
premier appel a Stable Audio. Le rapport reste donc disponible meme si une
sequence audio echoue.

Les marqueurs `;;` sont consideres comme des changements de plan possibles.
Ils ne creent pas automatiquement un son. Un detail purement visuel ou
psychologique peut donc etre ignore.

Pour un document Word, les styles `Titre` et `Heading` definissent les
chapitres. Dans un fichier texte, les lignes commencant par `Chapitre`,
`Chapter`, `Partie`, `Livre`, `Prologue` ou `Epilogue` sont reconnues. Sans
titre explicite, tout le document est traite comme un seul chapitre.

Les lignes de dialogue commencant par un tiret, ainsi que les repliques entre
guillemets, sont retirees avant l'analyse. Les prompts finaux sont egalement
filtres pour interdire parole, voix, reactions vocales, narration et musique.

## Synchronisation avec les voix

Le champ `Pistes voix par chapitre` accepte plusieurs fichiers WAV ou MP3,
fournis dans le meme ordre que les chapitres du document. Leur nombre doit
correspondre exactement au nombre de chapitres analyses.

Dans ce mode :

- la voix n'est jamais modifiee ni incluse dans la piste SFX ;
- chaque piste SFX a exactement la duree de sa piste voix de reference ;
- chaque son est place d'apres la position de son extrait narratif dans le chapitre ;
- les silences sont preserves entre les evenements ;
- le manifeste enregistre les positions en secondes et le fichier vocal utilise.

Les pistes voix et SFX peuvent ensuite etre superposees directement dans un
logiciel audio. Sans fichier vocal, les sequences sont simplement mises bout a
bout avec le fondu configure dans l'interface.

Ce calage est narratif et proportionnel : si l'extrait n'est pas retrouve, la
position proposee par Ollama sert de repli. Il ne reconnait pas encore les
horodatages exacts des mots dans le fichier vocal. Le plan editable permet de
corriger manuellement la colonne `Position dans le chapitre (%)` avant la
generation.

## Regles de sortie

- format : WAV PCM 16 bits ;
- frequence : 44,1 kHz ;
- canaux : stereo ;
- une generation ne modifie jamais une piste vocale existante ;
- `outputs/` reste exclu de Git.

Une future etape pourra ajouter un alignement force mot a mot et une option de
conversion en 48 kHz, sans remplacer les fichiers sources.

## Licence et acces

Le code d'inference Stable Audio est fourni par Stability AI. Les poids sont
soumis a la Stability AI Community License et aux conditions indiquees sur la
fiche Hugging Face du modele. Il faut les consulter avant un usage commercial
ou organisationnel.

- [depot Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) ;
- [modele Small-SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) ;
- [licence Stability AI](https://stability.ai/license).

## Variables avancees

| Variable | Role |
| --- | --- |
| `BOTQWEN_SFX_BACKEND` | force `mlx` ou `tflite` |
| `BOTQWEN_SFX_ENGINE` | chemin d'un runtime Stable Audio deja installe |
| `BOTQWEN_SFX_PORT` | premier port essaye, `7861` par defaut |
