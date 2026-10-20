# Voix - atelier vocal local avec Qwen3-TTS

**Voix** est le nom de l'application. `botQwen2.py` reste son point d'entrée
technique afin de préserver les installations existantes. Elle réunit dans une
interface Gradio :

- la synthèse vocale directe avec Qwen3-TTS ;
- les 9 voix natives de Qwen et leur direction émotionnelle ;
- une bibliothèque locale de voix clonées ;
- la lecture intégrale de fichiers `.txt` et `.docx` ;
- des changements de voix, d'émotion et de pauses inscrits dans le texte ;
- un assistant vocal local associant un STT local, Ollama et Qwen3-TTS ;
- un mode speech-to-speech avec transcription et estimation d'émotion ;
- un dictionnaire de prononciation ;
- un banc de test séparé pour comparer les directions d'acteur ;
- un prototype Stable Audio séparé pour générer bruitages et ambiances ;
- une adaptation sonore automatique des romans, organisée et synchronisable
  par chapitre, avec un rapport des prompts réellement envoyés à Stable Audio.

Tout le traitement vocal est local. Aucune clé API n'est nécessaire. Ollama
doit simplement être lancé localement pour les fonctions de conversation,
de résumé et de réponse à partir d'un contexte.

Pour commencer sans parcourir toute la documentation technique, consulter le
[mode d'emploi de Voix](MODE_D_EMPLOI_VOIX.md).

## État des plateformes

| Plateforme | État actuel | Moteur |
| --- | --- | --- |
| Mac Apple Silicon (M1 à M4) | Version principale fonctionnelle | MLX / `mlx_audio` |
| Windows avec GPU NVIDIA | Backend inclus, validation matérielle Windows à effectuer | PyTorch CUDA / `qwen-tts` officiel |
| Windows sans GPU | Non pris en charge par cette version | - |
| Mac Intel | Portage CPU envisagé, pas encore inclus | PyTorch CPU |

Les modèles présents dans `models/` sont des modèles convertis pour MLX. Ils
ne fonctionnent pas sous Windows. Le backend NVIDIA télécharge automatiquement
les modèles Qwen standards depuis Hugging Face lors du premier lancement. Les
deux plateformes partagent ensuite la même interface, le même dictionnaire et
la même bibliothèque `voices_qwen.json`.

## Démarrage rapide

Sur une installation Mac déjà configurée :

```bash
source env_qwen/bin/activate
python botQwen2.py
```

Pour une nouvelle installation, suivre la section
[Installation neuve sur Mac Apple Silicon](#installation-neuve-sur-mac-apple-silicon)
ou [Installation Windows NVIDIA](#installation-windows-nvidia).

L'interface s'ouvre automatiquement sur :

```text
http://127.0.0.1:7860
```

Le chargement du STT et de Qwen3-TTS peut prendre un moment. Leur état est
affiché dans la colonne de gauche. Les deux modèles sont indépendants : le TTS
peut rester utilisable si le STT ne se charge pas.

### Modèles TTS locaux

Sur Apple Silicon, `botQwen2.py` recherche les modèles MLX dans `models/`
avec l'ordre de priorité suivant :

1. `CustomVoice` pour les voix natives et les instructions émotionnelles ;
2. `VoiceDesign` si aucun modèle CustomVoice n'est présent ;
3. `Base` pour le clonage vocal.

Dans l'installation actuelle, les deux modèles importants sont :

```text
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
└── Qwen3-TTS-12Hz-1.7B-Base-8bit/
```

L'application change de modèle en arrière-plan lorsque l'on passe d'une voix
native à une voix clonée. Il faut attendre que le statut confirme la fin du
chargement avant de relancer une synthèse.

Sous Windows, les identifiants Hugging Face par défaut sont
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` et
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

## Fonctions de `botQwen2.py`

### TTS Direct

Cet onglet transforme immédiatement un texte en fichier audio. Il accepte :

- une voix native Qwen ;
- une voix clonée de la bibliothèque ;
- les marqueurs de voix et d'émotion décrits plus bas ;
- les pauses réelles `;;` et `::` ;
- les substitutions du dictionnaire de prononciation.

Le résultat courant est écrit dans `tts_direct.wav`.

### Discussion

Cet onglet utilise un modèle local servi par Ollama, puis lit sa réponse avec
Qwen3-TTS. Il permet :

- les messages saisis au clavier ;
- une conversation au micro avec détection d'activité vocale ;
- le choix du modèle Ollama ;
- le réglage de la température ;
- plusieurs longueurs de réponse, de la phrase unique à la réponse exhaustive ;
- un system prompt personnalisé ;
- l'effacement et l'export JSON de la conversation.

Le micro se désactive pendant la lecture de la réponse afin d'éviter que le
bot ne se réécoute lui-même.

Ollama doit répondre sur son adresse locale habituelle :

```text
http://localhost:11434
```

Vérifier les modèles disponibles avec :

```bash
ollama list
```

Le modèle par défaut de repli est `mistral-nemo`.

### Fichier de contexte

Un fichier `.txt` ou `.docx` peut être chargé depuis la colonne de gauche pour :

- fournir un contexte à la conversation ;
- limiter strictement les réponses à ce contexte ;
- demander un résumé automatique ;
- lancer une lecture vocale de prévisualisation.

Le mode strict demande au modèle de signaler clairement lorsqu'une information
n'est pas présente dans le document.

### Lecture autonome

Cet onglet synthétise intégralement un ou plusieurs fichiers `.txt` ou `.docx`.
Chaque document produit un fichier WAV portant le même nom dans le dossier de
sortie choisi. Par défaut, ce dossier est `~/Downloads`.

Le texte est découpé en passages d'environ 800 caractères. Chaque passage est
écrit immédiatement sur disque pour limiter l'occupation mémoire lors de la
lecture d'un manuscrit long. La synthèse peut être arrêtée depuis l'interface.

### Speech-to-Speech

Le pipeline suit trois étapes :

1. le STT actif transcrit l'audio source ;
2. une heuristique basée sur l'énergie, la hauteur et le rythme estime une
   émotion globale ;
3. Qwen3-TTS resynthétise le texte corrigé avec la voix choisie.

Ce mode n'est pas une conversion de voix et ne reproduit pas exactement la
prosodie d'origine. Le timing, les pauses, le débit et les accents toniques ne
sont pas conservés. L'émotion proposée est un point de départ modifiable, pas
une mesure fiable de l'état émotionnel d'une personne.

### Dictionnaire de prononciation

Le lexique est enregistré dans `settings_narrateur.json`. Il permet par
exemple de remplacer :

```text
H.G.W. -> Agé Vaï
M.     -> Monsieur
```

Les correspondances sont appliquées du terme le plus long au plus court et,
quand c'est possible, sur des mots complets. Le dictionnaire agit sur le TTS
Direct et la Lecture autonome.

Le même onglet règle aussi la durée des pauses `;;` et `::`.

### Bibliothèque de voix clonées

Les profils sont décrits dans `voices_qwen.json` et leurs fichiers audio sont
conservés dans `voices/`. Pour créer un profil, il faut fournir :

- un extrait vocal propre de quelques secondes ;
- la transcription exacte de cet extrait ;
- un nom de profil.

Le modèle `Base` utilise l'audio et sa transcription pour reproduire le timbre.
Une référence neutre propre est généralement préférable à une référence
transformée artificiellement.

L'interface permet aussi d'associer à un profil des références séparées pour
`happy`, `sad`, `excited`, `curious` et `angry`. Cette méthode ne correspond
pas à une instruction émotionnelle appliquée à la voix clonée : chaque émotion
provient de son propre échantillon de référence. Si cet échantillon n'existe
pas, l'application revient à la référence neutre.

Supprimer un profil de l'interface retire son entrée de `voices_qwen.json`,
mais ne supprime pas automatiquement les fichiers audio du dossier `voices/`.

## Voix natives et émotions

Le modèle CustomVoice fournit les neuf locuteurs suivants :

| Voix | Profil général |
| --- | --- |
| `Ryan` | homme anglophone, particulièrement convaincant en français dans les essais locaux |
| `Aiden` | homme anglophone |
| `Vivian` | femme |
| `Serena` | femme |
| `Uncle_Fu` | homme |
| `Dylan` | homme |
| `Eric` | homme |
| `Ono_Anna` | femme japonaise |
| `Sohee` | femme coréenne |

Toutes peuvent parler les langues prises en charge par Qwen3-TTS, même si leur
langue native donne généralement le résultat le plus naturel. L'interface
propose : français, détection automatique, anglais, chinois, japonais, coréen,
allemand, espagnol, italien, portugais et russe.

Les émotions visibles dans l'application sont :

```text
neutral, happy, sad, excited, curious, angry
```

Ce ne sont pas les éléments d'une liste fermée imposée par Qwen. Dans
`botQwen2.py`, ce sont des raccourcis traduits en instructions naturelles pour
le modèle CustomVoice. Qwen accepte des directions d'acteur libres, par
exemple colère froide, joie fragile, tendresse, fatigue ou inquiétude masquée.
Le script de test décrit plus bas sert précisément à explorer ces nuances.

## Marqueurs dans les textes

Les marqueurs sont utilisables dans TTS Direct et Lecture autonome. Ils
s'appliquent jusqu'au marqueur suivant.

### Émotions

```text
[neutre] [joyeux] [triste] [excité] [curieux] [colère]
[neutral] [happy] [sad] [excited] [curious] [angry]
```

Exemple :

```text
[Ryan] [triste] Je pensais que tu ne reviendrais plus.
[joyeux] Mais tu es là. C'est merveilleux !
```

### Voix natives

```text
[Ryan] [Aiden] [Vivian] [Serena] [Uncle_Fu]
[Dylan] [Eric] [Ono_Anna] [Sohee]
```

### Voix clonées

La syntaxe `[voix:NomDuProfil]` sélectionne un profil de `voices_qwen.json`.
La recherche du nom ne tient pas compte des majuscules et minuscules.

```text
[voix:Homme Podalydes] Il paraît qu'elle commence à se montrer vexée.
[voix:Femme Fée] Et pourtant, personne ne lui avait rien promis.
```

Plusieurs voix clonées peuvent être alternées dans le même texte. En revanche,
il ne faut pas mélanger voix natives et voix clonées au sein d'une même
synthèse : elles reposent respectivement sur les modèles CustomVoice et Base,
et l'application ne recharge pas le modèle entre deux segments.

### Pauses réelles

```text
;;  pause moyenne
::  pause longue
```

La durée est tirée au hasard entre les bornes configurées dans l'onglet
Dictionnaire. Mettre la même valeur minimale et maximale produit une pause
fixe. Les réglages associés à `.`, `!`, `?`, `,` et `!!!` sont actuellement
enregistrés mais ne sont pas encore appliqués au signal audio.

## Direction d'acteur expérimentale

`test_actor_direction.py` est un banc de test isolé. Il ne modifie ni
`botQwen2.py`, ni l'environnement, ni la bibliothèque de voix. Chaque campagne
est écrite dans un nouveau dossier horodaté sous
`outputs/actor_direction_tests/` avec :

- les fichiers WAV ;
- un `manifest.json` décrivant les paramètres ;
- un `ratings.csv` destiné à l'écoute comparative.

Afficher toutes les options :

```bash
source env_qwen/bin/activate
python test_actor_direction.py --help
```

Comparer neutre, colère explosive et joie chaleureuse avec Ryan :

```bash
python test_actor_direction.py --suite contrast --native-speaker Ryan --native-only
```

Comparer les neuf voix natives :

```bash
python test_actor_direction.py --suite contrast --native-speaker all --native-only
```

Tester uniquement une voix clonée avec sa référence neutre :

```bash
python test_actor_direction.py --suite contrast --clone-profile "Homme Podalydes" --clone-only
```

Les directions disponibles couvrent notamment le neutre, la colère contenue,
la colère froide, la colère explosive, la joie chaleureuse, la joie fragile,
la tendresse, la fatigue, la honte, le mépris discret, l'inquiétude masquée et
le soulagement nerveux.

Les essais réalisés dans ce dossier montrent une différence importante : les
voix natives CustomVoice, notamment Ryan, répondent clairement aux instructions
de jeu ; une voix clonée par le modèle Base à partir d'une référence neutre
conserve bien son timbre, mais varie beaucoup moins émotionnellement.

## Organisation du dossier

```text
QWEN/
├── botQwen2.py                 application principale
├── botqwen_runtime.py          backends MLX et NVIDIA CUDA
├── test_actor_direction.py     comparateur de directions d'acteur
├── sfx_app.py                  interface autonome de bruitages et ambiances
├── sfx_runtime.py              adaptateur du moteur Stable Audio isole
├── scene_sfx.py                analyse de roman et montage par chapitre
├── verify_runtime.py           diagnostic sans téléchargement de modèle
├── install_windows_nvidia.ps1  installation Windows
├── start_windows_nvidia.ps1    lancement Windows
├── stop_windows_nvidia.ps1     arrêt contrôlé du serveur Windows
├── LANCER_VOIX.bat             lancement Windows par double-clic
├── ARRETER_VOIX.bat            arrêt Windows par double-clic
├── LANCER_BOTQWEN.bat          lancement Windows par double-clic
├── ARRETER_BOTQWEN.bat         arrêt Windows par double-clic
├── install_sfx_macos.sh        installation SFX sur Apple Silicon
├── start_sfx_macos.sh          lancement du Studio SFX sur Mac
├── install_sfx_windows.ps1     installation SFX TFLite sous Windows
├── start_sfx_windows.ps1       lancement du Studio SFX sous Windows
├── requirements-windows-nvidia.txt
├── STABLE_AUDIO_SFX.md         installation et utilisation du prototype SFX
├── qwen3_tts_emotif_codex.md   protocole de recherche émotionnelle
├── settings_narrateur.json     lexique et réglages des pauses
├── voices_qwen.json            index des profils clonés
├── *.example.json              exemples publiables de configuration
├── voices/                     références audio des profils
├── models/                     modèles MLX téléchargés localement
├── outputs/                    campagnes et fichiers produits
├── requirements.txt            dépendances de l'environnement Apple Silicon
└── env_qwen/                   environnement Python local existant
```

Les anciens prototypes (`main.py`, `botQwen.py`, `botQwen1.py` et les variantes
`copie`) sont exclus de Git. Le seul point d'entrée pris en charge est
`botQwen2.py`.

Les données vocales, `voices_qwen.json`, `settings_narrateur.json` et les
documents de contexte sont exclus de Git. Les fichiers `*.example.json`
montrent le format attendu sans publier les voix ni le lexique de travail.

## Installation neuve sur Mac Apple Silicon

Prérequis : un Mac Apple Silicon, Python 3.11, Git, Homebrew et environ 10 Go
libres pour les deux modèles TTS. Depuis un clone neuf :

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p models
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit
python verify_runtime.py
python botQwen2.py
```

Les deux téléchargements correspondent à :

- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` pour les voix natives ;
- `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` pour le clonage.

`requirements.txt` contient uniquement les dépendances directes. La version
`mlx-audio==0.4.2` est celle validée avec ce projet ; ses dépendances MLX et
Transformers sont résolues automatiquement par `pip`.

Ollama est facultatif pour le TTS seul, mais nécessaire pour Discussion,
Résumé automatique et les réponses fondées sur un contexte.

## Architecture multiplateforme

L'interface et la logique éditoriale restent dans `botQwen2.py`. Le module
`botqwen_runtime.py` sélectionne automatiquement le moteur :

```text
Interface et fonctions communes
├── Backend MLX     -> Mac Apple Silicon, Voxtral et modèles mlx-community
└── Backend CUDA    -> Windows NVIDIA, Whisper et modèles Qwen standards
```

Les imports lourds sont différés : PyTorch et `qwen-tts` ne sont jamais
importés sur le Mac Apple Silicon, tandis que Windows n'a pas besoin de MLX.
Le backend peut aussi être forcé avec `BOTQWEN_BACKEND=mlx` ou
`BOTQWEN_BACKEND=cuda`.

## Installation Windows NVIDIA

Pour installer le projet depuis une copie sur disque dur, sans utiliser GitHub,
consulter [INSTALLATION_WINDOWS_DISQUE.md](INSTALLATION_WINDOWS_DISQUE.md).

Le prototype autonome de bruitages est documenté dans
[STABLE_AUDIO_SFX.md](STABLE_AUDIO_SFX.md). Son moteur et ses dépendances sont
installés séparément de Qwen3-TTS.

### Prérequis

- Windows 10 ou 11 en 64 bits ;
- une carte NVIDIA compatible CUDA avec un pilote récent ;
- Python 3.12 installé depuis python.org avec le lanceur `py` ;
- Git ;
- environ 15 Go libres pour l'environnement et les modèles ;
- Ollama pour les fonctions de conversation, facultatif pour le TTS seul.

Une carte de 12 Go de VRAM ou plus est recommandée pour le modèle 1.7B, Whisper
et l'interface complète. Une carte de 8 Go peut essayer le modèle 0.6B, mais ce
dernier ne prend pas en charge les instructions émotionnelles CustomVoice.

### Installation

Dans PowerShell :

```powershell
git clone https://github.com/kapi2800/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows_nvidia.ps1
```

Le script :

1. vérifie `nvidia-smi` et Python 3.12 ;
2. crée `.venv-windows` sans toucher aux autres environnements ;
3. installe PyTorch CUDA ;
4. installe Qwen3-TTS officiel et les dépendances de l'application ;
5. exécute `verify_runtime.py`.

Le canal PyTorch par défaut est `cu128`. Pour un autre pilote :

```powershell
.\install_windows_nvidia.ps1 -CudaWheel cu126
```

### Lancement

Après l'installation initiale, double-cliquer sur :

```text
LANCER_VOIX.bat
```

Le bouton `Arrêter Voix` de l'interface ferme proprement le serveur. En cas
de navigateur bloqué, `ARRETER_VOIX.bat` cible uniquement le processus dont
le PID a été enregistré par le lanceur. Fermer le seul onglet du navigateur ne
suffit pas à arrêter l'application.

Le lancement manuel reste disponible :

```powershell
.\start_windows_nvidia.ps1
```

Le premier lancement télécharge CustomVoice 1.7B, Whisper Small puis, lors de
la première activation du clonage, Base 1.7B. Les fichiers restent dans le
cache Hugging Face de l'utilisateur.

L'installation ne se relance pas à chaque session : utilisez ensuite seulement
`start_windows_nvidia.ps1`. Le changement CustomVoice/Base est synchronisé avec
la génération afin qu'une voix clonée ne parte jamais avec le modèle natif
encore en mémoire. Le clonage reçoit aussi explicitement la langue sélectionnée.

Pour une carte disposant de moins de VRAM :

```powershell
.\start_windows_nvidia.ps1 -ModelSize 0.6B
```

Pour choisir un autre modèle Whisper :

```powershell
.\start_windows_nvidia.ps1 -SttModel "openai/whisper-large-v3-turbo"
```

### Configuration avancée Windows

| Variable | Valeur par défaut | Rôle |
| --- | --- | --- |
| `BOTQWEN_BACKEND` | sélection automatique | `mlx` ou `cuda` |
| `BOTQWEN_TTS_SIZE` | `1.7B` | taille commune de CustomVoice et Base |
| `BOTQWEN_STT_MODEL` | `openai/whisper-small` | modèle de transcription |
| `BOTQWEN_CUDA_CUSTOM_MODEL` | modèle officiel 1.7B | identifiant ou chemin CustomVoice |
| `BOTQWEN_CUDA_BASE_MODEL` | modèle officiel 1.7B | identifiant ou chemin Base |
| `BOTQWEN_CUDA_ATTENTION` | `sdpa` | moteur d'attention PyTorch |

Les chemins vers des modèles standards téléchargés manuellement peuvent être
placés dans les deux variables de modèle. Les modèles `*-8bit` présents dans
`models/` restent réservés à MLX.

### Fonctionnalités Windows

TTS Direct, les neuf voix natives, la direction émotionnelle 1.7B, le clonage,
la lecture autonome, les marqueurs, les pauses, le dictionnaire, la discussion
Ollama, le micro et Speech-to-Speech utilisent la même interface. La seule
différence fonctionnelle est le STT : Voxtral sur Apple Silicon, Whisper sous
Windows.

Le code CUDA est couvert par des tests de dispatch exécutables sans GPU, mais
la génération réelle doit encore être validée sur une machine Windows NVIDIA.
Après installation, le test complet du GPU et de la synthèse se lance avec :

```powershell
$env:BOTQWEN_BACKEND = "cuda"
.\.venv-windows\Scripts\python.exe .\verify_runtime.py --smoke-tts
```

## Futur portage Mac Intel

Le backend CPU n'est pas encore inclus. Il pourra réutiliser le backend
PyTorch et les modèles Qwen standards, probablement en 0.6B. TTS Direct et
Lecture autonome sont des objectifs réalistes en acceptant une attente longue ;
la conversation vocale en temps réel risque de ne pas être confortable.

## Limites connues

- Les émotions des voix clonées ne réagissent pas aux instructions libres du
  modèle Base ; elles exigent des références émotionnelles distinctes.
- Les voix natives et clonées ne doivent pas être alternées dans une même
  synthèse.
- Speech-to-Speech estime une émotion globale, sans transfert fin de prosodie.
- Les pauses ordinaires de ponctuation sont configurables mais pas encore
  insérées dans l'audio ; seules `;;` et `::` le sont.
- Le premier chargement d'un modèle est long et le changement entre
  CustomVoice et Base recharge plusieurs gigaoctets de poids.
- Le chemin de sortie du TTS Direct est fixe et le fichier précédent est
  remplacé à chaque nouvelle synthèse.

## Dépannage

| Symptôme | Vérification |
| --- | --- |
| `mlx_audio` introuvable | Activer `env_qwen` avant de lancer le script |
| TTS indisponible sur Mac | Vérifier la présence de CustomVoice ou Base dans `models/` |
| `torch.cuda.is_available() == False` | Réinstaller PyTorch avec le canal CUDA adapté et mettre à jour le pilote NVIDIA |
| Mémoire CUDA insuffisante | Relancer avec `-ModelSize 0.6B` ou choisir un Whisper plus petit |
| Téléchargement Hugging Face interrompu | Relancer : le cache reprend normalement le téléchargement |
| Clonage sans résultat | Vérifier l'audio de référence et sa transcription exacte |
| Émotion sans effet sur une voix clonée | Ajouter une référence pour cette émotion ou utiliser la voix neutre |
| Ollama inaccessible | Lancer Ollama et vérifier `ollama list` |
| Port `7860` occupé | Arrêter l'ancien processus ou changer `server_port` en fin de script |
| `.docx` non lu | Installer `python-docx` dans l'environnement actif |
| Micro inaccessible | Vérifier les autorisations micro de Python ou du terminal dans le système |

## Projets utilisés

- [Qwen3-TTS officiel](https://github.com/QwenLM/Qwen3-TTS)
- [MLX Audio](https://github.com/Blaizzy/mlx-audio)
- [Modèles MLX Community](https://huggingface.co/mlx-community)
- [Ollama](https://ollama.com/)
- [Gradio](https://www.gradio.app/)
- [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3)
