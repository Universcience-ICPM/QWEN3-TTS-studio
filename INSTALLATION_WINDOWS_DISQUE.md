# Installer Voix sur un PC NVIDIA depuis un disque dur

Ce guide explique comment installer Voix sous Windows a partir d'une copie
locale du dossier `QWEN`, sans passer par GitHub.

Git et GitHub ne sont pas necessaires pour cette methode. Une connexion Internet
reste indispensable pendant l'installation des dependances et le premier
telechargement des modeles.

## Configuration requise

- Windows 10 ou 11 en 64 bits ;
- une carte graphique NVIDIA compatible CUDA ;
- un pilote NVIDIA recent ;
- Python 3.12 en 64 bits avec le lanceur `py` ;
- au moins 15 Go d'espace libre ;
- une connexion Internet pour l'installation initiale.

Une carte disposant de 12 Go de VRAM ou plus est recommandee pour le modele
1.7B, Whisper et l'interface complete.

Liens officiels :

- [telecharger le pilote NVIDIA](https://www.nvidia.com/Download/Find.aspx/) ;
- [telecharger Python 3.12.10 pour Windows 64 bits](https://www.python.org/downloads/release/python-31210/) ;
- [documentation d'installation de PyTorch](https://pytorch.org/get-started/locally/).

## 1. Preparer la copie

Copier le dossier `QWEN` du disque externe vers le SSD interne du PC, par
exemple :

```text
C:\QWEN
```

Il est deconseille d'executer l'application directement depuis le disque
externe, car l'environnement Python et les modeles effectuent de nombreuses
lectures sur le disque.

Une copie complete du dossier fonctionne. Pour gagner de la place pendant le
transfert, les elements suivants, propres au Mac ou regenerables, peuvent etre
omis :

```text
env_qwen/
models/
__pycache__/
outputs/
tts_direct.wav
vad_response.wav
vad_utterance.wav
botQwen copie.py
botQwen1.py
botQwen1-copie.py
botQwen2 copie.py
```

Ne pas supprimer `botQwen2.py`, qui est l'application principale.

Les fichiers essentiels pour Windows sont :

```text
botQwen2.py
botqwen_runtime.py
install_windows_nvidia.ps1
start_windows_nvidia.ps1
stop_windows_nvidia.ps1
LANCER_BOTQWEN.bat
ARRETER_BOTQWEN.bat
LANCER_VOIX.bat
ARRETER_VOIX.bat
requirements-windows-nvidia.txt
verify_runtime.py
```

Pour ajouter le prototype autonome de bruitages et d'ambiances, conserver
egalement :

```text
sfx_app.py
sfx_runtime.py
scene_sfx.py
install_sfx_windows.ps1
start_sfx_windows.ps1
STABLE_AUDIO_SFX.md
```

Pour retrouver les voix clonees, le dictionnaire et le contexte du Mac,
conserver egalement :

```text
voices/
voices_qwen.json
settings_narrateur.json
contexte.txt
```

Les chemins des voix enregistres dans `voices_qwen.json` sont relatifs au
dossier du projet. Une copie intacte de `voices/` et `voices_qwen.json` est donc
portable entre le Mac et Windows.

## 2. Installer les prerequis

1. Installer ou mettre a jour le pilote NVIDIA.
2. Installer Python 3.12 en 64 bits.
3. Conserver le lanceur Python `py` pendant l'installation.
4. Redemarrer Windows si l'installation du pilote NVIDIA le demande.

Ouvrir ensuite PowerShell et verifier les deux commandes suivantes :

```powershell
nvidia-smi
py -3.12 --version
```

La premiere doit afficher le nom de la carte NVIDIA et la version du pilote. La
seconde doit afficher une version de Python 3.12.

## 3. Installer Voix

Dans PowerShell, se placer dans le dossier copie :

```powershell
Set-Location "C:\QWEN"
```

Autoriser les scripts uniquement pour la fenetre PowerShell actuelle :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Lancer ensuite l'installateur :

```powershell
.\install_windows_nvidia.ps1
```

Cette commande sert a l'installation initiale. Elle ne doit pas etre relancee
a chaque demarrage de Windows. Une fois `.venv-windows` cree, la commande
quotidienne est uniquement `start_windows_nvidia.ps1`.

Le script effectue les operations suivantes :

1. verification de `nvidia-smi` et de Python 3.12 ;
2. creation de l'environnement Windows `.venv-windows` ;
3. installation de PyTorch avec CUDA ;
4. installation de Qwen3-TTS et des dependances de l'application ;
5. execution du diagnostic `verify_runtime.py`.

L'environnement Mac `env_qwen` n'est ni utilise ni modifie.

Par defaut, une nouvelle execution de l'installateur conserve les versions
deja installees. Pour demander volontairement leur mise a jour :

```powershell
.\install_windows_nvidia.ps1 -Update
```

Si l'installateur signale que SoX est absent, suivre la commande indiquee par
le script, puis relancer l'installation.

## 4. Lancer l'application par double-clic

Apres l'installation initiale, il suffit de double-cliquer sur :

```text
LANCER_VOIX.bat
```

Le lanceur verifie l'environnement, demarre le serveur et ouvre le navigateur.
Il applique lui-meme la politique PowerShell necessaire : aucune commande n'est
a saisir. Une petite fenetre de suivi reste ouverte pendant l'utilisation ;
elle peut etre minimisee.

Si Voix est deja lance, un nouveau double-clic ne cree pas un second serveur
et ouvre simplement `http://127.0.0.1:7860`.

### Lancement manuel, facultatif

Toujours depuis `C:\QWEN` :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start_windows_nvidia.ps1
```

`Set-ExecutionPolicy -Scope Process Bypass` doit etre repete dans une nouvelle
fenetre PowerShell, mais **pas** l'installation. Le lanceur effectue maintenant
un diagnostic non destructif de CUDA, PyTorch, Qwen et des chemins de voix
avant d'ouvrir l'application.

Le navigateur doit s'ouvrir automatiquement sur :

```text
http://127.0.0.1:7860
```

Le premier lancement telecharge CustomVoice 1.7B et Whisper Small. Le modele
Base 1.7B est telecharge lors de la premiere utilisation d'une voix clonee.
Les modeles sont ensuite conserves dans le cache Hugging Face de l'utilisateur.

Quand la case `Voix clonee` est activee, l'application charge le modele Base.
La premiere generation attend desormais la fin de ce chargement : elle ne peut
plus partir accidentellement avec CustomVoice/Ryan. La langue choisie dans
l'interface est aussi transmise explicitement au clonage.

Les dossiers `models/` provenant du Mac contiennent des conversions MLX. Ils ne
sont pas utilises par le backend NVIDIA et ne remplacent pas les modeles
PyTorch telecharges sous Windows.

## 5. Tester une synthese complete

Une fois l'installation terminee :

```powershell
$env:BOTQWEN_BACKEND = "cuda"
.\.venv-windows\Scripts\python.exe .\verify_runtime.py --smoke-tts
```

Ce test charge Qwen3-TTS sur le GPU et genere une courte synthese. Le premier
essai peut etre long en raison du telechargement du modele.

Pour tester directement un profil clone et produire un WAV de diagnostic :

```powershell
$env:BOTQWEN_BACKEND = "cuda"
.\.venv-windows\Scripts\python.exe .\verify_runtime.py --smoke-clone "Homme Podalydes"
```

Le resultat est ecrit dans `outputs\diagnostics\`. Le diagnostic signale aussi
les profils dont l'audio ou la transcription est absent.

## Carte graphique avec moins de VRAM

Pour essayer le modele 0.6B sur une carte disposant d'environ 8 Go de VRAM :

```powershell
.\start_windows_nvidia.ps1 -ModelSize 0.6B
```

Le modele 1.7B reste recommande pour la qualite et la direction emotionnelle.

## Choisir un autre canal CUDA

L'installateur utilise `cu128` par defaut. Si PyTorch ne s'installe pas ou si
le pilote est trop ancien, essayer :

```powershell
.\install_windows_nvidia.ps1 -CudaWheel cu126
```

En dernier recours :

```powershell
.\install_windows_nvidia.ps1 -CudaWheel cu121
```

## Ollama, facultatif

Ollama n'est pas necessaire pour le TTS Direct, le clonage ou la lecture de
documents. Il est requis pour les fonctions de discussion, de resume et de
reponse a partir d'un contexte.

Apres installation d'Ollama et d'un modele local, verifier dans PowerShell :

```powershell
ollama list
```

## Problemes courants

| Probleme | Verification |
| --- | --- |
| `nvidia-smi` est introuvable | Installer ou mettre a jour le pilote NVIDIA, puis redemarrer Windows |
| `py -3.12` est introuvable | Installer Python 3.12 en conservant le lanceur `py` |
| PowerShell refuse le script | Executer `Set-ExecutionPolicy -Scope Process Bypass` dans la meme fenetre |
| CUDA est indisponible | Verifier le pilote, puis essayer `-CudaWheel cu126` |
| Memoire GPU insuffisante | Fermer les applications GPU ou lancer le modele 0.6B |
| Le premier lancement semble bloque | Attendre la fin du telechargement des modeles Hugging Face |
| Une voix clonee est absente | Verifier `voices_qwen.json` et les fichiers correspondants dans `voices/` |
| La voix clonee ressemble a Ryan ou prend un accent inattendu | Mettre a jour `botQwen2.py`, verifier que `TTS Base (clonage) pret` s'affiche, puis lancer `verify_runtime.py --smoke-clone "Nom du profil"` |
| Il faut reinstaller apres chaque redemarrage | Verifier que `C:\QWEN\.venv-windows\Scripts\python.exe` existe et que le dossier n'est pas synchronise/nettoye par OneDrive, un antivirus ou un outil de nettoyage |
| Le port 7860 est occupe | Fermer l'ancienne instance de Voix avant de relancer |

## Arreter completement l'application

Fermer uniquement l'onglet du navigateur **n'arrete pas** le serveur. Utiliser
l'une des methodes suivantes :

1. cliquer sur `Arreter Voix` dans la colonne de gauche de l'interface ;
2. double-cliquer sur `ARRETER_VOIX.bat` si le navigateur ne repond plus ;
3. fermer la fenetre de lancement ou y appuyer sur `Ctrl+C`.

Le lanceur conserve uniquement l'identifiant du processus Voix. Le raccourci
d'arret demande d'abord une fermeture propre, attend jusqu'a 15 secondes, puis
force uniquement ce processus si necessaire. Le port `7860` est ensuite libere.

## Prototype de bruitages, facultatif

Stable Audio 3 Small-SFX s'installe dans son propre dossier et son propre
environnement. Il ne modifie pas `.venv-windows` et utilise le processeur sous
Windows, laissant le GPU NVIDIA disponible pour Qwen3-TTS.

Depuis `C:\QWEN` :

```powershell
.\install_sfx_windows.ps1
.\start_sfx_windows.ps1
```

Le Studio SFX s'ouvre sur :

```text
http://127.0.0.1:7861
```

Les WAV sont crees dans `outputs\sfx\`, sur des pistes distinctes des voix.
L'onglet `Scene de roman` peut aussi creer une piste SFX par chapitre dans
`outputs\sfx_scenes\`. Ollama doit etre lance pour analyser le texte. Des
fichiers WAV ou MP3 de voix fournis dans l'ordre des chapitres permettent de
produire des pistes SFX de meme duree, pretes a etre superposees.
Une connexion Internet est requise pour la premiere installation du moteur et
du modele. Les details et la licence sont presentes dans
`STABLE_AUDIO_SFX.md`.
