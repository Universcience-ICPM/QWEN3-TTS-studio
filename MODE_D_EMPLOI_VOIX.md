# Voix - mode d'emploi

**Voix** transforme un texte en parole avec les voix natives de Qwen3-TTS ou
avec vos propres voix clonées. L'application peut aussi lire des documents,
transcrire un enregistrement et créer séparément des bruitages d'ambiance.

## 1. Démarrer Voix

### Sur Windows avec une carte NVIDIA

Après l'installation initiale, double-cliquer sur :

```text
LANCER_VOIX.bat
```

Il n'est pas nécessaire de relancer `install_windows_nvidia.ps1` à chaque
utilisation. Le navigateur s'ouvre à l'adresse :

```text
http://127.0.0.1:7860
```

### Sur Mac Apple Silicon

Ouvrir Terminal dans le dossier de Voix, puis lancer :

```bash
source env_qwen/bin/activate
python botQwen2.py
```

Attendre que la zone **Modèles** indique que le TTS est prêt. Le premier
chargement d'un modèle peut prendre un moment.

## 2. Générer rapidement une voix

1. Ouvrir l'onglet **TTS Direct**.
2. Choisir la langue, la voix et le ton émotionnel dans la colonne de gauche.
3. Saisir le texte dans **Texte à synthétiser**.
4. Pour une voix clonée, cocher **Voix clonée** et choisir son profil.
5. Cliquer sur **Synthétiser**.

Le résultat est lu dans l'interface et enregistré dans `tts_direct.wav`.

Les voix natives disponibles sont `Ryan`, `Aiden`, `Vivian`, `Serena`,
`Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna` et `Sohee`.

## 3. Créer une voix clonée

1. Ouvrir l'onglet **Voix clonée**.
2. Fournir un enregistrement propre de 5 à 15 secondes, sans musique ni bruit.
3. Écrire exactement tout ce qui est dit dans l'enregistrement.
4. Donner un nom au profil.
5. Cliquer sur **Ajouter à la bibliothèque**.

Une référence naturelle non transformée donne généralement le meilleur timbre.
Les fichiers `.wav`, `.mp3`, `.m4a` et `.ogg` sont acceptés.

### Émotions d'une voix clonée

Une voix native suit directement les instructions émotionnelles. Une voix
clonée suit surtout l'interprétation de son audio de référence. Pour obtenir
une colère ou une joie réellement marquée :

1. ouvrir **Enrichir en émotions** ;
2. choisir le profil et l'émotion ;
3. fournir un nouvel extrait de la même personne jouant cette émotion ;
4. ajouter sa transcription exacte.

Sans référence émotionnelle, Voix utilise l'extrait neutre. Les variantes
peuvent alors être très proches les unes des autres.

## 4. Lire un document entier

L'onglet **Lecture autonome** accepte un ou plusieurs fichiers `.txt` ou
`.docx`.

1. Déposer les documents.
2. Choisir le dossier de sortie.
3. Choisir une voix native ou cocher **Voix clonée**.
4. Cliquer sur **Synthétiser**.

Chaque document produit un fichier WAV portant son nom. Pour un roman, il est
préférable de fournir un fichier par chapitre afin de faciliter le montage et
la synchronisation avec les bruitages.

Ne pas mélanger une voix native et une voix clonée dans une même synthèse :
elles utilisent deux modèles différents. Il est en revanche possible
d'alterner plusieurs profils clonés.

## 5. Ajouter des indications dans le texte

Les indications restent actives jusqu'à la suivante.

```text
[Ryan] [triste] Je pensais que tu ne reviendrais plus. ;;
[joyeux] Mais tu es là !

[voix:Homme Podalydes] Je vous attendais. :: Depuis longtemps.
```

Indications émotionnelles disponibles :

```text
[neutre] [joyeux] [triste] [excité] [curieux] [colère]
```

Pauses :

```text
;;  pause moyenne
::  pause longue
```

La durée de ces pauses se règle dans l'onglet **Dictionnaire**.

## 6. Corriger une prononciation

Dans l'onglet **Dictionnaire**, indiquer le mot original puis la prononciation
souhaitée. Par exemple :

```text
H.G.W. -> Agé Vaï
M.     -> Monsieur
```

Le dictionnaire est appliqué au TTS Direct et à la Lecture autonome.

## 7. Autres fonctions

### Discussion

Écrire un message ou utiliser le micro pour converser avec un modèle Ollama
local. Ollama doit être installé, lancé et contenir au moins un modèle. Cette
fonction n'est pas nécessaire pour la synthèse vocale simple.

### Speech-to-Speech

Déposer ou enregistrer une voix, cliquer sur **Analyser**, corriger si besoin
le texte et l'émotion détectés, puis cliquer sur **Resynthétiser avec cette
voix**. Ce mode recrée le texte avec une autre voix ; il ne copie pas exactement
le rythme et l'interprétation de l'enregistrement original.

### Bruitages et ambiances

Le Studio SFX est une application séparée qui s'ouvre sur le port `7861` :

```bash
./start_sfx_macos.sh
```

Sous Windows, utiliser `start_sfx_windows.ps1`. L'onglet **Scène de roman**
analyse un fichier `.txt` ou `.docx`, crée une piste SFX par chapitre et accepte
des pistes voix WAV ou MP3 comme références de durée. Il ne génère ni dialogue
ni voix. Le rapport fourni avec le résultat contient le document de travail et
les prompts réellement transmis à Stable Audio.

## 8. Arrêter Voix

Fermer l'onglet du navigateur ne ferme pas le serveur.

- utiliser le bouton **Arrêter Voix** dans la colonne de gauche ;
- sous Windows, double-cliquer sur `ARRETER_VOIX.bat` si l'interface ne répond
  plus ;
- sur Mac, utiliser `Ctrl+C` dans le Terminal si nécessaire.

## Dépannage rapide

| Problème | Solution |
| --- | --- |
| La voix clonée ressemble à Ryan | Cocher **Voix clonée**, choisir le profil et attendre que **TTS Base (clonage) prêt** s'affiche |
| Les émotions clonées se ressemblent | Ajouter un extrait distinct et expressif pour chaque émotion |
| Le lancement Windows ne fonctionne plus | Utiliser `LANCER_VOIX.bat` ; ne relancer l'installation que si `.venv-windows` a disparu ou si le diagnostic l'exige |
| La discussion ne répond pas | Lancer Ollama et vérifier qu'un modèle local est installé |
| Le port 7860 est déjà utilisé | Arrêter l'ancienne instance avec `ARRETER_VOIX.bat` |
| Le premier essai est très long | Attendre la fin du téléchargement ou du chargement du modèle |

Pour l'installation et le diagnostic avancé, consulter [README.md](README.md)
et [INSTALLATION_WINDOWS_DISQUE.md](INSTALLATION_WINDOWS_DISQUE.md).
