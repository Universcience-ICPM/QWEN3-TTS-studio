#!/usr/bin/env python3
"""
botqwenClaude.py · Interface Gradio 6.x — STT local · Ollama LLM · Qwen3-TTS
──────────────────────────────────────────────────────────────────────────────
Version de botQwen2.py (qui reste intact) avec :
  • le TEST AUTOMATIQUE de chaque voix clonée à sa création (et de chaque voix d'émotion) :
      plusieurs réglages essayés, ressemblance et mots compris mesurés, le meilleur retenu ;
  • le RÉGLAGE de chaque voix en clair : la manière de copier (imitation / timbre seul) et
      le caractère (de « Très posée » à « Très vivante »), avec écoute avant de garder ;
  • une interface claire et épurée ;
  • une API pour l'utiliser à distance (onglet « À distance ») : un simple lien
      http://adresse:7860/dire?texte=Bonjour&voix=… fait parler la voix (mode d'emploi : API_iPhone.md).

Architecture :
  • Thread VAD continu
      → écoute micro en permanence, détecte la voix, STT, LLM, TTS
      → se tait pendant que le bot parle
  • Interface Gradio (navigateur)
      → conversation en temps réel
      → lecture autonome de fichiers .txt / .docx
      → clonage vocal local via Qwen3-TTS (sans clé API)
      → marqueurs [Ryan] [triste] [voix:Profil] directement dans le manuscrit

Lancement :
    double-clic sur botqwenClaude.command   (sur ce Mac seulement)
    double-clic sur botqwenClaude_reseau.command   (accessible depuis les autres machines du réseau)
    sur un PC Windows : LANCER_BOTQWENCLAUDE.bat  ou  LANCER_BOTQWENCLAUDE_RESEAU.bat
    ou : ./env_qwen/bin/python botqwenClaude.py
Variables d'environnement facultatives :
    BOTQWEN_RESEAU=1            ouvre l'interface et l'API au réseau local (0.0.0.0)
    BOTQWEN_MOT_DE_PASSE=…      demande un mot de passe (identifiant : voix)
    BOTQWEN_PORT=7860           le port
"""

import json
import os
import random
import re
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from botqwen_runtime import create_runtime, infer_model_type
import botqwenClaude_reglages as R

_SCRIPT_DIR = Path(__file__).resolve().parent
_runtime = create_runtime(_SCRIPT_DIR)

# ── Gradio ─────────────────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print("\n  pip install gradio --break-system-packages\n")
    raise SystemExit(1)

# ── python-docx optionnel ──────────────────────────────────────────────────
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── librosa optionnel (détection d'émotion heuristique — Speech-to-Speech) ─
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ── Dictionnaire de prononciation ─────────────────────────────────────────
PRON_FILE   = Path(__file__).parent / "settings_narrateur.json"
_pron: dict = {
    "lexicon":                {},
    "pause_strong":           500,    # . ! ?   (enregistré, non appliqué à l'audio)
    "pause_comma":            300,    # ,       (enregistré, non appliqué à l'audio)
    "pause_triple_excl":      800,    # !!!     (enregistré, non appliqué à l'audio)
    # Pauses RÉELLEMENT insérées dans l'audio — durée aléatoire entre min et max
    "pause_double_semi_min":  700,    # ;;  min
    "pause_double_semi_max":  1500,   # ;;  max
    "pause_double_colon_min": 1000,   # ::  min
    "pause_double_colon_max": 3000,   # ::  max
}
_pron_status: str = ""


def _load_pron() -> str:
    global _pron, _pron_status
    if not PRON_FILE.exists():
        _pron_status = "ℹ️ Pas de settings_narrateur.json (optionnel)"
        return _pron_status
    try:
        with open(PRON_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Rétrocompat : ancien format à valeur unique pour :: et ;;
        old_semi  = data.get("pause_double_semi")
        old_colon = data.get("pause_double_colon")
        _pron.update({
            "lexicon":                data.get("lexicon", {}),
            "pause_strong":           data.get("pause_strong",      500),
            "pause_comma":            data.get("pause_comma",       300),
            "pause_triple_excl":      data.get("pause_triple_excl", 800),
            "pause_double_semi_min":  data.get("pause_double_semi_min",
                                               old_semi  if old_semi  is not None else 700),
            "pause_double_semi_max":  data.get("pause_double_semi_max",
                                               old_semi  if old_semi  is not None else 1500),
            "pause_double_colon_min": data.get("pause_double_colon_min",
                                               old_colon if old_colon is not None else 1000),
            "pause_double_colon_max": data.get("pause_double_colon_max",
                                               old_colon if old_colon is not None else 3000),
        })
        n = len(_pron["lexicon"])
        _pron_status = f"✅ {PRON_FILE.name}  ·  {n} entrée(s) dans le lexique"
    except Exception as exc:
        _pron_status = f"⚠️ Erreur : {exc}"
    return _pron_status


def _pron_data_as_rows() -> list[list[str]]:
    """Retourne les entrées du lexique triées alphabétiquement par mot original."""
    return [[k, v] for k, v in sorted(_pron.get("lexicon", {}).items(),
                                       key=lambda x: x[0].lower())]


def apply_pronunciation(text: str) -> str:
    r"""
    Applique le lexique de prononciation.

    Deux précautions nécessaires :
      1. Tri par longueur décroissante — sans ça, une entrée courte comme
         "mora" peut modifier une partie du texte AVANT qu'une entrée plus
         longue comme "Moramanga" n'ait la chance de matcher le mot entier
         (le texte a alors déjà changé sous ses pieds).
      2. Limites de mot (\b) — sans ça, "moral" matche aussi la sous-chaîne
         "mora" à l'intérieur d'un autre mot, ou un remplacement s'applique
         à un mot qui n'a que l'air d'être le mot voulu.
    Les mots contenant des caractères non-alphanumériques aux extrémités
    (ex. "taxi-be", "H.G.W.") ne peuvent pas toujours utiliser \b (qui ne
    matche qu'entre \w et non-\w) ; on retombe alors sur un remplacement
    simple sans limite pour ces entrées-là.
    """
    lexicon = _pron.get("lexicon", {})
    for word in sorted(lexicon, key=len, reverse=True):
        repl = lexicon[word]
        pattern = re.escape(word)
        # \b ne fonctionne que si le mot commence/finit par un caractère \w
        if word and word[0].isalnum() and word[-1].isalnum():
            pattern = rf"\b{pattern}\b"
        text = re.sub(pattern, repl, text)
    return text


# ── Pauses paramétrables sur ponctuation spéciale (:: et ;;) ───────────────
# Ces marqueurs sont INSÉRÉS volontairement dans le manuscrit.
# Ils sont retirés du texte parlé et remplacés par un vrai silence dans l'audio.
# La durée est tirée au hasard entre min et max → pauses plus naturelles.
_PAUSE_MARKER_RE = re.compile(r'(::|;;)')


def _pause_silence(marker: str, sr: int) -> np.ndarray:
    """Retourne un tableau de silence (float32 mono) pour un marqueur :: ou ;;."""
    if marker == "::":
        lo = int(_pron.get("pause_double_colon_min", 1000))
        hi = int(_pron.get("pause_double_colon_max", 3000))
    else:  # ";;"
        lo = int(_pron.get("pause_double_semi_min", 700))
        hi = int(_pron.get("pause_double_semi_max", 1500))
    lo, hi = min(lo, hi), max(lo, hi)
    ms = random.randint(lo, hi) if hi > lo else lo
    n  = int(sr * ms / 1000)
    return np.zeros(max(0, n), dtype=np.float32)


def _split_on_pause_markers(text: str) -> list[tuple[str, str]]:
    """
    Découpe le texte sur les marqueurs :: et ;;.
    Retourne une liste de (texte_à_dire, marqueur_après) où marqueur ∈ {'::', ';;', ''}.
    Le marqueur lui-même est retiré du texte (il ne sera pas prononcé).
    """
    out: list[tuple[str, str]] = []
    last = 0
    for m in _PAUSE_MARKER_RE.finditer(text):
        out.append((text[last:m.start()], m.group(1)))
        last = m.end()
    out.append((text[last:], ""))
    return out


def _iter_chunk_audio(chunk_text: str, seg_emotion: str, seg_voice: str,
                      seg_clone_name: str | None, sr: int):
    """
    Génère successivement les tableaux audio d'un chunk :
      • la voix synthétisée pour chaque sous-morceau de texte,
      • un silence à chaque fois qu'un marqueur :: ou ;; est rencontré.

    seg_clone_name : nom d'un profil de la bibliothèque de voix clonées à
    utiliser pour ce segment, ou None pour une voix standard (seg_voice).
    """
    use_clone = seg_clone_name is not None
    for piece_text, marker in _split_on_pause_markers(chunk_text):
        piece_text = piece_text.strip()
        if piece_text:
            kw  = _build_synth_kwargs(piece_text, seg_emotion, seg_voice,
                                      use_clone, seg_clone_name or "")
            arr = _run_generate_audio(**kw)
            if arr is not None:
                yield np.array(arr, dtype=np.float32)
        if marker:
            sil = _pause_silence(marker, sr)
            if len(sil):
                yield sil


_load_pron()

# ══════════════════════════════════════════════════════════════════════════
#  ÉTAT GLOBAL
# ══════════════════════════════════════════════════════════════════════════

_cfg: dict = {
    "voice":       "Ryan",
    "model":       "mistral-nemo",
    "temperature": 0.7,
    "length_key":  "💬 Normal — 1 paragraphe",
    "strict":      False,
    "context":     "",
    "custom_sys":  "",
}

_history: list[dict] = []
_hist_lock = threading.Lock()

_vad_stop   = threading.Event()
_listen_ok  = threading.Event()
_listen_ok.set()
_vad_active = False
_vad_status = "⏸ En attente — cliquez sur « Démarrer l'écoute »"
_last_audio = None

_read_stop     = threading.Event()
_read_active   = False
_read_status   = "⏸ Prêt"
_read_progress = ""

# ── Modèles ────────────────────────────────────────────────────────────────
_MODELS_DIR = _SCRIPT_DIR / "models"
_STOP_REQUEST_FILE = _SCRIPT_DIR / ".botqwen-stop"


def _find_best_tts_model() -> str | Path:
    """Retourne le meilleur modèle du backend actif."""
    return _runtime.preferred_tts_source()


TTS_MODEL_PATH = _find_best_tts_model()
_TTS_MODEL_TYPE: str = infer_model_type(TTS_MODEL_PATH)

_stt             = None
_tts             = None
_tts_loaded_path : str | Path | None = None
_tts_switch_lock = threading.RLock()    # protège chargement ET génération
_models_status   = f"⏳ Chargement en cours · {_runtime.display_name}…"

# ── Fréquence d'échantillonnage TTS ────────────────────────────────────────
TTS_SR = _runtime.sample_rate


def _load_models():
    """Charge le STT puis Qwen3-TTS avec le backend de la plateforme.
    Les deux sont indépendants : si l'un échoue, l'autre continue."""
    global _stt, _tts, _tts_loaded_path, _models_status

    # ── STT ────────────────────────────────────────────────────────────────
    try:
        _models_status = f"⏳ Chargement STT ({_runtime.stt_name})…"
        print(f"[LOAD] STT {_runtime.stt_name} en cours…", flush=True)
        _stt = _runtime.load_stt()
        print("[LOAD] STT OK ✅", flush=True)
        _models_status = "✅ STT prêt  ·  ⏳ Chargement TTS…"
    except Exception as exc:
        print(f"[LOAD] STT ❌ : {exc}", flush=True)
        _models_status = f"⚠️ STT échoué ({exc})  ·  ⏳ TTS…"
        # On continue quand même — TTS peut fonctionner sans STT

    # ── TTS ────────────────────────────────────────────────────────────────
    try:
        print(f"[LOAD] TTS en cours depuis {TTS_MODEL_PATH}…", flush=True)
        _models_status = _models_status.replace("⏳ TTS…", "⏳ Chargement TTS…")
        with _tts_switch_lock:
            _tts             = _runtime.load_tts(TTS_MODEL_PATH)
            _tts_loaded_path = TTS_MODEL_PATH
        print("[LOAD] TTS OK ✅", flush=True)
        _models_status = f"✅ TTS Qwen3 prêt · {_runtime.display_name}" + (
            f"  ·  ✅ STT {_runtime.stt_name} prêt"
            if _stt else "  ·  ⚠️ STT indisponible")
    except Exception as exc:
        print(f"[LOAD] TTS ❌ : {exc}", flush=True)
        _models_status = f"❌ TTS : {exc}"


threading.Thread(target=_load_models, daemon=True).start()


def _ensure_tts_model(want_clone: bool) -> bool:
    """Charge de façon synchrone le modèle requis par la prochaine génération."""
    global _tts, _TTS_MODEL_TYPE, _tts_loaded_path, _models_status

    target_type = "base" if want_clone else "custom"
    target_path = _runtime.tts_source(target_type)
    if target_path is None:
        _models_status = f"❌ Modèle TTS {target_type} introuvable"
        print(f"[TTS SWITCH] {_models_status}", flush=True)
        return False

    with _tts_switch_lock:
        if (
            _tts is not None
            and _tts_loaded_path == target_path
            and _TTS_MODEL_TYPE == target_type
        ):
            return True

        model_label = "Base (clonage)" if want_clone else "CustomVoice (standard)"
        _models_status = f"⏳ Chargement TTS {model_label}…"
        print(f"[TTS SWITCH] → {_runtime.source_label(target_path)}", flush=True)
        try:
            if _runtime.release_before_tts_load and _tts is not None:
                previous_tts = _tts
                _tts = None
                _tts_loaded_path = None
                del previous_tts
                _runtime.release_tts()
            new_tts = _runtime.load_tts(target_path)
            _tts = new_tts
            _tts_loaded_path = target_path
            _TTS_MODEL_TYPE = infer_model_type(target_path)
            if _TTS_MODEL_TYPE != target_type:
                raise RuntimeError(
                    f"le modèle chargé est de type {_TTS_MODEL_TYPE}, attendu {target_type}"
                )
            _models_status = f"✅ TTS {model_label} prêt · {_runtime.display_name}"
            print(f"[TTS SWITCH] OK ✅  type={_TTS_MODEL_TYPE}", flush=True)
            return True
        except Exception as exc:
            _tts = None
            _tts_loaded_path = None
            _models_status = f"❌ Chargement TTS échoué : {exc}"
            print(f"[TTS SWITCH] ❌ {exc}", flush=True)
            return False


def _switch_tts_model(want_clone: bool) -> None:
    """
    Bascule le modèle TTS selon le mode voix sélectionné :
      • want_clone=True  → Base (ref_audio requis, clonage vocal)
      • want_clone=False → CustomVoice (voix nommées + instruct émotions)
    Le chargement se fait dans un thread pour ne pas bloquer l'UI.
    _models_status est mis à jour en continu → affiché via le timer Gradio.
    """
    threading.Thread(
        target=_ensure_tts_model,
        args=(want_clone,),
        daemon=True,
    ).start()


# ── Bibliothèque de voix clonées ──────────────────────────────────────────
# NOTE : _rebuild_clone_prompt() est aussi appelée après le chargement TTS
# pour rafraîchir si le fichier JSON a changé entretemps.
# Format JSON : { "NomProfil": {"ref_audio": "...", "ref_text": "..."}, ... }
# Rétrocompat : ancien format à champ unique migré automatiquement.
CLONE_FILE = Path(__file__).parent / "voices_qwen.json"

_clone_profiles : dict[str, dict] = {}   # bibliothèque complète
_clone_name     : str  = ""              # profil actif
_clone_ref_audio: str  = ""
_clone_ref_text : str  = ""
_clone_status   : str  = "⏸ Aucun profil vocal — créez-en un ci-dessous"
_use_clone      : bool = False

# ── Réglages de chaque voix clonée (banc d'essai banc_voix/banc_bibliotheque.command) ─────────────
# Le banc écrit, pour chaque voix de voices_qwen.json (et chaque voix d'émotion), une clé « reglages » :
#   température : 0.5 = voix régulière, qui ne « dérive » pas vers une autre voix ; 0.7 = plus vivante ;
#   mode_clone  : « imitation » (la voix de référence ET son texte guident la voix : le plus ressemblant)
#                 ou « timbre » (seul le timbre est gardé : pour une référence dans une autre langue).
# Une voix pas encore testée : TEMPERATURE_CLONE_DEFAUT, et le mode choisi selon la langue de sa référence.
TEMPERATURE_CLONE_DEFAUT = 0.5

_MOTS_FR = set("le la les un une des de du et est je tu il elle nous vous que qui pas ne en dans pour sur "
               "avec mais ou au aux ce cette son sa ses mon ma mes très plus été était".split())
_MOTS_EN = set("the a an and of to is was i you he she we they that this with for on in it not but "
               "my his her our there were have had be are".split())


def _langue_du_texte(texte: str) -> str:
    """« fr » ou « en », d'après les petits mots les plus courants."""
    mots = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", (texte or "").lower())
    fr = sum(m in _MOTS_FR for m in mots)
    en = sum(m in _MOTS_EN for m in mots)
    return "fr" if fr >= en else "en"


def _reglages_voix(profile_name: str, emotion: str, ref_audio: str, ref_text: str) -> tuple[float, str]:
    """(température, mode) pour la référence réellement utilisée (voix d'émotion ou voix neutre)."""
    prof = _clone_profiles.get(profile_name) or {}
    reglages = {}
    if emotion and emotion != "neutral":
        em = (prof.get("emotions") or {}).get(emotion) or {}
        if em.get("ref_audio") and _resolve_audio(em["ref_audio"]) == ref_audio:
            reglages = em.get("reglages") or {}
    if not reglages and (not ref_audio or _resolve_audio(prof.get("ref_audio", "")) == ref_audio):
        reglages = prof.get("reglages") or {}
    langue_sortie = {"french": "fr", "auto": "fr", "english": "en"}.get(str(_tts_language).lower(), "")
    try:
        temperature = float(reglages.get("temperature", TEMPERATURE_CLONE_DEFAUT))
    except (TypeError, ValueError):
        temperature = TEMPERATURE_CLONE_DEFAUT
    mode = reglages.get("mode_clone") if langue_sortie == "fr" else None   # le banc a mesuré en français
    if mode not in ("imitation", "timbre"):
        # une référence dans une autre langue que la voix fabriquée : en « imitation », elle avale
        # ou déforme des mots (banc d'essai du 7/10/2026) → on ne garde que le timbre
        mode = "imitation" if not ref_text or not langue_sortie or _langue_du_texte(ref_text) == langue_sortie \
            else "timbre"
    return temperature, mode


def _resume_reglages(profile_name: str) -> str:
    reglages = (_clone_profiles.get(profile_name) or {}).get("reglages") or {}
    if not reglages:
        return "réglage par défaut (pas encore testée)"
    return (f"{R.nom_mode(reglages.get('mode_clone') or 'imitation')} · "
            f"{R.nom_cran(reglages.get('temperature', TEMPERATURE_CLONE_DEFAUT))}"
            + (" · choisi à l'oreille" if reglages.get("fixe") else ""))


def _ordre_alphabetique(nom: str) -> tuple:
    """Tri à la française : sans tenir compte des majuscules ni des accents (« Élise » avec les E)."""
    import unicodedata
    sans_accents = "".join(c for c in unicodedata.normalize("NFD", nom) if unicodedata.category(c) != "Mn")
    return (sans_accents.casefold(), nom)


def _clone_choices() -> list[str]:
    return sorted(_clone_profiles.keys(), key=_ordre_alphabetique)


def _save_clone_file() -> None:
    try:
        CLONE_FILE.write_text(
            json.dumps(_clone_profiles, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as exc:
        print(f"[Clone] Sauvegarde échouée : {exc}")


def _rebuild_clone_prompt() -> None:
    """Charge la bibliothèque depuis voices_qwen.json (migre l'ancien format)."""
    global _clone_profiles, _clone_name, _clone_ref_audio, _clone_ref_text, _clone_status
    if not CLONE_FILE.exists():
        return
    try:
        data = json.loads(CLONE_FILE.read_text(encoding="utf-8"))
        # Migration ancien format (champ "ref_audio" à la racine)
        if "ref_audio" in data:
            name = data.get("name", "Profil 1")
            _clone_profiles = {name: {"ref_audio": data["ref_audio"],
                                      "ref_text":  data["ref_text"]}}
            _save_clone_file()
        else:
            _clone_profiles = {k: v for k, v in data.items()
                               if isinstance(v, dict)}
        # Sélectionne le premier profil valide
        for name, prof in _clone_profiles.items():
            ra = prof.get("ref_audio", "")
            rt = prof.get("ref_text",  "")
            ra_abs = _resolve_audio(ra)
            if ra and Path(ra_abs).exists() and rt:
                _clone_name      = name
                _clone_ref_audio = ra_abs   # absolu pour le TTS
                _clone_ref_text  = rt
                _clone_status    = f"✅ {len(_clone_profiles)} profil(s) · « {name} » actif"
                return
    except Exception as exc:
        _clone_status = f"⚠️ Rechargement échoué : {exc}"


def fn_select_clone(name: str):
    """Sélectionne un profil de la bibliothèque."""
    global _clone_name, _clone_ref_audio, _clone_ref_text, _clone_status
    if not name or name not in _clone_profiles:
        return _clone_status
    prof = _clone_profiles[name]
    _clone_name      = name
    _clone_ref_audio = prof.get("ref_audio", "")
    _clone_ref_text  = prof.get("ref_text",  "")
    _clone_status    = f"✅ Profil « {name} » sélectionné · {_resume_reglages(name)}"
    return _clone_status


_VOICES_DIR  = _SCRIPT_DIR / "voices"


def _resolve_audio(path: str) -> str:
    """Convertit un chemin audio relatif en absolu (base = dossier du script).
    Un chemin déjà absolu est retourné tel quel."""
    p = Path(path)
    return str(p if p.is_absolute() else _SCRIPT_DIR / p)


def _copy_to_voices(src: str, dest_stem: str) -> str:
    """
    Copie src (chemin temp Gradio ou autre) dans voices/ sous le nom dest_stem + extension.
    Retourne un chemin RELATIF au script (ex : voices/NomProfil.wav) pour stockage JSON.
    Si la copie échoue, retourne src en fallback.
    """
    try:
        _VOICES_DIR.mkdir(exist_ok=True)
        ext  = Path(src).suffix or ".wav"
        dest = _VOICES_DIR / f"{dest_stem}{ext}"
        if Path(src).resolve() != dest.resolve():   # évite de copier sur soi-même
            shutil.copy2(src, dest)
        return str(dest.relative_to(_SCRIPT_DIR))   # ← relatif
    except Exception as exc:
        print(f"[voices] Copie échouée ({exc}) — chemin temp conservé", flush=True)
        return src


def _ajouter_profil(audio_file, ref_text: str, voice_name: str) -> tuple[str, str | None]:
    """Ajoute un profil vocal à la bibliothèque → (message, nom du profil ou None si refusé).
    Copie le fichier audio dans voices/ pour garantir un chemin stable entre sessions."""
    global _clone_name, _clone_ref_audio, _clone_ref_text, _clone_status
    if _tts is None:
        return "❌ La voix Qwen n'est pas encore chargée — patientez quelques secondes.", None
    if not audio_file:
        return "❌ Ajoutez d'abord l'enregistrement de la voix.", None
    if not (ref_text or "").strip():
        return "❌ Écrivez mot pour mot ce qui est dit dans l'enregistrement.", None
    try:
        name        = (voice_name or "").strip() or Path(str(audio_file)).stem
        stable_path = _copy_to_voices(str(audio_file), name)      # relatif
        _clone_profiles[name] = {"ref_audio": stable_path, "ref_text": ref_text.strip()}
        _clone_name      = name
        _clone_ref_audio = _resolve_audio(stable_path)             # absolu pour le TTS
        _clone_ref_text  = ref_text.strip()
        _save_clone_file()
        _clone_status = f"✅ « {name} » ajoutée · {len(_clone_profiles)} voix dans la bibliothèque"
        return _clone_status, name
    except Exception as exc:
        _clone_status = f"❌ {exc}"
        return _clone_status, None


def fn_create_clone(audio_file, ref_text: str, voice_name: str):
    """Ajoute la voix, puis la teste aussitôt pour lui trouver son réglage (générateur : l'interface
    suit la progression). Sorties : statut, 3 listes de profils, progression du test, panneau de réglage."""
    rien = gr.update()
    message, nom = _ajouter_profil(audio_file, ref_text, voice_name)
    if not nom:
        yield (message, rien, rien, rien, "") + _panneau_inchange()
        return
    dd = gr.update(choices=_clone_choices(), value=nom)
    yield (message, dd, dd, dd, "⏳ Test de la voix…") + _panneau(nom, "")
    for progression in _tester_voix(nom, ""):
        yield (rien, rien, rien, rien, progression) + _panneau_inchange()
    yield (f"✅ « {nom} » ajoutée et réglée", rien, rien, rien, _derniere_progression) + _panneau(nom, "")


def fn_delete_clone(name: str):
    """Supprime un profil de la bibliothèque."""
    global _clone_name, _clone_ref_audio, _clone_ref_text, _clone_status
    _dd_empty = gr.update()
    if not name or name not in _clone_profiles:
        return f"⚠️ Profil introuvable : {name}", _dd_empty, _dd_empty, _dd_empty
    del _clone_profiles[name]
    _save_clone_file()
    if _clone_name == name:
        _clone_name = _clone_ref_audio = _clone_ref_text = ""
        for n, p in _clone_profiles.items():
            ra_abs = _resolve_audio(p.get("ref_audio", ""))
            if p.get("ref_audio") and Path(ra_abs).exists():
                _clone_name      = n
                _clone_ref_audio = ra_abs   # absolu pour le TTS
                _clone_ref_text  = p["ref_text"]
                break
    choices  = _clone_choices()
    _clone_status = (f"🗑 « {name} » supprimé · {len(choices)} profil(s)"
                     + (f" · « {_clone_name} » actif" if _clone_name else ""))
    dd = gr.update(choices=choices, value=_clone_name or None)
    return _clone_status, dd, dd, dd   # clone_dd, tts_clone_dd, read_clone_dd


def fn_add_emotion_sample(profile_name: str, emotion: str,
                          audio_file, ref_text: str) -> tuple[str, str]:
    """Ajoute (ou remplace) un sample d'émotion dans un profil existant.
    Copie le fichier audio dans voices/ pour garantir un chemin stable entre sessions."""
    if not profile_name or profile_name not in _clone_profiles:
        return f"⚠️ Profil introuvable : {profile_name!r}", ""
    if not audio_file:
        return "❌ Aucun fichier audio fourni", ""
    if not ref_text.strip():
        return "❌ Transcription manquante", ""
    prof        = _clone_profiles[profile_name]
    dest_stem   = f"{profile_name}-{emotion}"
    stable_path = _copy_to_voices(str(audio_file), dest_stem)
    emotions    = prof.setdefault("emotions", {})
    emotions[emotion] = {"ref_audio": stable_path, "ref_text": ref_text.strip()}
    _save_clone_file()
    status = f"✅ Version « {_NOMS_EMOTIONS.get(emotion, emotion)} » ajoutée à « {profile_name} »"
    return status, _emotion_coverage(profile_name)


def fn_add_emotion_and_test(profile_name: str, emotion: str, audio_file, ref_text: str):
    """Ajoute la version émotion, puis la teste (générateur). Sorties : statut, couverture,
    profil affiché dans le panneau de réglage, progression du test, panneau de réglage."""
    rien = gr.update()
    status, couverture = fn_add_emotion_sample(profile_name, emotion, audio_file, ref_text)
    if not status.startswith("✅"):
        yield (status, couverture or rien, rien, "") + _panneau_inchange()
        return
    yield (status, couverture, gr.update(value=profile_name), "⏳ Test de la version…") \
        + _panneau(profile_name, emotion)
    for progression in _tester_voix(profile_name, emotion):
        yield (rien, rien, rien, progression) + _panneau_inchange()
    yield (status + " et réglée", rien, rien, _derniere_progression) + _panneau(profile_name, emotion)


def fn_remove_emotion_sample(profile_name: str, emotion: str) -> tuple[str, str]:
    """Supprime le sample d'émotion d'un profil."""
    if not profile_name or profile_name not in _clone_profiles:
        return f"⚠️ Profil introuvable : {profile_name!r}", ""
    prof = _clone_profiles[profile_name]
    if emotion in prof.get("emotions", {}):
        del prof["emotions"][emotion]
        _save_clone_file()
    return f"🗑 Sample « {emotion} » supprimé de « {profile_name} »", \
           _emotion_coverage(profile_name)


def fn_select_clone_for_emotions(name: str) -> str:
    """Rafraîchit l'affichage de couverture émotionnelle lors du changement de profil."""
    return _emotion_coverage(name)


# ══════════════════════════════════════════════════════════════════════════
#  TEST ET RÉGLAGE DE CHAQUE VOIX CLONÉE
# ══════════════════════════════════════════════════════════════════════════
#
# À sa création, chaque voix (et chaque version émotion) est testée : la même phrase est fabriquée en
# « imitation » et en « timbre seul », à trois caractères (posée, vivante, très vivante). Pour chaque
# essai on mesure la RESSEMBLANCE avec l'enregistrement (empreinte de la voix, calculée par Qwen) et les
# MOTS COMPRIS (la reconnaissance vocale réécoute). Le réglage retenu est écrit dans voices_qwen.json
# (clé « reglages »), avec les résultats ; un essai de chaque réglage est gardé dans voices/essais/
# pour l'écouter. Dans l'interface, on peut ensuite changer le réglage à l'oreille et le garder
# (« fixe » : le banc d'essai de la bibliothèque n'y touchera plus).

_NOMS_EMOTIONS = {"neutral": "neutre", "happy": "joyeuse", "sad": "triste", "excited": "enthousiaste",
                  "curious": "curieuse", "angry": "en colère"}
_ESSAIS_DIR = _SCRIPT_DIR / "voices" / "essais"
_test_stop = threading.Event()
_test_lock = threading.Lock()            # un seul test à la fois
_derniere_progression = ""
PHRASE_ECOUTE = ("Bonjour ! Asseyez-vous, je vais vous raconter une histoire que personne ne connaît, "
                 "et vous me direz ce que vous en pensez.")


PRINCIPALE = "principale"      # la valeur affichée pour la voix principale (en interne : "")


def _version(version) -> str:
    return "" if version in (None, "", PRINCIPALE) else str(version)


def _cible(nom: str, version: str) -> dict:
    """Le dictionnaire de la voix principale (version "") ou d'une version émotion."""
    version = _version(version)
    prof = _clone_profiles.get(nom) or {}
    if version:
        return (prof.get("emotions") or {}).get(version) or {}
    return prof


def _ecrire_reglages(nom: str, version: str, reglages: dict) -> None:
    """Écrit les réglages d'une voix dans voices_qwen.json. Le fichier est relu juste avant : un autre
    programme (le banc d'essai, le podcast…) a pu le changer entre-temps."""
    version = _version(version)
    try:
        sur_disque = json.loads(CLONE_FILE.read_text(encoding="utf-8")) if CLONE_FILE.exists() else {}
    except Exception:
        sur_disque = {}
    for n, p in _clone_profiles.items():               # les profils créés dans cette session
        if isinstance(p, dict) and n not in sur_disque:
            sur_disque[n] = p
    prof = sur_disque.get(nom)
    if not isinstance(prof, dict):
        prof = sur_disque[nom] = dict(_clone_profiles.get(nom) or {})
    if version:
        em = (prof.setdefault("emotions", {})).setdefault(version, dict(_cible(nom, version)))
        em["reglages"] = reglages
    else:
        prof["reglages"] = reglages
    provisoire = CLONE_FILE.with_suffix(".tmp")
    provisoire.write_text(json.dumps(sur_disque, ensure_ascii=False, indent=2), encoding="utf-8")
    provisoire.replace(CLONE_FILE)
    _clone_profiles.clear()
    _clone_profiles.update({k: v for k, v in sur_disque.items() if isinstance(v, dict)})


def _dossier_essais(nom: str, version: str) -> Path:
    return _ESSAIS_DIR / R.nom_fichier(nom) / (_version(version) or "neutre")


def _fichier_essai(nom: str, version: str, mode: str, temperature: float) -> Path:
    return _dossier_essais(nom, version) / f"{mode}_{float(temperature):.1f}.wav"


def _vers_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Rééchantillonnage simple (pour la reconnaissance vocale, qui travaille à 16 kHz)."""
    if sr == 16000 or audio.size == 0:
        return audio.astype(np.float32)
    n = int(round(audio.size * 16000 / sr))
    return np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(np.float32)


def _mots_entendus(audio: np.ndarray) -> str | None:
    """Ce que la reconnaissance vocale entend (None si elle n'est pas chargée)."""
    if _stt is None:
        return None
    chemin = Path(_TTS_TMP_DIR) / "test_voix_stt.wav"
    sf.write(str(chemin), _vers_16k(audio, TTS_SR), 16000)
    try:
        return _stt.generate(str(chemin)).text.strip()
    except Exception as exc:
        print(f"[Test voix] reconnaissance vocale : {exc}", flush=True)
        return None


def _empreinteur():
    """Une fonction audio → empreinte de la voix (MLX seulement ; None ailleurs)."""
    fonction = getattr(_tts, "extract_speaker_embedding", None)
    if fonction is None:
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None

    def empreinte(audio: np.ndarray) -> np.ndarray:
        with _tts_switch_lock:
            e = np.asarray(fonction(mx.array(np.asarray(audio, np.float32))), np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)
    return empreinte


def _lire_reference(chemin: str) -> np.ndarray:
    try:                                   # MLX : le même lecteur que Qwen (lit aussi les .mp3 et .m4a)
        from mlx_audio.utils import load_audio
        return np.asarray(load_audio(chemin, sample_rate=TTS_SR), np.float32).reshape(-1)
    except Exception:
        pass
    data, sr = sf.read(chemin, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TTS_SR:
        n = int(round(data.size * TTS_SR / sr))
        data = np.interp(np.linspace(0, data.size - 1, n), np.arange(data.size), data).astype(np.float32)
    return data


def _fabriquer(nom: str, version: str, texte: str, mode: str, temperature: float,
               langue: str | None = None) -> np.ndarray | None:
    """Fait dire un texte à une voix clonée avec un réglage précis (sans toucher à son réglage)."""
    cible = _cible(nom, version)
    ref_audio = _resolve_audio(cible.get("ref_audio", ""))
    if not cible.get("ref_audio") or not Path(ref_audio).exists():
        return None
    ref_text = cible.get("ref_text", "")
    return _run_generate_audio(text=texte, ref_audio=ref_audio,
                               ref_text=ref_text if mode == "imitation" and ref_text.strip() else None,
                               temperature=float(temperature), language=langue or _tts_language)


def _tester_voix(nom: str, version: str):
    """Teste une voix (générateur : renvoie au fur et à mesure une ligne de progression)."""
    global _derniere_progression
    version = _version(version)
    etiquette = nom + (f" ({_NOMS_EMOTIONS.get(version, version)})" if version else "")
    cible = _cible(nom, version)
    ref_audio = _resolve_audio(cible.get("ref_audio", ""))
    ref_text = cible.get("ref_text", "")
    if not cible.get("ref_audio") or not Path(ref_audio).exists():
        _derniere_progression = f"⚠️ {etiquette} : enregistrement introuvable, pas de test."
        yield _derniere_progression
        return
    if not _test_lock.acquire(blocking=False):
        _derniere_progression = "⚠️ Un test est déjà en cours : attendez qu'il se termine."
        yield _derniere_progression
        return
    try:
        _test_stop.clear()
        if not _ensure_tts_model(True):
            _derniere_progression = "❌ La voix Qwen (modèle de clonage) n'a pas pu être chargée."
            yield _derniere_progression
            return
        empreinte = _empreinteur()
        reference = None
        if empreinte:
            try:
                reference = empreinte(_lire_reference(ref_audio))
            except Exception as exc:
                print(f"[Test voix] empreinte de la référence impossible : {exc}", flush=True)
                empreinte = None
        modes = ["imitation", "timbre"] if ref_text.strip() else ["timbre"]
        plan = [(t, m, p) for t in R.TEST_TEMPERATURES for m in modes for p in R.TEST_PHRASES]
        dossier = _dossier_essais(nom, version)
        dossier.mkdir(parents=True, exist_ok=True)
        essais, debut = [], time.time()
        for k, (t, mode, phrase) in enumerate(plan, 1):
            if _test_stop.is_set():
                break
            reste = ""
            if k > 1:
                reste = f" · encore ≈ {max(1, round((time.time() - debut) / (k - 1) * (len(plan) - k + 1)))} s"
            _derniere_progression = (f"⏳ Test de « {etiquette} » : essai {k}/{len(plan)} — "
                                     f"{R.nom_mode(mode)} · {R.nom_cran(t)}{reste}")
            yield _derniere_progression
            audio = _fabriquer(nom, version, phrase, mode, t, langue="French")   # le test est en français
            if audio is None or audio.size == 0:
                continue
            audio = np.asarray(audio, np.float32).reshape(-1)
            if phrase == R.TEST_PHRASES[0]:
                sf.write(str(_fichier_essai(nom, version, mode, t)), audio, TTS_SR)
            entendu = _mots_entendus(audio)
            try:
                ressemblance = float(empreinte(audio) @ reference) if empreinte else None
            except Exception as exc:
                print(f"[Test voix] empreinte impossible : {exc}", flush=True)
                ressemblance = None
            essais.append({"temperature": t, "mode": mode,
                           "ressemblance": ressemblance,
                           "erreurs": R.taux_erreur(phrase, entendu) if entendu is not None else None})
        if not essais:
            _derniere_progression = f"⏹ Test de « {etiquette} » arrêté avant le premier essai : réglage inchangé."
            yield _derniere_progression
            return
        choix, bilans = R.choisir(essais)
        if _test_stop.is_set():
            choix["verdict"] += " (test interrompu)"
        _ecrire_reglages(nom, version, dict(choix, bilans=bilans))
        mesures = [] if empreinte else ["ressemblance non mesurée sur cette machine"]
        if _stt is None:
            mesures.append("mots compris non mesurés (reconnaissance vocale pas encore chargée)")
        reglage = f"{R.nom_mode(choix['mode_clone'])} · {R.nom_cran(choix['temperature'])}"
        if _test_stop.is_set():
            _derniere_progression = (f"⏹ Test de « {etiquette} » arrêté après {len(essais)} essai(s) sur "
                                     f"{len(plan)} → réglage provisoire : {reglage}")
        else:
            _derniere_progression = (f"✅ « {etiquette} » testée en {round(time.time() - debut)} s → {reglage}"
                                     + (f" ({' ; '.join(mesures)})" if mesures else ""))
        yield _derniere_progression
    finally:
        _test_lock.release()


# Les résultats détaillés du banc d'essai de toute la bibliothèque (banc_voix/banc_bibliotheque.command),
# quand il a été lancé : on les affiche pour les voix qui n'ont pas été testées ici.
_RESULTATS_BANC = _SCRIPT_DIR.parent / "AVATAR-Claude" / "banc_voix" / "bibliotheque" / "resultats.json"
_banc_cache: dict = {"mtime": None, "donnees": {}}


def _bilans_du_banc(nom: str, version: str) -> list:
    try:
        mtime = _RESULTATS_BANC.stat().st_mtime
    except OSError:
        return []
    if _banc_cache["mtime"] != mtime:
        try:
            _banc_cache["donnees"] = json.loads(_RESULTATS_BANC.read_text(encoding="utf-8"))
        except Exception:
            _banc_cache["donnees"] = {}
        _banc_cache["mtime"] = mtime
    cle = f"{nom}|{version}" if version else nom
    return (_banc_cache["donnees"].get(cle) or {}).get("bilans") or []


def _versions(nom: str) -> list[tuple[str, str]]:
    """Les versions d'une voix : la principale, puis ses versions émotion."""
    prof = _clone_profiles.get(nom) or {}
    sortie = [("Voix principale", PRINCIPALE)]
    for em in (prof.get("emotions") or {}):
        sortie.append((f"Version {_NOMS_EMOTIONS.get(em, em)}", em))
    return sortie


def _reglage_affiche(nom: str, version: str) -> tuple[str, float]:
    reglages = _cible(nom, version).get("reglages") or {}
    mode = reglages.get("mode_clone") if reglages.get("mode_clone") in R.MODES else None
    if mode is None:
        ref_text = _cible(nom, version).get("ref_text", "")
        mode = "imitation" if ref_text.strip() else "timbre"
    return mode, R.cran_le_plus_proche(reglages.get("temperature", TEMPERATURE_CLONE_DEFAUT))


def _essai_existant(nom: str, version: str, mode: str, t: float) -> str | None:
    f = _fichier_essai(nom, version, mode, t)
    return str(f) if f.exists() else None


def _texte_cran(t: float) -> str:
    return f"**{R.nom_cran(t)}** — {R.explication_cran(t)}"


def _texte_mode(mode: str) -> str:
    return f"**{R.nom_mode(mode)}** — {R.MODES.get(mode, ('', ''))[1]}"


def _panneau(nom: str, version: str) -> tuple:
    """Le panneau « Réglage de la voix » pour une voix : version, réglage actuel, résultats du test,
    manière de copier, caractère (et son explication), essai à écouter."""
    if not nom or nom not in _clone_profiles:
        vide = "_Choisissez une voix dans la bibliothèque._"
        return (gr.update(choices=[("Voix principale", PRINCIPALE)], value=PRINCIPALE), vide,
                "_Pas de résultats._", gr.update(), gr.update(), _texte_mode("imitation"),
                _texte_cran(TEMPERATURE_CLONE_DEFAUT), None)
    version = _version(version)
    versions = _versions(nom)
    if (version or PRINCIPALE) not in [v for _, v in versions]:
        version = ""
    reglages = _cible(nom, version).get("reglages") or {}
    mode, t = _reglage_affiche(nom, version)
    bilans = reglages.get("bilans") or _bilans_du_banc(nom, version)
    return (gr.update(choices=versions, value=version or PRINCIPALE),
            R.resume_reglage(reglages, TEMPERATURE_CLONE_DEFAUT),
            R.tableau_bilans(bilans, reglages) if bilans else
            "_Pas encore de résultats détaillés pour cette voix : « Refaire le test » les fabrique "
            "(une minute environ) et garde un essai de chaque réglage à écouter._",
            gr.update(value=mode), gr.update(value=t), _texte_mode(mode), _texte_cran(t),
            _essai_existant(nom, version, mode, t))


def _panneau_inchange() -> tuple:
    return tuple(gr.update() for _ in range(8))


def fn_panneau(nom: str, version: str = ""):
    return _panneau(nom, _version(version))


def fn_choix_reglage(nom: str, version: str, mode: str, t: float):
    """On change la manière ou le caractère : l'explication suit, et l'essai déjà fabriqué
    pour ce réglage (s'il existe) est prêt à écouter."""
    t = R.cran_le_plus_proche(t)
    essai = _essai_existant(nom, _version(version), mode, t) if nom else None
    note = "" if essai else "\n\n_Pas encore d'essai à ce réglage : cliquez sur « Écouter avec ce réglage »._"
    return _texte_mode(mode), _texte_cran(t) + note, essai


def fn_ecouter_reglage(nom: str, version: str, mode: str, t: float, phrase: str):
    if not nom or nom not in _clone_profiles:
        return None, "❌ Choisissez d'abord une voix."
    audio = _fabriquer(nom, _version(version), (phrase or "").strip() or PHRASE_ECOUTE, mode,
                       R.cran_le_plus_proche(t))
    if audio is None:
        return None, "❌ La voix n'a pas pu être fabriquée (voir le terminal)."
    sortie = _nom_enregistrement(nom, (phrase or "").strip() or PHRASE_ECOUTE,
                                 f"{R.nom_mode(mode)} {R.nom_cran(t)}")
    sf.write(str(sortie), np.asarray(audio, np.float32).reshape(-1), TTS_SR)
    return str(sortie), f"🎧 {R.nom_mode(mode)} · {R.nom_cran(t)}"


def fn_garder_reglage(nom: str, version: str, mode: str, t: float):
    if not nom or nom not in _clone_profiles:
        return ("❌ Choisissez d'abord une voix.",) + _panneau_inchange()
    version = _version(version)
    avant = _cible(nom, version).get("reglages") or {}
    reglages = {"temperature": R.cran_le_plus_proche(t), "mode_clone": mode, "fixe": True,
                "verdict": "choisi à l'oreille", "teste_le": time.strftime("%Y-%m-%d")}
    if avant.get("bilans"):
        reglages["bilans"] = avant["bilans"]
    _ecrire_reglages(nom, version, reglages)
    return (f"✅ Réglage gardé pour « {nom} » : {R.nom_mode(mode)} · {R.nom_cran(t)}",) + _panneau(nom, version)


def fn_refaire_test(nom: str, version: str):
    if not nom or nom not in _clone_profiles:
        yield ("❌ Choisissez d'abord une voix.",) + _panneau_inchange()
        return
    for progression in _tester_voix(nom, _version(version)):
        yield (progression,) + _panneau_inchange()
    yield (_derniere_progression,) + _panneau(nom, _version(version))


def fn_arreter_test():
    _test_stop.set()
    return "⏹ Arrêt demandé : le test s'arrête après l'essai en cours (le réglage est choisi sur ce qui a été fait)."


def fn_toggle_clone(enabled: bool):
    """Active / désactive le mode voix clonée (mode conversation VAD).
    Déclenche aussi le changement de modèle TTS en arrière-plan."""
    global _use_clone, _clone_status
    _use_clone = enabled
    _switch_tts_model(want_clone=enabled)
    if enabled:
        if not _clone_ref_audio:
            _clone_status = "⚠️ Aucun profil — créez-en un dans l'onglet Voix clonée"
        else:
            _clone_status = f"🎤 Voix clonée active — « {_clone_name} »"
    else:
        _clone_status = "💻 Voix standard — Qwen3 CustomVoice"
    return _clone_status


def fn_toggle_clone_visibility(v: bool):
    """Gère la visibilité du dropdown de profil cloné ET déclenche le switch de modèle.
    Utilisé dans TTS Direct et Lecture autonome."""
    _switch_tts_model(want_clone=v)
    return gr.update(visible=v)


def _get_ref_for_emotion(profile_name: str, emotion: str) -> tuple[str, str]:
    """
    Retourne (ref_audio, ref_text) pour un profil et une émotion donnés.

    Logique :
      1. Cherche un sample d'émotion spécifique dans prof["emotions"][emotion]
      2. Si absent ou fichier inexistant → tombe sur le sample neutre du profil
      3. Si profil inexistant → retourne les globals actuels (_clone_ref_audio/text)
    """
    if not profile_name or profile_name not in _clone_profiles:
        return _clone_ref_audio, _clone_ref_text
    prof = _clone_profiles[profile_name]
    # Tente le sample spécifique à l'émotion
    if emotion and emotion != "neutral":
        em_data = prof.get("emotions", {}).get(emotion, {})
        ra = em_data.get("ref_audio", "")
        rt = em_data.get("ref_text",  "")
        ra_abs = _resolve_audio(ra)
        if ra and Path(ra_abs).exists() and rt:
            return ra_abs, rt   # absolu pour le TTS
    # Fallback : sample neutre du profil (résolu en absolu)
    return _resolve_audio(prof.get("ref_audio", _clone_ref_audio)), \
           prof.get("ref_text", _clone_ref_text)


def _emotion_coverage(profile_name: str) -> str:
    """Résumé des émotions couvertes par un profil (pour l'UI)."""
    if not profile_name or profile_name not in _clone_profiles:
        return ""
    prof = _clone_profiles[profile_name]
    emotions = prof.get("emotions", {})
    covered = [e for e in EMOTIONS if e != "neutral" and e in emotions
               and Path(_resolve_audio(emotions[e].get("ref_audio", ""))).exists()]
    if not covered:
        return "⚠️ Aucun sample d'émotion — marqueurs [triste] [joyeux] sans effet"
    labels = {
        "happy": "joyeux 😊", "sad": "triste 😢",
        "excited": "excité ⚡", "curious": "curieux 🤔", "angry": "colère 😤",
    }
    ok   = " · ".join(labels.get(e, e) for e in covered)
    miss = [labels.get(e, e) for e in EMOTIONS if e != "neutral" and e not in covered]
    out  = f"✅ Couverts : {ok}"
    if miss:
        out += f"\n⬜ Manquants : {', '.join(miss)}"
    return out


def _build_synth_kwargs(text: str, seg_emotion: str, seg_voice: str,
                        use_clone: bool, clone_profile_name: str) -> dict:
    """
    Construit les kwargs pour _run_generate_audio.

    Voix clonée  → ref_audio + ref_text spécifiques à l'émotion si disponibles,
                   sinon sample neutre. La langue de sortie reste forcée par
                   l'interface ; voice/instruct ne sont pas transmis au modèle Base.
    Voix standard → voice (CustomVoice uniquement) + instruct (émotions) + language forcé.
    """
    kw: dict = {"text": text}
    if use_clone and clone_profile_name:
        ref_audio, ref_text = _get_ref_for_emotion(clone_profile_name, seg_emotion)
        if ref_audio:
            temperature, mode = _reglages_voix(clone_profile_name, seg_emotion, ref_audio, ref_text)
            kw["ref_audio"]   = ref_audio
            kw["ref_text"]    = ref_text if mode == "imitation" else None   # « timbre » : sans le texte
            kw["temperature"] = temperature
            kw["language"]    = _tts_language
            return kw      # émotion portée par le sample, langue forcée par l'interface
    # ── Voix standard (ou fallback si aucun ref_audio résolu) ─────────────────
    # _run_generate_audio garantit que CustomVoice sera chargé avant l'appel.
    kw["voice"] = seg_voice
    instruct = _EMOTION_TO_INSTRUCT.get(seg_emotion, "") or None
    if instruct:
        kw["instruct"] = instruct
    # language : forcé en français pour une meilleure prosodie
    kw["language"] = _tts_language
    return kw


# ── Locuteurs Qwen3-TTS ────────────────────────────────────────────────────
VOICES = [
    "Ryan",       # homme anglophone — bon pour le français
    "Aiden",      # homme anglophone
    "Vivian",     # femme
    "Serena",     # femme
    "Uncle_Fu",   # homme
    "Dylan",      # homme
    "Eric",       # homme
    "Ono_Anna",   # femme japonaise
    "Sohee",      # femme coréenne
]

# ── Langue TTS ─────────────────────────────────────────────────────────────
LANGUAGES = [
    "French", "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "Spanish", "Italian", "Portuguese", "Russian",
]
_tts_language: str = "French"

# ── Émotions → instruction naturelle (paramètre instruct de Qwen3-TTS) ─────
EMOTIONS = ["neutral", "happy", "sad", "excited", "curious", "angry"]

_EMOTION_TO_INSTRUCT: dict[str, str] = {
    "neutral":  "",
    "happy":    "Speak in a warm, joyful, upbeat tone.",
    "sad":      "Speak in a sad, melancholic, subdued tone.",
    "excited":  "Speak in an excited, energetic, enthusiastic tone.",
    "curious":  "Speak in a curious, thoughtful, inquisitive tone.",
    "angry":    "Speak in a loud, sharp, aggressive tone, as if shouting in anger.",
}

_active_emotion: str = "neutral"

# ── Longueur intelligente ──────────────────────────────────────────────────
LONGUEUR: dict[str, str] = {
    "⚡ Flash — 1 phrase":
        "Réponds en UNE seule phrase courte, sans liste ni paragraphe.",
    "✂️ Bref — 2 à 3 phrases":
        "Réponds en 2 ou 3 phrases. Va droit au but, pas d'introduction.",
    "💬 Normal — 1 paragraphe":
        "Réponds en un paragraphe naturel de 4 à 6 phrases.",
    "📖 Développé — 2 à 3 paragraphes":
        "Développe ta réponse en 2 ou 3 paragraphes bien structurés.",
    "📚 Exhaustif — réponse complète":
        "Réponds de façon complète et approfondie, autant de paragraphes que nécessaire.",
    "🎲 Libre — sans contrainte": "",
}

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MAX_HIST        = 12


# ══════════════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════

def _push(role: str, content: str):
    with _hist_lock:
        _history.append({"role": role, "content": content})


def _get_history() -> list[dict]:
    with _hist_lock:
        return list(_history)


def get_ollama_models() -> list[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return names if names else ["mistral-nemo"]
    except Exception:
        return ["mistral-nemo"]


def build_system(cfg: dict) -> str:
    length_instr = LONGUEUR.get(cfg["length_key"], "")
    custom  = (cfg.get("custom_sys") or "").strip()
    context = cfg.get("context") or ""
    strict  = cfg.get("strict", False)

    if custom:
        sys = custom
    elif strict and context:
        sys = (
            "Tu es un assistant vocal francophone.\n"
            "Réponds UNIQUEMENT en te basant sur le contexte ci-dessous. "
            "Si une information n'y figure pas, dis-le clairement.\n\n"
            f"=== CONTEXTE ===\n{context}\n=== FIN ==="
        )
    else:
        sys = "Tu es un assistant vocal francophone. Sois naturel et précis."
        if context:
            sys += f"\n\nContexte :\n{context[:4000]}"
    if length_instr:
        sys += f"\n\n{length_instr}"
    return sys


def call_ollama(user_text: str, history: list[dict], cfg: dict) -> str:
    system   = build_system(cfg)
    recent   = history[-(MAX_HIST * 2):]
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]}
                 for m in recent if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": user_text})
    resp = requests.post(
        OLLAMA_URL,
        json={"model": cfg["model"], "messages": messages, "stream": False,
              "options": {"temperature": cfg["temperature"]}},
        timeout=90,
    )
    resp.raise_for_status()
    return (resp.json().get("message", {}).get("content") or "").strip()


# ── Synthèse TTS Qwen3 ────────────────────────────────────────────────────

# Dossier temp persistant pour la durée de la session :
# créé une seule fois, réutilisé chunk après chunk → évite le coût mkdir/rmdir répété.
_TTS_TMP_DIR: str = tempfile.mkdtemp(prefix="qwen_tts_session_")


def _run_generate_audio(**kwargs) -> np.ndarray | None:
    """Génère un tableau audio avec le backend actif."""
    try:
        want_clone = bool(kwargs.get("ref_audio"))
        expected_type = "base" if want_clone else "custom"
        with _tts_switch_lock:
            if not _ensure_tts_model(want_clone):
                return None
            if _TTS_MODEL_TYPE != expected_type:
                raise RuntimeError(
                    f"modèle TTS {_TTS_MODEL_TYPE} actif, {expected_type} requis"
                )
            return _runtime.generate_audio(
                model=_tts,
                model_type=_TTS_MODEL_TYPE,
                temp_dir=_TTS_TMP_DIR,
                kwargs=kwargs,
            )
    except Exception as exc:
        print(f"[Qwen TTS generate] {exc}", flush=True)
        return None


def _qwen_synth(text: str, voice: str, emotion: str = "") -> np.ndarray | None:
    """Synthèse voix standard — délègue à _build_synth_kwargs."""
    kw = _build_synth_kwargs(text, emotion, voice, False, "")
    return _run_generate_audio(**kw)


def _qwen_clone_synth(text: str, emotion: str = "",
                      profile_name: str = "") -> np.ndarray | None:
    """
    Synthèse voix clonée.
    L'émotion est exprimée via le sample de référence (ref_audio),
    PAS via instruct (ignoré par le modèle Base en mode clonage).
    """
    name = profile_name or _clone_name
    kw = _build_synth_kwargs(text, emotion, "", True, name)
    if "ref_audio" not in kw:
        return None    # aucun profil résolvable
    return _run_generate_audio(**kw)


def _nom_enregistrement(voix: str, texte: str, precision: str = "") -> Path:
    """Un nom de fichier différent pour chaque son fabriqué, lisible une fois téléchargé
    (sur un téléphone, tous les sons s'appelaient pareil) :
        Femme_centenaire_Bonjour_Louise_ca_va_2026-10-15_17h32m05.wav"""
    debut = " ".join(re.sub(r"\[[^\]]*\]|::|;;", " ", texte or "").split()[:5])
    morceaux = [R.nom_fichier(voix or "voix")[:30], R.nom_fichier(debut)[:40] if debut else "",
                R.nom_fichier(precision) if precision else "", time.strftime("%Y-%m-%d_%Hh%Mm%S")]
    nom = "_".join(m for m in morceaux if m)
    dossier = Path(_TTS_TMP_DIR) / "enregistrements"
    dossier.mkdir(parents=True, exist_ok=True)
    chemin, k = dossier / f"{nom}.wav", 2
    while chemin.exists():
        chemin, k = dossier / f"{nom}-{k}.wav", k + 1
    return chemin


def synthesize(text: str, voice: str, out: str = "bot_answer.wav") -> str | None:
    """Synthèse TTS → fichier WAV (un nom différent à chaque fois). Utilise la voix clonée si activée."""
    if _use_clone and _clone_ref_audio:
        audio = _qwen_clone_synth(text, _active_emotion)
    else:
        audio = _qwen_synth(text, voice, _active_emotion)
    if audio is None:
        return None
    try:
        chemin = _nom_enregistrement(_clone_name if (_use_clone and _clone_ref_audio) else voice, text)
        sf.write(str(chemin), audio, TTS_SR)
        return str(chemin)
    except Exception as exc:
        print(f"[TTS write] {exc}")
        return None


def read_file(raw) -> tuple[str, str]:
    if raw is None:
        return "", "Aucun fichier"
    path = Path(str(raw))
    try:
        if path.suffix.lower() == ".docx":
            if not HAS_DOCX:
                return "", "❌ pip install python-docx --break-system-packages"
            doc  = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
    except Exception as exc:
        return "", f"❌ {exc}"
    return text, f"✅ {path.name} · {len(text):,} caractères"


# ══════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-SPEECH  (STT → détection émotion heuristique → resynthèse TTS)
# ══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT — lire avant utilisation :
# Ceci n'est PAS un vrai transfert de prosodie (voice conversion). Qwen3-TTS
# ne "copie" pas le timing/l'intonation exacte du fichier source. Le pipeline
# transcrit l'audio, estime une émotion GLOBALE et grossière à partir
# du signal (énergie, variance de pitch, débit), puis resynthétise le texte
# avec la voix choisie et ce marqueur d'émotion. Le phrasé fin de l'original
# (pauses, débit exact, accents toniques) n'est pas préservé — seul le
# contenu + une tonalité approximative le sont.

_S2S_EMOTION_LABELS = {
    "neutral": "😐 neutre", "happy": "😊 joyeux", "sad": "😢 triste",
    "excited": "⚡ excité", "curious": "🤔 curieux", "angry": "😤 colère",
}


def _estimate_emotion_from_audio(path: str) -> tuple[str, str]:
    """
    Heuristique grossière d'émotion globale à partir du signal audio (via librosa) :
      • énergie RMS moyenne         → indicateur d'intensité
      • variance du pitch (F0)      → indicateur d'expressivité / questionnement
      • débit approximatif (ZCR)    → indicateur de rythme

    Retourne (emotion, note_explicative). Best-effort — jamais fiable à 100 %,
    affiché comme point de départ éditable, pas comme vérité absolue.
    """
    if not HAS_LIBROSA:
        return "neutral", "ℹ️ librosa non installé — émotion par défaut (neutre)"
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        if y.size == 0:
            return "neutral", "⚠️ Audio vide"

        rms = float(np.mean(librosa.feature.rms(y=y)))
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        pitch_std = float(np.nanstd(f0_voiced)) if f0_voiced.size > 3 else 0.0
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Seuils empiriques, à ajuster à l'usage — pas de calibration formelle.
        # ZCR élevé (friction/débit dur) est le signal le plus distinctif de la
        # colère — testé en priorité, avant "curious", pour éviter qu'un pitch
        # variable classe à tort un ton colérique en "curieux".
        if zcr > 0.15 and rms > 0.04:
            emotion = "angry"
        elif rms > 0.08 and pitch_std > 40:
            emotion = "excited"
        elif rms < 0.02 and pitch_std < 15:
            emotion = "sad"
        elif pitch_std > 35:
            emotion = "curious"
        elif rms > 0.05:
            emotion = "happy"
        else:
            emotion = "neutral"

        note = (f"RMS={rms:.3f} · variance pitch={pitch_std:.1f} · ZCR={zcr:.3f} "
                f"— estimation grossière, à corriger si besoin")
        return emotion, note
    except Exception as exc:
        return "neutral", f"⚠️ Détection échouée ({exc}) — neutre par défaut"


def fn_s2s_transcribe(audio_path):
    """
    Étape 1 : transcrit l'audio source + estime une émotion globale.
    Renvoie (texte_transcrit, emotion_détectée, note_détection, statut).
    """
    if audio_path is None:
        return "", "neutral", "", "❌ Aucun fichier audio fourni"
    if _stt is None:
        return "", "neutral", "", f"❌ STT ({_runtime.stt_name}) pas encore chargé — patientez…"

    try:
        text = _stt.generate(str(audio_path)).text.strip()
    except Exception as exc:
        return "", "neutral", "", f"❌ Transcription échouée : {exc}"

    if not text:
        return "", "neutral", "", "⚠️ Transcription vide — audio inaudible ?"

    emotion, note = _estimate_emotion_from_audio(str(audio_path))
    status = f"✅ Transcrit · {len(text)} caractères · émotion estimée : {_S2S_EMOTION_LABELS.get(emotion, emotion)}"
    return text, emotion, note, status


def fn_s2s_resynthesize(text: str, emotion: str, voice: str,
                        use_clone: bool, clone_name: str):
    """
    Étape 2 : resynthétise le texte (édité ou non) avec la voix/le profil choisi
    et le marqueur d'émotion (détecté ou corrigé manuellement).
    Réutilise le même chemin que fn_tts_direct — marqueurs [Ryan]/[triste] etc.
    restent utilisables si l'utilisateur les ajoute à la main dans le texte.
    """
    if not text or not text.strip():
        return None, "❌ Aucun texte à synthétiser"

    global _active_emotion
    prev_emotion = _active_emotion
    _active_emotion = emotion if emotion in EMOTIONS else "neutral"
    try:
        out_path = fn_tts_direct(text, voice, use_clone, clone_name)
    finally:
        _active_emotion = prev_emotion

    if out_path is None:
        return None, "❌ Synthèse échouée"
    voice_label = f"🎤 {clone_name}" if use_clone and clone_name else voice

    # Avertit si l'émotion choisie n'a en réalité aucun effet sur ce résultat :
    #  • voix clonée sans sample dédié à cette émotion → fallback silencieux au sample neutre
    #  • voix standard hors modèle CustomVoice → instruct jamais envoyé au TTS
    emotion_ignored = False
    if use_clone and clone_name:
        prof = _clone_profiles.get(clone_name, {})
        has_sample = emotion != "neutral" and emotion in prof.get("emotions", {})
        if emotion != "neutral" and not has_sample:
            emotion_ignored = True
    elif _TTS_MODEL_TYPE != "custom":
        emotion_ignored = True

    status = f"✅ Resynthétisé avec {voice_label} · émotion demandée : {_S2S_EMOTION_LABELS.get(emotion, emotion)}"
    if emotion_ignored:
        if use_clone:
            status += (f"\n⚠️ Ignorée : le profil « {clone_name} » n'a pas de sample "
                       f"« {emotion} » — sample neutre utilisé à la place. "
                       f"Ajoutez-en un dans l'onglet Voix clonée.")
        else:
            status += (f"\n⚠️ Ignorée : le modèle TTS chargé est « {_TTS_MODEL_TYPE} », "
                       f"pas CustomVoice — les marqueurs d'émotion sur voix standard "
                       f"nécessitent le modèle CustomVoice.")
    return out_path, status


# ══════════════════════════════════════════════════════════════════════════
#  BOUCLE VAD
# ══════════════════════════════════════════════════════════════════════════

def _process_utterance(audio: np.ndarray, sr: int):
    global _last_audio, _vad_status
    if _stt is None:
        return
    tmp_wav = "vad_utterance.wav"
    sf.write(tmp_wav, audio, sr)
    try:
        user_text = _stt.generate(tmp_wav).text.strip()
    except Exception as exc:
        print(f"[STT] {exc}")
        return
    if not user_text:
        return

    _push("user", f"🎙 {user_text}")
    _vad_status = "🧠 Réflexion…"

    cfg  = dict(_cfg)
    hist = _get_history()
    try:
        answer = call_ollama(user_text, hist, cfg)
    except Exception as exc:
        _push("assistant", f"⚠️ LLM : {exc}")
        _vad_status = "👂 Écoute active…"
        return

    _push("assistant", answer)
    _vad_status = "🔊 Synthèse vocale…"

    _listen_ok.clear()
    try:
        out_path = synthesize(answer, cfg["voice"], "vad_response.wav")
        if out_path:
            _last_audio = out_path
            data, rate = sf.read(out_path)
            sd.play(data, rate)
            sd.wait()
    except Exception as exc:
        print(f"[TTS/Playback] {exc}")
    finally:
        _listen_ok.set()
        _vad_status = "👂 Écoute active…"


def _vad_loop():
    global _vad_active, _vad_status

    SR      = 16000
    FRAME_S = int(SR * 30 / 1000)
    MIN_SP  = int(300 / 30)
    SIL_F   = int(600 / 30)
    MAX_F   = int(8.0 * 1000 / 30)

    _vad_status = "🎙 Calibration du micro (1 s — ne parlez pas)…"
    noise = []
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=FRAME_S) as stream:
            for _ in range(int(1000 / 30)):
                fr, _ = stream.read(FRAME_S)
                noise.append(float(np.mean(fr.reshape(-1) ** 2)))
    except Exception as exc:
        _vad_status = f"❌ Micro inaccessible : {exc}"
        _vad_active = False
        return

    threshold = float(np.mean(noise)) * 5.0
    _vad_status = "👂 Écoute active…"

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=FRAME_S) as stream:
            buf, started, sp_cnt, sil_cnt = [], False, 0, 0

            while not _vad_stop.is_set():
                fr, _ = stream.read(FRAME_S)
                fr    = fr.reshape(-1)

                if not _listen_ok.is_set():
                    buf, started, sp_cnt, sil_cnt = [], False, 0, 0
                    continue

                buf.append(fr)
                e = float(np.mean(fr ** 2))

                if not started:
                    sp_cnt = sp_cnt + 1 if e > threshold else 0
                    if sp_cnt >= MIN_SP:
                        started  = True
                        sil_cnt  = 0
                        buf      = buf[-MIN_SP:]
                        _vad_status = "🔴 Enregistrement…"
                else:
                    sil_cnt = sil_cnt + 1 if e < threshold else 0
                    if sil_cnt >= SIL_F or len(buf) >= MAX_F:
                        audio = np.concatenate(buf)
                        buf, started, sp_cnt, sil_cnt = [], False, 0, 0
                        _vad_status = "⚙️ Traitement…"
                        _process_utterance(audio, SR)

    except Exception as exc:
        _vad_status = f"❌ Erreur VAD : {exc}"
    finally:
        _vad_active = False
        _vad_status = "⏸ Arrêté"


# ══════════════════════════════════════════════════════════════════════════
#  CALLBACKS GRADIO
# ══════════════════════════════════════════════════════════════════════════

def fn_toggle_vad():
    global _vad_active, _vad_status
    if not _vad_active:
        if _stt is None or _tts is None:
            return (gr.update(value="▶ Démarrer l'écoute", variant="primary"),
                    "❌ Modèles pas encore chargés — patientez…")
        _vad_stop.clear()
        _listen_ok.set()
        _vad_active = True
        threading.Thread(target=_vad_loop, daemon=True).start()
        return (gr.update(value="⏹ Arrêter l'écoute", variant="stop"),
                "🎙 Calibration…")
    else:
        _vad_stop.set()
        _listen_ok.set()
        sd.stop()
        _vad_active = False
        _vad_status = "⏸ Arrêté"
        return (gr.update(value="▶ Démarrer l'écoute", variant="primary"),
                "⏸ Arrêté")


def fn_sync_config(voice, model, temp, length, strict, custom_sys):
    _cfg.update({
        "voice":       voice or "Ryan",
        "model":       model or "mistral-nemo",
        "temperature": temp  if temp  is not None else 0.7,
        "length_key":  length or "💬 Normal — 1 paragraphe",
        "strict":      bool(strict),
        "custom_sys":  custom_sys or "",
    })


def _cfg_badge() -> str:
    ctx    = _cfg.get("context", "")
    ctx_info = f" · 📄 {len(ctx):,} car." if ctx else ""
    strict   = " · 🔒 Strict" if _cfg.get("strict") else ""
    clone_tag = f" · 🎤 {_clone_name}" if _use_clone and _clone_ref_audio else ""
    return (
        f"🧠 **{_cfg['model']}**"
        f"  ·  🎙 **{_cfg['voice']}**"
        f"  ·  📏 **{_cfg['length_key'].split('—')[0].strip()}**"
        f"  ·  🌡 {_cfg['temperature']:.2f}"
        f"{ctx_info}{strict}{clone_tag}"
    )


def fn_poll() -> tuple:
    return (
        _get_history(),
        _vad_status,
        _models_status,
        _cfg_badge(),
        _read_status,
        _read_progress,
    )


def fn_shutdown_server():
    """Ferme Gradio après avoir laissé le temps à la réponse UI de partir."""
    def request_stop():
        _STOP_REQUEST_FILE.touch()

    threading.Timer(0.5, request_stop).start()


def fn_set_emotion(emotion: str):
    global _active_emotion
    _active_emotion = emotion if emotion in EMOTIONS else "neutral"


def fn_set_language(lang: str):
    global _tts_language
    _tts_language = lang if lang in LANGUAGES else "French"


# ── Lecture autonome ───────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 800) -> list[str]:
    sentences, chunks, cur = [], [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for s in line.replace("? ", "?|||").replace("! ", "!|||").replace(". ", ".|||").split("|||"):
            s = s.strip()
            if s:
                sentences.append(s)
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur += s + " "
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s + " "
    if cur:
        chunks.append(cur.strip())
    return chunks


_EMOTION_MAP: dict[str, str] = {
    "neutral":   "neutral", "neutre":   "neutral",
    "happy":     "happy",   "joyeux":   "happy",   "joyeuse": "happy",
    "heureux":   "happy",   "heureuse": "happy",
    "sad":       "sad",     "triste":   "sad",
    "excited":   "excited", "excité":   "excited", "excitée": "excited",
    "curious":   "curious", "curieux":  "curious", "curieuse":"curious",
    "angry":     "angry",   "colère":   "angry",   "en colère":"angry",
}

_VOICE_SET = {v.lower() for v in VOICES}

# Tri du plus long au plus court → évite qu'un préfixe masque un terme plus long
_MARKER_TERMS = sorted(
    list(_EMOTION_MAP.keys()) + VOICES,
    key=len, reverse=True
)
# Marqueurs fixes ([Ryan], [triste]…) OU marqueur dynamique [voix:NomDuProfil]
# pour cibler un profil de la bibliothèque de voix clonées (voices_qwen.json).
# Le nom du profil n'est pas connu à la compilation (bibliothèque modifiable
# en cours d'usage) — d'où un groupe séparé plutôt qu'une liste figée.
_MARKER_RE = re.compile(
    r'[\[\（【]\s*(?:voix\s*:\s*(?P<clone>[^\]\）】]+?)|(?P<fixed>'
    + '|'.join(re.escape(t) for t in _MARKER_TERMS)
    + r'))\s*[\]\）】]',
    flags=re.IGNORECASE | re.UNICODE,
)

# Caractères invisibles que Word peut insérer
_INVISIBLE = re.compile(r'[\u200b\u200c\u200d\ufeff\u00ad]')


def _normalize_text(text: str) -> str:
    """Supprime les caractères invisibles et normalise les apostrophes/guillemets."""
    import unicodedata
    text = _INVISIBLE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return text


def _parse_segments(text: str, default_voice: str,
                    default_clone: str | None = None) -> list[tuple[str, str, str | None, str]]:
    """
    Découpe le texte en segments (émotion, voix, profil_cloné, texte) selon
    les marqueurs.

    Marqueurs d'émotion : [neutre] [joyeux] [triste] [excité] [curieux] [colère]
                          [neutral] [happy] [sad] [excited] [curious] [angry]
    Marqueurs de voix standard : [Ryan] [Aiden] [Vivian] [Serena] [Uncle_Fu] …
    Marqueur de voix clonée   : [voix:NomDuProfil] — cible un profil de la
                                bibliothèque (voices_qwen.json), insensible à
                                la casse. Si le profil n'existe pas, un
                                avertissement est loggé et la voix en cours
                                (standard ou clonée) est conservée.

    Combinables dans n'importe quel ordre : [Ryan] [triste], [voix:Marielle] [triste]
    S'appliquent jusqu'au prochain marqueur.

    IMPORTANT — limite technique : basculer entre [voix:Profil] (clonage,
    modèle "Base") et [Ryan]-style (modèle "CustomVoice") en cours de texte
    NE recharge PAS le modèle TTS à la volée (trop lent). Utilisez
    [voix:Profil1] / [voix:Profil2] entre profils clonés dans un même texte,
    mais ne mélangez pas avec les voix standard dans le même passage.

    default_clone : nom du profil cloné actif au tout début du texte, avant
    tout marqueur — reflète la case "Voix clonée" + le profil sélectionné
    dans l'UI. None = voix standard par défaut.
    """
    text  = _normalize_text(text)
    segments: list[tuple[str, str, str | None, str]] = []
    current_emotion = _active_emotion
    current_voice   = default_voice
    current_clone   = default_clone

    pos = 0
    for m in _MARKER_RE.finditer(text):
        chunk_text = text[pos:m.start()]
        if chunk_text.strip():
            for chunk in _chunk_text(chunk_text):
                if chunk.strip():
                    segments.append((current_emotion, current_voice, current_clone, chunk))
        pos = m.end()

        clone_target = m.group("clone")
        fixed_term   = m.group("fixed")

        if clone_target:
            clone_target = clone_target.strip()
            match = next((n for n in _clone_profiles if n.lower() == clone_target.lower()), None)
            if match:
                current_clone = match
                print(f"[Marker] voix clonée → {match}", flush=True)
            else:
                print(f"[Marker] ⚠️ profil cloné introuvable : « {clone_target} » "
                      f"— voix inchangée", flush=True)
        elif fixed_term:
            low = fixed_term.strip().lower()
            if low in _EMOTION_MAP:
                current_emotion = _EMOTION_MAP[low]
                print(f"[Marker] émotion → {current_emotion}", flush=True)
            elif low in _VOICE_SET:
                current_voice = next((v for v in VOICES if v.lower() == low),
                                     current_voice)
                current_clone = None   # bascule explicitement sur voix standard
                print(f"[Marker] voix → {current_voice}", flush=True)

    tail = text[pos:]
    if tail.strip():
        for chunk in _chunk_text(tail):
            if chunk.strip():
                segments.append((current_emotion, current_voice, current_clone, chunk))

    if not segments:
        print("[Marker] Aucun marqueur trouvé — lecture brute.", flush=True)
        return [(current_emotion, current_voice, current_clone, c) for c in _chunk_text(text)]
    return segments


def _reading_thread(paths: list[str], voice: str, out_dir: Path,
                    use_clone: bool, clone_profile_name: str):
    """
    Lit chaque chunk → écrit immédiatement sur disque → libère la mémoire.
    L'émotion sur voix clonée est résolue via _get_ref_for_emotion()
    (sample de référence spécifique à l'émotion, pas via instruct).
    """
    import gc
    global _read_active, _read_status, _read_progress
    total_files = len(paths)
    SR          = TTS_SR
    saved_files: list[str] = []
    default_clone = clone_profile_name if (use_clone and clone_profile_name) else None

    for fi, fpath in enumerate(paths):
        if _read_stop.is_set():
            break

        src = Path(fpath)
        text, read_err = read_file(fpath)
        if not text:
            _read_status   = f"⚠️ {src.name} — {read_err}"
            _read_progress = ""
            gc.collect()
            continue

        text     = apply_pronunciation(text)
        segments = _parse_segments(text, voice, default_clone)
        n_chunks = len(segments)

        out_wav   = out_dir / f"{src.stem}.wav"
        _read_status = f"🔊 Synthèse : {src.name}"

        wav_out   = None
        n_written = 0

        try:
            for ci, (seg_emotion, seg_voice, seg_clone, chunk_text) in enumerate(segments):
                if _read_stop.is_set():
                    break

                voice_label = f"🎤 {seg_clone}" if seg_clone else seg_voice
                _read_progress = (
                    f"Fichier {fi + 1}/{total_files}  ·  "
                    f"Chunk {ci + 1}/{n_chunks}  [{voice_label} · {seg_emotion}]"
                )
                try:
                    produced = False
                    for arr in _iter_chunk_audio(
                            chunk_text, seg_emotion, seg_voice, seg_clone, SR):
                        produced = True
                        if wav_out is None:
                            channels = 1 if arr.ndim == 1 else arr.shape[1]
                            wav_out  = sf.SoundFile(
                                str(out_wav), mode="w",
                                samplerate=SR, channels=channels,
                                subtype="PCM_16")
                        # Aligne le silence (mono) sur un éventuel audio stéréo
                        if wav_out.channels > 1 and arr.ndim == 1:
                            arr = np.column_stack([arr] * wav_out.channels)
                        wav_out.write(arr)
                        n_written += len(arr)
                        del arr
                    if not produced:
                        print(f"[TTS] chunk {ci+1}/{n_chunks} vide "
                              f"({voice_label}·{seg_emotion})", flush=True)
                except Exception as exc:
                    print(f"[TTS] chunk {ci+1}/{n_chunks} : {exc}", flush=True)
                finally:
                    gc.collect()

        finally:
            if wav_out is not None:
                wav_out.close()

        if _read_stop.is_set():
            break

        if n_written > 0:
            saved_files.append(str(out_wav))
            _read_status   = f"✅ {fi + 1}/{total_files} sauvegardé"
            _read_progress = "\n".join(saved_files)
        else:
            _read_status = f"⚠️ {src.name} — aucun audio généré"

        gc.collect()

    _read_active = False
    if _read_stop.is_set():
        _read_status = "⏹ Arrêtée"
    elif saved_files:
        _read_status = f"✅ Terminé — {len(saved_files)} fichier(s) sauvegardé(s)"
    else:
        _read_status = "⚠️ Aucun fichier sauvegardé"


def fn_start_reading(files, voice: str, out_dir: str,
                     use_clone: bool, clone_name: str):
    global _read_active, _read_status, _read_progress
    if not files:
        return "❌ Aucun fichier sélectionné", ""

    out_path = Path(out_dir.strip()) if out_dir and out_dir.strip() else Path.home() / "Downloads"
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return f"❌ Dossier de sortie inaccessible : {exc}", ""

    if not isinstance(files, list):
        files = [files]
    paths = [str(f) for f in files if f]
    if not paths:
        return "❌ Fichiers invalides", ""

    _read_stop.clear()
    _read_active   = True
    _read_status   = "⏳ Démarrage…"
    _read_progress = ""

    threading.Thread(
        target=_reading_thread,
        args=(paths, voice, out_path, use_clone, clone_name or ""),
        daemon=True,
    ).start()
    return _read_status, _read_progress


def fn_stop_reading():
    global _read_active
    _read_stop.set()
    sd.stop()
    _read_active = False
    return "⏹ Arrêtée", ""


def fn_add_lexicon_entry(mot: str, pron: str):
    """Ajoute ou remplace une entrée dans le lexique et sauvegarde immédiatement."""
    mot  = mot.strip()
    pron = pron.strip()
    if not mot:
        return "⚠️ Mot vide", _pron_data_as_rows(), mot, pron
    if not pron:
        return "⚠️ Prononciation vide", _pron_data_as_rows(), mot, pron
    action = "mis à jour" if mot in _pron["lexicon"] else "ajouté"
    _pron["lexicon"][mot] = pron
    _save_pron()
    return (f"✅ « {mot} = {pron} » {action} · {len(_pron['lexicon'])} entrée(s)",
            _pron_data_as_rows(), "", "")   # efface les champs après ajout


def fn_delete_lexicon_entry(mot: str):
    """Supprime une entrée du lexique et sauvegarde immédiatement."""
    mot = mot.strip()
    if not mot:
        return "⚠️ Mot vide", _pron_data_as_rows(), mot
    if mot not in _pron["lexicon"]:
        return f"⚠️ « {mot} » introuvable dans le lexique", _pron_data_as_rows(), mot
    del _pron["lexicon"][mot]
    _save_pron()
    return (f"🗑 « {mot} » supprimé · {len(_pron['lexicon'])} entrée(s)",
            _pron_data_as_rows(), "")


def _save_pron():
    """Sauvegarde _pron dans settings_narrateur.json (préserve les autres clés)."""
    base: dict = {}
    if PRON_FILE.exists():
        try:
            with open(PRON_FILE, encoding="utf-8") as f:
                base = json.load(f)
        except Exception:
            pass
    base.update({
        "pause_strong":           _pron.get("pause_strong",           500),
        "pause_comma":            _pron.get("pause_comma",            300),
        "pause_triple_excl":      _pron.get("pause_triple_excl",      800),
        "pause_double_semi_min":  _pron.get("pause_double_semi_min",  700),
        "pause_double_semi_max":  _pron.get("pause_double_semi_max",  1500),
        "pause_double_colon_min": _pron.get("pause_double_colon_min", 1000),
        "pause_double_colon_max": _pron.get("pause_double_colon_max", 3000),
        "lexicon":                _pron.get("lexicon", {}),
    })
    with open(PRON_FILE, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=4)


def fn_save_lexicon(df_val, p_strong, p_comma, p_texcl,
                    p_dsemi_min, p_dsemi_max, p_dcolon_min, p_dcolon_max):
    global _pron
    rows = list(df_val) if df_val else []
    lexicon: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2:
            mot  = str(row[0]).strip() if row[0] else ""
            pron = str(row[1]).strip() if row[1] else ""
            if mot and pron:
                lexicon[mot] = pron

    base: dict = {}
    if PRON_FILE.exists():
        try:
            with open(PRON_FILE, encoding="utf-8") as f:
                base = json.load(f)
        except Exception:
            pass

    # Retire les anciennes clés à valeur unique si présentes (migration)
    base.pop("pause_double_semi",  None)
    base.pop("pause_double_colon", None)

    base.update({
        "pause_strong":           int(p_strong     or 500),
        "pause_comma":            int(p_comma      or 300),
        "pause_triple_excl":      int(p_texcl      or 800),
        "pause_double_semi_min":  int(p_dsemi_min  or 700),
        "pause_double_semi_max":  int(p_dsemi_max  or 1500),
        "pause_double_colon_min": int(p_dcolon_min or 1000),
        "pause_double_colon_max": int(p_dcolon_max or 3000),
        "lexicon":                lexicon,
    })
    try:
        with open(PRON_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=4)
        _pron.update({
            "lexicon":                lexicon,
            "pause_strong":           base["pause_strong"],
            "pause_comma":            base["pause_comma"],
            "pause_triple_excl":      base["pause_triple_excl"],
            "pause_double_semi_min":  base["pause_double_semi_min"],
            "pause_double_semi_max":  base["pause_double_semi_max"],
            "pause_double_colon_min": base["pause_double_colon_min"],
            "pause_double_colon_max": base["pause_double_colon_max"],
        })
        return f"✅ {len(lexicon)} entrée(s) · pauses sauvegardées dans {PRON_FILE.name}"
    except Exception as exc:
        return f"⚠️ Erreur : {exc}"


def fn_reload_pron():
    status = _load_pron()
    return (
        status,
        _pron_data_as_rows(),
        _pron.get("pause_strong",           500),
        _pron.get("pause_comma",            300),
        _pron.get("pause_triple_excl",      800),
        _pron.get("pause_double_semi_min",  700),
        _pron.get("pause_double_semi_max",  1500),
        _pron.get("pause_double_colon_min", 1000),
        _pron.get("pause_double_colon_max", 3000),
    )


def fn_tts_direct(text: str, voice: str, use_clone: bool, clone_name: str) -> str | None:
    """
    Synthèse TTS directe — parse les marqueurs [émotion] [voix] [voix:Profil]
    et séquence les segments exactement comme la Lecture autonome.
    Renvoie un chemin WAV prêt à jouer.
    """
    import gc
    text = apply_pronunciation(text.strip())
    if not text:
        return None

    default_clone = clone_name if (use_clone and clone_name) else None
    # Découper selon les marqueurs (émotion + voix + voix:Profil)
    segments = _parse_segments(text, voice, default_clone)   # → [(emotion, voice, clone, chunk), …]

    out_path = Path(__file__).parent / "tts_direct.wav"
    out_path.unlink(missing_ok=True)       # sinon, en cas d'échec, l'ancien son serait renvoyé
    wav_out  = None
    try:
        for seg_emotion, seg_voice, seg_clone, chunk in segments:
            # Émotion sur voix clonée → sample de référence spécifique à l'émotion
            # Émotion sur voix standard → instruct (CustomVoice)
            # Les marqueurs :: et ;; insèrent un silence (durée aléatoire).
            for arr in _iter_chunk_audio(chunk, seg_emotion, seg_voice,
                                         seg_clone, TTS_SR):
                if wav_out is None:
                    channels = 1 if arr.ndim == 1 else arr.shape[1]
                    wav_out  = sf.SoundFile(
                        str(out_path), mode="w",
                        samplerate=TTS_SR, channels=channels,
                        subtype="PCM_16")
                if wav_out.channels > 1 and arr.ndim == 1:
                    arr = np.column_stack([arr] * wav_out.channels)
                wav_out.write(arr)
                del arr
            gc.collect()
    finally:
        if wav_out is not None:
            wav_out.close()

    if not (out_path.exists() and out_path.stat().st_size > 0):
        return None
    # une copie avec un nom propre à ce son (voix, début du texte, date et heure)
    final = _nom_enregistrement(clone_name if (use_clone and clone_name) else voice, text)
    shutil.copy2(out_path, final)
    return str(final)


def fn_send_text(user_text, voice, model, temp, length, strict, custom_sys):
    if not user_text.strip():
        yield _get_history(), None, ""
        return
    fn_sync_config(voice, model, temp, length, strict, custom_sys)
    _push("user", user_text)
    yield _get_history(), None, ""
    cfg  = dict(_cfg)
    hist = _get_history()
    try:
        answer = call_ollama(user_text, hist, cfg)
    except Exception as exc:
        answer = f"⚠️ LLM : {exc}"
    _push("assistant", answer)
    audio = synthesize(answer, voice)
    yield _get_history(), audio, ""


def fn_load_file(file_obj, action, voice, model, temp, length, strict, custom_sys):
    text, status = read_file(file_obj)
    if not text:
        yield _cfg.get("context", ""), status, _get_history(), None
        return
    _cfg["context"] = text
    filename = Path(str(file_obj)).name if file_obj else "fichier"

    if action == "📌 Utiliser comme contexte":
        yield text, status + " — contexte actif ✓", _get_history(), None

    elif action == "📝 Résumé automatique":
        fn_sync_config(voice, model, temp, length, strict, custom_sys)
        _push("user", f"📄 Résumé de « {filename} »")
        yield text, status, _get_history(), None
        prompt = (f"Voici le contenu du fichier « {filename} ».\n\n"
                  f"{text[:5000]}\n\nFais-en un résumé clair et structuré.")
        cfg = dict(_cfg)
        cfg["length_key"] = "📖 Développé — 2 à 3 paragraphes"
        try:
            answer = call_ollama(prompt, _get_history(), cfg)
        except Exception as exc:
            answer = f"⚠️ {exc}"
        _push("assistant", answer)
        audio = synthesize(answer, voice)
        yield text, status + " — contexte actif ✓", _get_history(), audio

    elif action == "🔊 Lire à voix haute":
        chunks = _chunk_text(text)
        note = f"🔊 Lecture de « {filename} » ({len(chunks)} passages)…"
        _push("assistant", note)
        yield text, status + " — lecture…", _get_history(), None
        preview = " ".join(chunks[:3])
        audio = synthesize(preview, voice, "reading_preview.wav")
        yield text, status + " — lecture lancée", _get_history(), audio

    else:
        yield text, status, _get_history(), None


def fn_clear():
    with _hist_lock:
        _history.clear()
    return [], None


def fn_export(history):
    path = "conversation_export.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return gr.update(value=path, visible=True)


def fn_refresh_models():
    models = get_ollama_models()
    return gr.update(choices=models, value=models[0] if models else "mistral-nemo")


def fn_preview_voice(voice):
    sample = f"Bonjour, je suis la voix {voice}. Je peux lire vos textes à voix haute."
    return synthesize(sample, voice, "voice_preview.wav")


# ══════════════════════════════════════════════════════════════════════════
#  API — UTILISER À DISTANCE
# ══════════════════════════════════════════════════════════════════════════
#
#   /parler : (texte, voix, émotion, langue) → un fichier .wav
#   /voix   : la liste des voix et des émotions
# Utilisables avec gradio_client (Python, JavaScript) ou en HTTP (curl) : voir l'onglet
# « À distance ». Les marqueurs [triste] [voix:Profil] :: ;; et le dictionnaire de
# prononciation s'appliquent comme dans l'interface.

_api_lock = threading.Lock()
_RESEAU = os.getenv("BOTQWEN_RESEAU", "").strip().lower() in ("1", "oui", "yes", "true")
def _premier_port_libre(depart: int) -> int:
    """7860, ou le suivant s'il est déjà pris (par exemple si botQwen2 tourne aussi)."""
    for port in range(depart, depart + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0" if _RESEAU else "127.0.0.1", port))
                return port
            except OSError:
                continue
    return depart


_PORT = _premier_port_libre(int(os.getenv("BOTQWEN_PORT", "7860") or 7860))
_MOT_DE_PASSE = os.getenv("BOTQWEN_MOT_DE_PASSE", "").strip()


def api_parler(texte: str, voix: str, emotion: str, langue: str):
    """Fait dire un texte par une voix (voix clonée de la bibliothèque, ou voix standard comme Ryan).
    Renvoie le fichier audio .wav."""
    global _active_emotion, _tts_language
    texte = (texte or "").strip()
    if not texte:
        raise gr.Error("Le texte est vide.")
    voix = (voix or "").strip()
    clone = next((n for n in _clone_profiles if n.lower() == voix.lower()), None)
    standard = next((v for v in VOICES if v.lower() == voix.lower()), None)
    if voix and not clone and not standard:
        raise gr.Error(f"Voix inconnue : « {voix} ». La liste des voix : /voix.json")
    if not voix:
        clone = _clone_name or None
        standard = None if clone else VOICES[0]
    emotion = _EMOTION_MAP.get((emotion or "neutral").strip().lower(), "neutral")
    langue = next((l for l in LANGUAGES if l.lower() == (langue or "French").strip().lower()), "French")
    with _api_lock:
        avant = (_active_emotion, _tts_language)
        _active_emotion, _tts_language = emotion, langue
        try:
            chemin = fn_tts_direct(texte, standard or VOICES[0], bool(clone), clone or "")
        finally:
            _active_emotion, _tts_language = avant
        if not chemin:
            raise gr.Error("La synthèse a échoué (voir la fenêtre du terminal sur l'ordinateur).")
    return chemin               # déjà un nom propre à ce son


def api_voix():
    """La liste des voix utilisables par /parler."""
    return {
        "voix_clonees": {n: {"versions": ["neutral"] + list((p.get("emotions") or {}).keys()),
                             "reglage": _resume_reglages(n)}
                         for n, p in sorted(_clone_profiles.items(), key=lambda x: _ordre_alphabetique(x[0]))},
        "voix_standard": VOICES,
        "emotions": EMOTIONS,
        "langues": LANGUAGES,
    }


def _adresses_reseau() -> list[str]:
    """Les adresses IP de cette machine (Mac ou PC) sur le réseau local, recalculées à chaque appel :
    si on change de Wi-Fi ou si la box donne une autre adresse, la nouvelle est trouvée.
    (Aucun paquet n'est envoyé : on demande seulement au système par où il passerait.)"""
    ips: list[str] = []
    for cible in ("10.255.255.255", "192.168.255.255", "172.31.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((cible, 1))
                ip = s.getsockname()[0]
            if ip and not ip.startswith(("127.", "0.")) and ip not in ips:
                ips.append(ip)
        except OSError:
            continue
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def _adresse_reseau() -> str:
    ips = _adresses_reseau()
    return ips[0] if ips else "adresse-introuvable"


def _nom_bonjour() -> str | None:
    """Le nom de la machine sur le réseau local (« Mac-de-Dario.local »). Il ne change pas quand
    l'adresse IP change : un iPhone le trouve tout seul (Bonjour) ; un PC Windows 10/11 aussi."""
    import platform
    import subprocess
    nom = ""
    if platform.system() == "Darwin":
        try:
            nom = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
        except Exception:
            nom = ""
    if not nom:
        nom = socket.gethostname().split(".")[0]
    nom = re.sub(r"[^A-Za-z0-9-]", "-", nom).strip("-")
    return f"{nom}.local" if nom else None


def _hote_du_navigateur(request) -> str | None:
    """L'adresse que le navigateur a utilisée pour ouvrir cette page (si ce n'est pas 127.0.0.1) :
    c'est forcément une adresse qui marche depuis le réseau."""
    try:
        hote = (request.headers.get("host") or "").split(":")[0]
    except Exception:
        return None
    if hote and hote not in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"):
        return hote
    return None


def _texte_api(request=None) -> str:
    local = f"http://127.0.0.1:{_PORT}/"
    ips = _adresses_reseau()
    bonjour = _nom_bonjour()
    hote = _hote_du_navigateur(request) if request is not None else None
    principale = hote or (ips[0] if ips else None)
    reseau = f"http://{principale}:{_PORT}/" if principale else None
    adresse = reseau if (_RESEAU and reseau) else local
    auth = ', auth=("voix", "le-mot-de-passe")' if _MOT_DE_PASSE else ""
    mdp = "&mdp=le-mot-de-passe" if _MOT_DE_PASSE else ""
    exemple_voix = next(iter(sorted(_clone_profiles, key=_ordre_alphabetique)), "Ryan")
    from urllib.parse import quote
    lien = f"{adresse}dire?voix={quote(exemple_voix)}&texte={quote('Bonjour, je suis prête.')}{mdp}"
    autres = [f"`http://{ip}:{_PORT}/`" for ip in ips if ip != principale]
    if bonjour:
        autres.append(f"`http://{bonjour}:{_PORT}/` (ce nom ne change pas, même si l'adresse IP change)")
    if _RESEAU:
        etat = (f"L'API est **ouverte au réseau local**. Depuis un iPhone, un autre ordinateur, un Raspberry Pi… "
                f"sur le même réseau, l'adresse est **{reseau or '(aucun réseau trouvé)'}**")
        etat += (" — mot de passe demandé (identifiant `voix`)." if _MOT_DE_PASSE else
                 ". Aucun mot de passe : n'importe qui sur ce réseau peut s'en servir.")
        if autres:
            etat += "\n\nAutres adresses de cette machine : " + " · ".join(autres)
    else:
        etat = (f"L'API répond **sur cette machine seulement**, à l'adresse **{local}**. Pour l'utiliser depuis "
                "un iPhone ou une autre machine, lancez plutôt `botqwenClaude_reseau.command` (Mac) ou "
                "`LANCER_BOTQWENCLAUDE_RESEAU.bat` (PC)"
                + (f" : l'adresse sera alors **{reseau}**" if reseau else "")
                + (f", ou `http://{bonjour}:{_PORT}/`." if bonjour else "."))
    etat += ("\n\nL'adresse est recalculée chaque fois que vous ouvrez cet onglet (changement de Wi-Fi, "
             "nouvelle adresse donnée par la box…).")
    return f"""{etat}

### Le plus simple : un lien

Ouvrez ce lien dans un navigateur (Safari sur iPhone, par exemple) : la voix dit la phrase.

[{lien}]({lien})

`texte`, `voix`, `emotion` (`neutral`, `happy`, `sad`, `excited`, `curious`, `angry`) et `langue` (`French`…) se règlent dans le lien. La liste des voix : [{adresse}voix.json]({adresse}voix.json). Le mode d'emploi pour l'iPhone (Safari, Raccourcis, Siri) est dans le fichier `API_iPhone.md`, à côté de ce programme.

### Ce que l'API sait faire

| Adresse | Ce qu'on lui donne | Ce qu'elle rend |
|---|---|---|
| `/dire` | dans le lien (GET) ou en JSON (POST) : `texte`, `voix`, `emotion`, `langue`{", `mdp`" if _MOT_DE_PASSE else ""} | le son `.wav`, directement |
| `/voix.json` | rien | la liste des voix, de leurs versions et des émotions |
| `/parler` (API Gradio) | le texte, la voix, l'émotion, la langue | un fichier `.wav` |

Le texte peut contenir les mêmes marqueurs que dans l'interface : `[triste]`, `[voix:{exemple_voix}]`, `::` et `;;` pour les pauses. Le dictionnaire de prononciation s'applique.

### En Python

```python
import requests
r = requests.post("{adresse}dire", json={{
    "texte": "Bonjour, je suis prête.",
    "voix": "{exemple_voix}",
    "emotion": "neutral",{chr(10) + '    "mdp": "le-mot-de-passe",' if _MOT_DE_PASSE else ""}
}})
open("reponse.wav", "wb").write(r.content)
```

Ou avec `gradio_client` (`pip install gradio_client`) :

```python
from gradio_client import Client
client = Client("{adresse}"{auth})
fichier = client.predict("Bonjour, je suis prête.", "{exemple_voix}", "neutral", "French", api_name="/parler")
```

### En ligne de commande (curl)

```bash
curl -o reponse.wav "{adresse}dire?voix={quote(exemple_voix)}&texte=Bonjour{mdp}"
```

La documentation complète de l'API Gradio s'ouvre par le lien vers l'API, tout en bas de cette page."""


# ── Journal des connexions : qui, depuis le réseau, arrive jusqu'à Voix ? ─────────────────────────
# Pour comprendre un appareil qui « ne se connecte pas » : si son nom n'apparaît jamais dans le terminal,
# c'est qu'il n'atteint pas l'ordinateur (autre Wi-Fi, VPN…) ; s'il apparaît, la connexion est bonne.
_appareils_vus: set = set()


def _appareil(user_agent: str) -> str:
    ua = user_agent or ""
    if "iPhone" in ua:
        appareil = "iPhone"
    elif "iPad" in ua or ("Macintosh" in ua and "Mobile" in ua):
        appareil = "iPad"
    elif "Android" in ua:
        appareil = "Android"
    elif "Windows" in ua:
        appareil = "PC Windows"
    elif "Macintosh" in ua:
        appareil = "Mac"
    elif "curl" in ua.lower() or "python" in ua.lower():
        appareil = "programme"
    else:
        appareil = "appareil"
    for nav in ("CriOS", "FxiOS", "EdgiOS", "Chrome", "Firefox", "Safari", "Shortcuts", "Raccourcis"):
        if nav in ua:
            return f"{appareil}, {dict(CriOS='Chrome', FxiOS='Firefox', EdgiOS='Edge').get(nav, nav)}"
    return appareil


class _JournalConnexions:
    """Écrit dans le terminal la première visite de chaque appareil du réseau, et chaque /dire."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            ip = (scope.get("client") or ("?", 0))[0]
            if ip not in ("127.0.0.1", "::1", "localhost"):
                entetes = dict(scope.get("headers") or [])
                qui = _appareil(entetes.get(b"user-agent", b"").decode(errors="ignore"))
                chemin = scope.get("path", "")
                if (ip, qui) not in _appareils_vus:
                    _appareils_vus.add((ip, qui))
                    print(f"📱 Connexion depuis {ip} ({qui})", flush=True)
                if chemin in ("/dire", "/ping"):
                    print(f"   {chemin} demandé par {ip} ({qui})", flush=True)
        await self.app(scope, receive, send)


# ── Adresses simples : /dire et /voix.json (pour un iPhone, un navigateur, curl…) ──────────────
def _installer_adresses_simples(app) -> None:
    """Ajoute à l'application deux adresses sans détour, qui rendent directement le son :
         GET  /dire?texte=…&voix=…&emotion=…&langue=…     (un lien suffit : Safari le lit)
         POST /dire   {"texte": …, "voix": …}             (pour les textes longs)
         GET  /voix.json"""
    from fastapi import Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.routing import APIRoute

    def faire(params: dict):
        if _MOT_DE_PASSE and str(params.get("mdp") or "") != _MOT_DE_PASSE:
            return JSONResponse({"erreur": "mot de passe absent ou faux (paramètre mdp)"}, status_code=401)
        try:
            chemin = api_parler(str(params.get("texte") or ""), str(params.get("voix") or ""),
                                str(params.get("emotion") or "neutral"), str(params.get("langue") or "French"))
        except gr.Error as exc:
            return JSONResponse({"erreur": getattr(exc, "message", str(exc))}, status_code=400)
        except Exception as exc:
            return JSONResponse({"erreur": str(exc)}, status_code=500)
        return FileResponse(chemin, media_type="audio/wav",
                            headers={"Content-Disposition": f'inline; filename="{Path(chemin).name}"'})

    def dire_get(texte: str = "", voix: str = "", emotion: str = "neutral", langue: str = "French",
                 mdp: str = ""):
        return faire({"texte": texte, "voix": voix, "emotion": emotion, "langue": langue, "mdp": mdp})

    async def dire_post(request: Request):
        try:
            donnees = await request.json()
        except Exception:
            donnees = dict(await request.form())
        import anyio
        return await anyio.to_thread.run_sync(faire, donnees if isinstance(donnees, dict) else {})

    def ping():
        from fastapi.responses import HTMLResponse
        return HTMLResponse("""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Voix répond</title></head>
<body style="font-family:-apple-system,system-ui,sans-serif;background:#F1F3F2;color:#1F2429;padding:32px;line-height:1.5">
<h1 style="font-size:40px;margin:0 0 12px">Voix répond ✓</h1>
<p>Ce téléphone atteint bien l'ordinateur. Si l'interface ne s'ouvre pas, essayez le lien qui parle :</p>
<p><a href="/dire?texte=Bonjour%2C%20je%20t%27entends." style="font-size:20px;color:#2F5E8F">▶ Faire dire « Bonjour, je t'entends. »</a></p>
<p><a href="/" style="color:#2F5E8F">Ouvrir l'interface complète</a></p>
</body></html>""")

    def voix_json(mdp: str = ""):
        if _MOT_DE_PASSE and mdp != _MOT_DE_PASSE:
            return JSONResponse({"erreur": "mot de passe absent ou faux (paramètre mdp)"}, status_code=401)
        return JSONResponse(api_voix())

    for route in (APIRoute("/dire", dire_get, methods=["GET"]),
                  APIRoute("/dire", dire_post, methods=["POST"]),
                  APIRoute("/voix.json", voix_json, methods=["GET"]),
                  APIRoute("/ping", ping, methods=["GET"])):
        app.router.routes.insert(0, route)       # avant les adresses de Gradio


# ══════════════════════════════════════════════════════════════════════════
#  INTERFACE GRADIO
# ══════════════════════════════════════════════════════════════════════════

_ollama_models = get_ollama_models()

# ── Apparence : un studio clair et épuré ──────────────────────────────────
# Murs clairs, surfaces blanches, texte graphite ; le rouge « antenne » ne sert qu'au micro qui
# écoute et aux boutons d'arrêt ; le dégradé bleu → ambre ne sert qu'à l'échelle du caractère d'une voix.
_MUR      = "#F1F3F2"
_BLANC    = "#FFFFFF"
_GRAPHITE = "#1F2429"
_ACIER    = "#5F6872"
_TRAIT    = "#D8DCDF"
_ANTENNE  = "#D63B3B"
_POSEE    = "#3C6EA6"
_VIVE     = "#D7852E"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill=_MUR,
    body_text_color=_GRAPHITE,
    body_text_color_subdued=_ACIER,
    background_fill_primary=_BLANC,
    background_fill_secondary=_MUR,
    border_color_primary=_TRAIT,
    border_color_accent=_GRAPHITE,
    block_background_fill=_BLANC,
    block_border_color=_TRAIT,
    block_border_width="1px",
    block_shadow="none",
    block_radius="12px",
    block_title_text_color=_GRAPHITE,
    block_title_text_weight="600",
    block_label_background_fill=_BLANC,
    block_label_text_color=_ACIER,
    block_label_text_weight="500",
    block_info_text_color=_ACIER,
    panel_background_fill=_BLANC,
    panel_border_color=_TRAIT,
    input_background_fill="#F5F6F6",
    input_border_color="#E4E7E9",
    input_border_color_focus=_GRAPHITE,
    input_shadow="none",
    input_shadow_focus=f"0 0 0 2px {_GRAPHITE}22",
    input_placeholder_color="#9AA2AA",
    checkbox_background_color=_BLANC,
    checkbox_border_color="#B9BFC4",
    checkbox_border_color_focus=_GRAPHITE,
    checkbox_border_color_selected=_GRAPHITE,
    checkbox_background_color_selected=_GRAPHITE,
    checkbox_label_background_fill=_BLANC,
    checkbox_label_background_fill_hover=_MUR,
    checkbox_label_background_fill_selected=_GRAPHITE,
    checkbox_label_border_color=_TRAIT,
    checkbox_label_text_color=_GRAPHITE,
    checkbox_label_text_color_selected=_BLANC,
    checkbox_label_shadow="none",
    accordion_text_color=_GRAPHITE,
    table_text_color=_GRAPHITE,
    table_even_background_fill=_BLANC,
    table_odd_background_fill="#F7F8F8",
    table_border_color=_TRAIT,
    button_primary_background_fill=_GRAPHITE,
    button_primary_background_fill_hover="#384049",
    button_primary_text_color=_BLANC,
    button_primary_text_color_hover=_BLANC,
    button_primary_border_color=_GRAPHITE,
    button_primary_shadow="none",
    button_secondary_background_fill=_BLANC,
    button_secondary_background_fill_hover=_MUR,
    button_secondary_text_color=_GRAPHITE,
    button_secondary_text_color_hover=_GRAPHITE,
    button_secondary_border_color="#C3C9CE",
    button_secondary_shadow="none",
    button_cancel_background_fill=_BLANC,
    button_cancel_background_fill_hover="#FBEDED",
    button_cancel_text_color=_ANTENNE,
    button_cancel_text_color_hover=_ANTENNE,
    button_cancel_border_color="#EDC4C4",
    button_border_width="1px",
    button_small_padding="6px 14px",
    button_large_radius="10px",
    button_small_radius="8px",
    slider_color=_GRAPHITE,
    error_background_fill="#FBEDED",
    error_border_color=_ANTENNE,
    error_text_color="#A32424",
    code_background_fill="#F4F5F6",
    link_text_color="#2F5E8F",
    link_text_color_hover=_GRAPHITE,
    link_text_color_visited="#2F5E8F",
    shadow_drop="none",
    shadow_drop_lg="none",
)

CSS = f"""
.gradio-container {{ max-width: 1440px !important; margin: 0 auto !important; padding: 0 28px 40px !important; }}
footer {{ opacity: .7; }}

/* En-tête : le mot « Voix » fait office de logo */
#entete {{ display: flex; align-items: stretch; justify-content: space-between; gap: 24px;
          padding: 28px 2px 18px; border-bottom: 1px solid {_TRAIT}; margin-bottom: 18px; flex-wrap: nowrap; }}
#entete .block, #entete .html-container, #entete .prose {{ padding: 0 !important; border: none !important;
          background: transparent !important; box-shadow: none !important; }}
#entete-droite {{ display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between;
                 gap: 14px; flex-grow: 0 !important; }}
#arreter {{ background: {_ANTENNE} !important; border-color: {_ANTENNE} !important; color: {_BLANC} !important;
           font-weight: 650; padding: 7px 18px !important; width: auto !important; min-width: 0 !important;
           align-self: flex-end; }}
#entete-onde {{ align-self: flex-end; width: auto !important; }}
#entete-onde .onde {{ justify-content: flex-end; }}
#arreter:hover {{ background: #B92F2F !important; border-color: #B92F2F !important; }}
#arreter:focus-visible {{ outline: 2px solid {_GRAPHITE}; outline-offset: 2px; }}
#entete h1 {{ font-size: 56px; line-height: .9; font-weight: 700; letter-spacing: -0.045em; margin: 0;
             color: {_GRAPHITE}; }}
#entete p {{ margin: 6px 0 0; color: {_ACIER}; font-size: 15px; max-width: 60ch; }}
#entete .onde {{ display: flex; align-items: center; gap: 3px; height: 40px; margin-bottom: 4px; }}
#entete .onde i {{ display: block; width: 4px; border-radius: 2px; background: {_GRAPHITE}; opacity: .85; }}

/* Colonne de gauche : une console, des sections séparées par des traits (pas de cartes) */
#console {{ background: {_BLANC}; border: 1px solid {_TRAIT}; border-radius: 14px; padding: 18px 16px 14px;
           gap: 6px !important; }}
#console > .block:not(button), #console > div > .block:not(button), #console .form {{ border: none !important;
           box-shadow: none !important; }}
.section-titre {{ margin: 14px 0 2px !important; padding-top: 14px; border-top: 1px solid {_TRAIT}; }}
.section-titre h3, .section-titre p {{ font-size: 15px !important; font-weight: 650 !important; margin: 0 !important;
                                       color: {_GRAPHITE}; }}
.section-titre.premiere {{ border-top: none; padding-top: 0; margin-top: 2px !important; }}

/* Le micro : bouton principal ; rouge « antenne » quand il écoute */
#micro button, button#micro {{ height: 56px; font-size: 16px; font-weight: 650; letter-spacing: -0.01em; }}
#micro.stop, button#micro.stop {{ background: {_ANTENNE} !important; border-color: {_ANTENNE} !important;
                                  color: {_BLANC} !important; }}

/* Onglets : du texte souligné, pas des boutons */
.onglets > .tab-wrapper, .onglets .tab-container {{ border-bottom: 1px solid {_TRAIT}; }}
.onglets button[role="tab"] {{ font-size: 15px; font-weight: 550; color: {_ACIER}; padding: 10px 14px;
                               border: none !important; background: transparent !important; }}
.onglets button[role="tab"][aria-selected="true"] {{ color: {_GRAPHITE}; box-shadow: inset 0 -2px 0 {_GRAPHITE}; }}

/* Textes d'aide : courts, gris, lisibles */
.aide, .aide p {{ color: {_ACIER}; font-size: 14.5px; line-height: 1.5; max-width: 78ch; }}
.aide code {{ font-size: 13px; }}

/* L'échelle du caractère : un curseur en 5 crans, du posé (bleu) au vivant (ambre) */
#echelle .wrap {{ display: grid !important; grid-template-columns: repeat(5, 1fr); gap: 0 !important;
                 border-radius: 10px; overflow: hidden; border: 1px solid {_TRAIT};
                 background: linear-gradient(90deg, {_POSEE}14, {_VIVE}14); }}
#echelle label {{ justify-content: center; text-align: center; border: none !important; border-radius: 0 !important;
                 background: transparent !important; padding: 14px 6px !important; margin: 0 !important;
                 font-weight: 550; color: {_GRAPHITE}; box-shadow: none !important; cursor: pointer; }}
#echelle label + label {{ border-left: 1px solid {_TRAIT} !important; }}
#echelle label input {{ position: absolute; opacity: 0; pointer-events: none; }}
#echelle label:nth-child(1).selected {{ background: {_POSEE} !important; color: #fff !important; }}
#echelle label:nth-child(2).selected {{ background: #6F7B8C !important; color: #fff !important; }}
#echelle label:nth-child(3).selected {{ background: #8E8476 !important; color: #fff !important; }}
#echelle label:nth-child(4).selected {{ background: #C07A3A !important; color: #fff !important; }}
#echelle label:nth-child(5).selected {{ background: {_VIVE} !important; color: #fff !important; }}
#echelle-bornes {{ display: flex; justify-content: space-between; color: {_ACIER}; font-size: 13px; margin-top: -4px; }}
#echelle-bornes span:first-child {{ color: {_POSEE}; }}
#echelle-bornes span:last-child {{ color: {_VIVE}; }}

/* La manière de copier : deux choix côte à côte */
#maniere .wrap {{ display: grid !important; grid-template-columns: 1fr 1fr; gap: 8px !important; }}
#maniere label {{ justify-content: center; padding: 12px !important; font-weight: 600; }}

/* Le réglage actuel d'une voix */
#reglage-actuel {{ background: #F6F7F7; border-left: 3px solid {_GRAPHITE}; padding: 10px 14px; border-radius: 0 8px 8px 0; }}
#reglage-actuel p {{ margin: 0; }}
#progression p {{ margin: 0; color: {_ACIER}; }}

/* Code de l'onglet API */
#api pre {{ font-size: 13px; border: 1px solid {_TRAIT}; border-radius: 10px; }}
#api table {{ font-size: 14px; border-collapse: collapse; }}
#api th, #api td {{ border: 1px solid {_TRAIT} !important; padding: 8px 10px; vertical-align: top; }}
#api th {{ background: #F6F7F7; font-weight: 600; }}
.aide table {{ border-collapse: collapse; font-size: 13.5px; width: 100%; border: none !important; }}
.aide th, .aide td {{ border: none !important; border-bottom: 1px solid {_TRAIT} !important; padding: 7px 10px 7px 0; }}

@media (max-width: 900px) {{
  .gradio-container {{ padding: 0 14px 30px !important; }}
  #entete h1 {{ font-size: 42px; }}
  #echelle label {{ padding: 12px 2px !important; font-size: 13px; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""

# La petite onde de l'en-tête : des barres de hauteurs fixes (une « empreinte » de voix, sans animation)
_ONDE = "".join(f'<i style="height:{h}px"></i>' for h in
                (8, 14, 22, 30, 18, 36, 26, 12, 20, 34, 40, 24, 16, 28, 10, 18, 8))

# Charge la bibliothèque de profils AVANT de construire l'UI
# (pas besoin du TTS pour ça — lecture JSON uniquement)
_rebuild_clone_prompt()

_CHOIX_EMOTIONS = [("Neutre", "neutral"), ("Joyeuse", "happy"), ("Triste", "sad"),
                   ("Enthousiaste", "excited"), ("Curieuse", "curious"), ("En colère", "angry")]
_CHOIX_MODES = [(R.nom_mode(m), m) for m in R.MODES]
_CHOIX_CRANS = [(nom, t) for t, nom, _ in R.CRANS]


def _valeurs_initiales():
    """Le panneau de réglage tel qu'il s'affiche au démarrage (voix active)."""
    p = _panneau(_clone_name, "")
    v = p[0] if isinstance(p[0], dict) else {}
    m = p[3].get("value", "imitation") if isinstance(p[3], dict) else "imitation"
    c = p[4].get("value", TEMPERATURE_CLONE_DEFAUT) if isinstance(p[4], dict) else TEMPERATURE_CLONE_DEFAUT
    return (v.get("choices") or [("Voix principale", PRINCIPALE)], v.get("value", PRINCIPALE), p[1], p[2],
            m or "imitation", R.cran_le_plus_proche(c), p[5], p[6], p[7])


(_p_versions, _p_version, _p_actuel, _p_bilans, _p_mode, _p_cran, _p_mode_md, _p_cran_md,
 _p_audio) = _valeurs_initiales()

with gr.Blocks(title="Voix — assistant vocal local") as demo:

    ctx_state = gr.State("")

    with gr.Row(elem_id="entete", equal_height=False):
        gr.HTML(f"""<div><h1>Voix</h1>
  <p>Assistant vocal local : {_runtime.stt_name} pour écouter, Ollama pour répondre, Qwen3-TTS pour parler.
  Rien ne sort de cette machine.</p></div>""", elem_id="entete-titre")
        with gr.Column(scale=0, min_width=170, elem_id="entete-droite"):
            shutdown_btn = gr.Button("Arrêter", elem_id="arreter", size="sm")
            gr.HTML(f'<div class="onde" aria-hidden="true">{_ONDE}</div>', elem_id="entete-onde")

    with gr.Row(equal_height=False):

        # ════════════════════════════════════════════════════════════════
        #  CONSOLE (colonne de gauche)
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=310, elem_id="console"):

            vad_btn = gr.Button("Démarrer l'écoute du micro", variant="primary", size="lg",
                                elem_id="micro")
            vad_status_lbl = gr.Textbox(
                value=_vad_status, label="Micro",
                interactive=False, max_lines=1)

            gr.Markdown("### Voix standard", elem_classes=["section-titre"])
            voice_dd = gr.Dropdown(
                choices=VOICES, value="Ryan",
                label="Locuteur")
            language_dd = gr.Dropdown(
                choices=LANGUAGES, value="French",
                label="Langue parlée",
                info="French pour des textes en français.")
            emotion_dd = gr.Dropdown(
                choices=_CHOIX_EMOTIONS, value="neutral",
                label="Ton",
                info="Aussi possible dans le texte : [joyeux], [triste]…")
            with gr.Row():
                prev_btn    = gr.Button("Écouter", size="sm", scale=1, min_width=90)
                preview_out = gr.Audio(autoplay=True, show_label=False, scale=3)

            gr.Markdown("### Modèle de langue", elem_classes=["section-titre"])
            with gr.Row():
                model_dd    = gr.Dropdown(
                    choices=_ollama_models,
                    value=_ollama_models[0] if _ollama_models else "mistral-nemo",
                    label="Modèle", show_label=False, scale=4)
                refresh_btn = gr.Button("Actualiser", size="sm", scale=1, min_width=90)
            length_dd = gr.Dropdown(
                choices=list(LONGUEUR.keys()),
                value="💬 Normal — 1 paragraphe",
                label="Longueur des réponses")

            gr.Markdown("### Fichier de contexte", elem_classes=["section-titre"])
            file_in = gr.File(
                label="Un .txt ou un .docx",
                file_types=[".txt", ".docx"],
                file_count="single",
                type="filepath", height=110)
            file_action = gr.Radio(
                choices=[
                    "📌 Utiliser comme contexte",
                    "📝 Résumé automatique",
                    "🔊 Lire à voix haute",
                ],
                value="📌 Utiliser comme contexte",
                show_label=False)
            load_btn   = gr.Button("Charger le fichier", variant="primary", size="sm")
            ctx_status = gr.Textbox(
                value="Aucun contexte", label="Contexte actif",
                interactive=False, max_lines=2)
            strict_chk = gr.Checkbox(
                label="Répondre seulement d'après ce fichier", value=False)

            with gr.Accordion("Réglages avancés du modèle de langue", open=False):
                temp_sl = gr.Slider(0.0, 2.0, 0.7, step=0.05,
                                    label="Liberté des réponses",
                                    info="0 : toujours la même réponse, prudente · 0,7 : naturel · "
                                         "au-delà de 1 : inventif, parfois incohérent.")
                sys_box = gr.Textbox(
                    label="Consigne personnalisée (system prompt)",
                    placeholder="Laissez vide pour la consigne automatique…",
                    lines=4)

            models_lbl = gr.Textbox(
                value=_models_status, label="Modèles",
                interactive=False, max_lines=2, elem_classes=["section-titre"])

        # ════════════════════════════════════════════════════════════════
        #  ONGLETS
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=3):
            with gr.Tabs(elem_classes=["onglets"]):

                # ── Lire un texte (TTS direct) ────────────────────────
                with gr.Tab("Lire"):
                    _tts_caps = {
                        "base":   "Le modèle chargé au démarrage est le modèle de clonage : les voix "
                                  "standard (Ryan, Vivian…) et leurs tons demandent le modèle CustomVoice, "
                                  "chargé automatiquement quand il le faut.",
                        "custom": "Voix standard et tons disponibles ; les voix clonées chargent "
                                  "automatiquement le modèle de clonage.",
                        "design": "Modèle VoiceDesign : décrivez la voix voulue.",
                    }.get(_TTS_MODEL_TYPE, "")
                    gr.Markdown(
                        "Tapez un texte, choisissez une voix, écoutez. Le texte peut changer de ton "
                        "`[triste]` ou de voix `[voix:Marielle]` en cours de route ; `::` et `;;` "
                        "ajoutent un silence. Les voix clonées et les voix standard ne se mélangent "
                        f"pas dans un même texte. {_tts_caps}",
                        elem_classes=["aide"])
                    tts_text_in = gr.Textbox(
                        label="Texte",
                        placeholder=(
                            "Bonjour ! [triste] Je ne pensais pas vous revoir…\n"
                            "[voix:Marielle] Et pourtant, me voilà."),
                        lines=7)
                    with gr.Row():
                        tts_voice_dd = gr.Dropdown(
                            label="Voix standard",
                            choices=VOICES, value=VOICES[0], scale=2)
                        tts_use_clone_chk = gr.Checkbox(
                            label="Voix clonée", value=False, scale=1)
                        tts_clone_dd = gr.Dropdown(
                            label="Voix clonée",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            scale=2, visible=False)
                    tts_synth_btn = gr.Button(
                        "Faire lire le texte", variant="primary")
                    tts_audio_out = gr.Audio(
                        label="Résultat", autoplay=True)

                # ── Discussion ────────────────────────────────────────
                with gr.Tab("Discussion"):
                    cfg_display = gr.Markdown(value=_cfg_badge(), elem_classes=["aide"])
                    chatbot = gr.Chatbot(
                        label="Conversation", height=460,
                        layout="bubble", render_markdown=True,
                        avatar_images=(None, None))
                    audio_out = gr.Audio(label="Réponse parlée", autoplay=True)
                    with gr.Row():
                        text_in  = gr.Textbox(
                            placeholder="Écrivez un message (Entrée pour envoyer), ou parlez au micro…",
                            show_label=False, scale=5, lines=1, max_lines=3)
                        send_btn = gr.Button(
                            "Envoyer", variant="primary", scale=1, min_width=110)
                    with gr.Row():
                        clear_btn  = gr.Button("Effacer la conversation", variant="stop", size="sm")
                        export_btn = gr.Button("Exporter en JSON", size="sm")
                    export_dl = gr.File(label="Export", visible=False)

                # ── Lecture de fichiers ───────────────────────────────
                with gr.Tab("Fichiers"):
                    gr.Markdown(
                        "Transforme un ou plusieurs fichiers `.txt` / `.docx` en fichiers `.wav`, en entier, "
                        "sans passer par la discussion.",
                        elem_classes=["aide"])
                    with gr.Accordion("Les marqueurs à placer dans le manuscrit", open=False):
                        gr.Markdown(
                            "**Voix standard** : `[Ryan]` `[Aiden]` `[Vivian]` `[Serena]` `[Uncle_Fu]` "
                            "`[Dylan]` `[Eric]` `[Ono_Anna]` `[Sohee]`  \n"
                            "**Voix clonée** : `[voix:NomDeLaVoix]`, par exemple `[voix:Marielle]`. "
                            "On peut passer d'une voix clonée à une autre dans le même texte, mais pas "
                            "d'une voix clonée à une voix standard.  \n"
                            "**Ton** : `[neutre]` `[joyeux]` `[triste]` `[excité]` `[curieux]` `[colère]` "
                            "(ou en anglais : `[happy]` `[sad]`…). Avec une voix clonée, le ton passe par "
                            "sa version émotion, si elle existe.  \n"
                            "**Pauses** : `::` (longue) et `;;` (moyenne) ; durées réglables dans "
                            "l'onglet Dictionnaire.  \n\n"
                            "Exemple : `[voix:Marielle] [triste]` puis le texte du personnage.",
                            elem_classes=["aide"])

                    read_files_in = gr.File(
                        label="Fichiers à lire (plusieurs possibles)",
                        file_types=[".txt", ".docx"],
                        file_count="multiple",
                        type="filepath", height=130)

                    out_dir_box = gr.Textbox(
                        value=str(Path.home() / "Downloads"),
                        label="Dossier où enregistrer les .wav",
                        placeholder="/Users/toi/Documents",
                        interactive=True, max_lines=1)

                    with gr.Row():
                        read_voice_dd = gr.Dropdown(
                            label="Voix standard",
                            choices=VOICES, value=VOICES[0], scale=2)
                        read_use_clone_chk = gr.Checkbox(
                            label="Voix clonée", value=False, scale=1)
                        read_clone_dd = gr.Dropdown(
                            label="Voix clonée",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            scale=2, visible=False)
                    # clone_chk / clone_mode_lbl conservés pour les connexions héritées
                    clone_chk      = gr.Checkbox(visible=False, value=False)
                    clone_mode_lbl = gr.Textbox(visible=False, value="")

                    with gr.Row():
                        read_start_btn = gr.Button(
                            "Lire les fichiers", variant="primary", scale=2)
                        read_stop_btn  = gr.Button(
                            "Arrêter", variant="stop", scale=1)

                    read_status_lbl = gr.Textbox(
                        value=_read_status, label="État",
                        interactive=False, max_lines=1)
                    read_progress_lbl = gr.Textbox(
                        value=_read_progress, label="Fichiers enregistrés",
                        interactive=False, max_lines=4)

                # ── Voix clonées ──────────────────────────────────────
                with gr.Tab("Voix clonées"):
                    gr.Markdown(
                        "Une voix clonée se fabrique à partir de quelques secondes d'enregistrement, sur "
                        "cette machine. À sa création, elle est testée automatiquement pour lui trouver le "
                        "réglage qui la garde la plus fidèle ; vous pouvez ensuite l'ajuster à l'oreille. "
                        "Tout est enregistré dans `voices_qwen.json`, que lisent aussi le bot et le podcast.",
                        elem_classes=["aide"])

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            gr.Markdown("### Bibliothèque")
                            with gr.Row():
                                clone_dd = gr.Dropdown(
                                    label="Voix",
                                    choices=_clone_choices(),
                                    value=_clone_name or None,
                                    scale=4, interactive=True, filterable=True)
                                clone_del_btn = gr.Button(
                                    "Supprimer", variant="stop", size="sm",
                                    scale=1, min_width=100)
                            clone_status_lbl = gr.Textbox(
                                value=_clone_status, label="Statut",
                                interactive=False, max_lines=2)

                            gr.Markdown("### Réglage de la voix")
                            rg_version_dd = gr.Dropdown(
                                label="Version", choices=_p_versions, value=_p_version,
                                info="Chaque version émotion a son propre réglage.")
                            rg_actuel_md = gr.Markdown(_p_actuel, elem_id="reglage-actuel")

                            rg_mode_radio = gr.Radio(
                                choices=_CHOIX_MODES, value=_p_mode,
                                label="Manière de copier la voix", elem_id="maniere")
                            rg_mode_md = gr.Markdown(_p_mode_md, elem_classes=["aide"])

                            rg_cran_radio = gr.Radio(
                                choices=_CHOIX_CRANS, value=_p_cran,
                                label="Caractère", elem_id="echelle")
                            gr.HTML('<div id="echelle-bornes"><span>plus régulière</span>'
                                    '<span>plus expressive</span></div>')
                            rg_cran_md = gr.Markdown(_p_cran_md, elem_classes=["aide"])

                            rg_phrase_box = gr.Textbox(
                                label="Phrase pour écouter",
                                value=PHRASE_ECOUTE, lines=2)
                            with gr.Row():
                                rg_ecouter_btn = gr.Button("Écouter avec ce réglage", scale=2)
                                rg_garder_btn  = gr.Button("Garder ce réglage", variant="primary", scale=2)
                            rg_audio = gr.Audio(value=_p_audio, label="Écoute", autoplay=False, type="filepath")
                            rg_etat_md = gr.Markdown(elem_id="progression")

                        with gr.Column(scale=2):
                            gr.Markdown("### Résultats du test")
                            rg_bilans_md = gr.Markdown(_p_bilans, elem_classes=["aide"])
                            gr.Markdown(
                                "La **ressemblance** compare l'empreinte de la voix fabriquée à celle de "
                                "l'enregistrement ; les **mots compris** sont comptés par la reconnaissance "
                                "vocale, qui réécoute chaque essai. Aucune mesure n'entend l'accent ou le "
                                "grain : pour ça, fiez-vous à votre oreille.",
                                elem_classes=["aide"])
                            with gr.Row():
                                rg_test_btn = gr.Button("Refaire le test", size="sm")
                                rg_stop_btn = gr.Button("Arrêter le test", variant="stop", size="sm")
                            rg_progression_md = gr.Markdown(elem_id="progression")

                            gr.Markdown("### Ajouter une voix")
                            clone_audio_in = gr.File(
                                label="Enregistrement de la voix, ton neutre, 5 à 15 secondes",
                                file_types=[".wav", ".mp3", ".m4a", ".ogg"],
                                file_count="single",
                                type="filepath", height=110)
                            clone_ref_text_box = gr.Textbox(
                                label="Ce qui est dit, mot pour mot",
                                placeholder="La transcription exacte rend la copie plus fidèle…",
                                lines=3)
                            clone_name_box = gr.Textbox(
                                label="Nom de la voix",
                                placeholder="Par exemple : Narratrice Hanta",
                                max_lines=1)
                            clone_btn = gr.Button(
                                "Ajouter et tester la voix", variant="primary")
                            gr.Markdown("Le test démarre tout seul et dure une minute environ.",
                                        elem_classes=["aide"])

                    with gr.Accordion("Versions émotion d'une voix", open=False):
                        gr.Markdown(
                            "Avec une voix clonée, le ton ([triste], [colère]…) passe par un "
                            "**enregistrement de la même voix dans cet état** : 5 à 15 secondes, avec ce "
                            "qui est dit mot pour mot. Sans version pour un ton, la voix principale est "
                            "utilisée. Chaque version ajoutée est testée et réglée comme une voix.",
                            elem_classes=["aide"])
                        with gr.Row():
                            em_profile_dd = gr.Dropdown(
                                label="Voix",
                                choices=_clone_choices(),
                                value=_clone_name or None,
                                interactive=True, filterable=True)
                            em_emotion_dd = gr.Dropdown(
                                label="Émotion",
                                choices=[c for c in _CHOIX_EMOTIONS if c[1] != "neutral"],
                                value="sad")
                        em_coverage_lbl = gr.Textbox(
                            value=_emotion_coverage(_clone_name),
                            label="Versions existantes",
                            interactive=False, max_lines=3)
                        em_audio_in = gr.File(
                            label="Enregistrement de cette émotion",
                            file_types=[".wav", ".mp3", ".m4a", ".ogg"],
                            file_count="single", type="filepath", height=110)
                        em_text_box = gr.Textbox(
                            label="Ce qui est dit, mot pour mot",
                            lines=3)
                        with gr.Row():
                            em_add_btn = gr.Button(
                                "Ajouter et tester cette version", variant="primary", scale=2)
                            em_del_btn = gr.Button(
                                "Supprimer cette version", variant="stop", scale=1)
                        em_status_lbl = gr.Textbox(
                            value="", label="Statut",
                            interactive=False, max_lines=2)

                # ── Redire un enregistrement (Speech-to-Speech) ───────
                with gr.Tab("Redire"):
                    gr.Markdown(
                        f"Un enregistrement est transcrit ({_runtime.stt_name}), son ton général est estimé, "
                        "puis le texte est redit par la voix de votre choix. Ce n'est pas une conversion de "
                        "voix : le débit, les pauses et l'intonation exacte de l'original ne sont pas "
                        "gardés, seulement le texte et un ton approché.",
                        elem_classes=["aide"])
                    if not HAS_LIBROSA:
                        gr.Markdown(
                            "`librosa` n'est pas installé : le ton estimé sera toujours « neutre ». "
                            "Pour l'installer : `pip install librosa`",
                            elem_classes=["aide"])

                    s2s_audio_in = gr.Audio(
                        label="Enregistrement à redire",
                        sources=["upload", "microphone"], type="filepath")
                    s2s_transcribe_btn = gr.Button(
                        "Transcrire et estimer le ton",
                        variant="primary")
                    s2s_status_lbl = gr.Textbox(
                        value="", label="Statut", interactive=False, max_lines=2)

                    gr.Markdown("### Texte et ton, modifiables")
                    s2s_text_box = gr.Textbox(
                        label="Texte transcrit",
                        lines=5, placeholder="Le texte transcrit apparaîtra ici…")
                    with gr.Row():
                        s2s_emotion_dd = gr.Dropdown(
                            choices=_CHOIX_EMOTIONS, value="neutral",
                            label="Ton estimé", scale=1)
                        s2s_note_lbl = gr.Textbox(
                            value="", label="Détail de l'estimation",
                            interactive=False, max_lines=2, scale=2)

                    gr.Markdown("### Nouvelle voix")
                    with gr.Row():
                        s2s_voice_dd = gr.Dropdown(
                            label="Voix standard",
                            choices=VOICES, value=VOICES[0], scale=2)
                        s2s_use_clone_chk = gr.Checkbox(
                            label="Voix clonée", value=False, scale=1)
                        s2s_clone_dd = gr.Dropdown(
                            label="Voix clonée",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            scale=2, visible=False)
                    s2s_synth_btn = gr.Button(
                        "Redire avec cette voix", variant="primary")
                    s2s_audio_out = gr.Audio(
                        label="Résultat", autoplay=True)
                    s2s_result_lbl = gr.Textbox(
                        value="", label="Statut",
                        interactive=False, max_lines=2)

                # ── Dictionnaire ──────────────────────────────────────
                with gr.Tab("Dictionnaire"):
                    gr.Markdown(
                        "Corrige la prononciation des mots et des sigles, et règle les silences ajoutés par "
                        "`::` et `;;`. S'applique aux onglets « Lire » et « Fichiers », et à l'API.",
                        elem_classes=["aide"])

                    pron_status_lbl = gr.Textbox(
                        value=_pron_status,
                        label=f"Fichier · {PRON_FILE.name}",
                        interactive=False, max_lines=1)

                    gr.Markdown("### Prononciation")
                    gr.Markdown(
                        "Par exemple : `H.G.W.` se dit `Agé Vaï`, `M.` se dit `Monsieur`.",
                        elem_classes=["aide"])
                    with gr.Row():
                        lex_mot_box  = gr.Textbox(
                            label="Mot écrit",
                            placeholder="H.G.W.",
                            scale=2, max_lines=1)
                        lex_pron_box = gr.Textbox(
                            label="Se prononce",
                            placeholder="Agé Vaï",
                            scale=2, max_lines=1)
                        lex_add_btn  = gr.Button(
                            "Ajouter", variant="primary", scale=1, size="sm")

                    with gr.Row():
                        lex_del_box = gr.Textbox(
                            label="Retirer un mot",
                            placeholder="Le mot exact à retirer",
                            scale=3, max_lines=1)
                        lex_del_btn = gr.Button(
                            "Retirer", variant="stop", scale=1, size="sm")

                    lexicon_df = gr.Dataframe(
                        value=_pron_data_as_rows(),
                        headers=["Mot écrit", "Se prononce"],
                        datatype=["str", "str"],
                        column_count=(2, "fixed"),
                        row_count=(8, "dynamic"),
                        type="array",
                        label="Lexique",
                        interactive=False)

                    gr.Markdown("### Silences (en millisecondes)")
                    gr.Markdown(
                        "`::` et `;;` insèrent un vrai silence, d'une durée tirée au hasard entre le minimum "
                        "et le maximum (mêmes valeurs pour une durée fixe). Les marqueurs ne sont pas lus.",
                        elem_classes=["aide"])
                    with gr.Row():
                        p_dcolon_min_nb = gr.Number(
                            value=_pron.get("pause_double_colon_min", 1000),
                            label=":: minimum", precision=0, minimum=0, step=100)
                        p_dcolon_max_nb = gr.Number(
                            value=_pron.get("pause_double_colon_max", 3000),
                            label=":: maximum", precision=0, minimum=0, step=100)
                        p_dsemi_min_nb  = gr.Number(
                            value=_pron.get("pause_double_semi_min",  700),
                            label=";; minimum", precision=0, minimum=0, step=50)
                        p_dsemi_max_nb  = gr.Number(
                            value=_pron.get("pause_double_semi_max",  1500),
                            label=";; maximum", precision=0, minimum=0, step=50)

                    with gr.Accordion("Silences enregistrés mais pas encore appliqués", open=False):
                        with gr.Row():
                            p_strong_nb = gr.Number(
                                value=_pron.get("pause_strong",       500),
                                label=". ! ?", precision=0, minimum=0, step=50)
                            p_comma_nb  = gr.Number(
                                value=_pron.get("pause_comma",         300),
                                label=",",     precision=0, minimum=0, step=50)
                            p_texcl_nb  = gr.Number(
                                value=_pron.get("pause_triple_excl",   800),
                                label="!!!",   precision=0, minimum=0, step=50)
                    with gr.Row():
                        save_pron_btn   = gr.Button(
                            "Enregistrer les silences", variant="primary", size="sm", scale=2)
                        reload_pron_btn = gr.Button(
                            "Recharger le fichier", size="sm", scale=1)

                # ── Utiliser à distance (API) ─────────────────────────
                with gr.Tab("À distance") as onglet_api:
                    api_md = gr.Markdown(_texte_api(), elem_id="api")
                    api_actualiser_btn = gr.Button("Actualiser l'adresse", size="sm")

    # ── Points d'API (invisibles dans l'interface) ─────────────────────────
    with gr.Column(visible=False):
        api_texte_in   = gr.Textbox(label="texte")
        api_voix_in    = gr.Textbox(label="voix")
        api_emotion_in = gr.Textbox(label="emotion", value="neutral")
        api_langue_in  = gr.Textbox(label="langue", value="French")
        api_audio_out  = gr.Audio(label="audio", type="filepath")
        api_parler_btn = gr.Button("parler")
        api_liste_out  = gr.JSON(label="voix")
        api_liste_btn  = gr.Button("voix")
    api_parler_btn.click(
        api_parler, inputs=[api_texte_in, api_voix_in, api_emotion_in, api_langue_in],
        outputs=[api_audio_out], api_name="parler",
        api_description="Fait dire un texte par une voix (clonée ou standard) ; renvoie un .wav.")
    def fn_texte_api(request: gr.Request):
        return _texte_api(request)

    onglet_api.select(fn_texte_api, outputs=[api_md], api_visibility="private")
    api_actualiser_btn.click(fn_texte_api, outputs=[api_md], api_visibility="private")
    api_liste_btn.click(api_voix, outputs=[api_liste_out], api_name="voix",
                        api_description="La liste des voix, de leurs versions, des émotions et des langues.")

    # ── Timer ──────────────────────────────────────────────────────────────
    timer = gr.Timer(value=1.0)
    timer.tick(
        fn_poll,
        outputs=[chatbot, vad_status_lbl, models_lbl, cfg_display,
                 read_status_lbl, read_progress_lbl],
        api_visibility="private")

    # ══════════════════════════════════════════════════════════════════════
    #  CONNEXIONS
    # ══════════════════════════════════════════════════════════════════════
    _cache = dict(api_visibility="private")       # ces événements ne font pas partie de l'API publique

    _settings = [voice_dd, model_dd, temp_sl, length_dd, strict_chk, sys_box]

    for _w in _settings:
        _w.change(fn_sync_config, inputs=_settings, **_cache)

    def fn_toggle_vad_ui():
        bouton, etat = fn_toggle_vad()
        ecoute = _vad_active
        return (gr.update(value="Arrêter l'écoute du micro" if ecoute else "Démarrer l'écoute du micro",
                          variant="stop" if ecoute else "primary"), etat)

    vad_btn.click(fn_toggle_vad_ui, outputs=[vad_btn, vad_status_lbl], **_cache)

    send_btn.click(
        fn_send_text, inputs=[text_in] + _settings,
        outputs=[chatbot, audio_out, text_in], **_cache)
    text_in.submit(
        fn_send_text, inputs=[text_in] + _settings,
        outputs=[chatbot, audio_out, text_in], **_cache)

    load_btn.click(
        fn_load_file,
        inputs=[file_in, file_action] + _settings,
        outputs=[ctx_state, ctx_status, chatbot, audio_out], **_cache)

    read_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[read_use_clone_chk], outputs=[read_clone_dd], **_cache)
    read_start_btn.click(
        fn_start_reading,
        inputs=[read_files_in, read_voice_dd, out_dir_box,
                read_use_clone_chk, read_clone_dd],
        outputs=[read_status_lbl, read_progress_lbl], **_cache)
    read_stop_btn.click(fn_stop_reading, outputs=[read_status_lbl, read_progress_lbl], **_cache)
    clone_chk.change(fn_toggle_clone, inputs=[clone_chk], outputs=[clone_mode_lbl], **_cache)

    # ── Voix clonées : bibliothèque et réglage ────────────────────────────
    _panneau_sorties = [rg_version_dd, rg_actuel_md, rg_bilans_md, rg_mode_radio, rg_cran_radio,
                        rg_mode_md, rg_cran_md, rg_audio]

    clone_dd.change(fn_select_clone, inputs=[clone_dd], outputs=[clone_status_lbl], **_cache)
    clone_dd.change(lambda n: fn_panneau(n, ""), inputs=[clone_dd], outputs=_panneau_sorties, **_cache)
    rg_version_dd.input(fn_panneau, inputs=[clone_dd, rg_version_dd], outputs=_panneau_sorties, **_cache)

    rg_mode_radio.input(fn_choix_reglage, inputs=[clone_dd, rg_version_dd, rg_mode_radio, rg_cran_radio],
                        outputs=[rg_mode_md, rg_cran_md, rg_audio], **_cache)
    rg_cran_radio.input(fn_choix_reglage, inputs=[clone_dd, rg_version_dd, rg_mode_radio, rg_cran_radio],
                        outputs=[rg_mode_md, rg_cran_md, rg_audio], **_cache)
    rg_ecouter_btn.click(fn_ecouter_reglage,
                         inputs=[clone_dd, rg_version_dd, rg_mode_radio, rg_cran_radio, rg_phrase_box],
                         outputs=[rg_audio, rg_etat_md], **_cache)
    rg_garder_btn.click(fn_garder_reglage, inputs=[clone_dd, rg_version_dd, rg_mode_radio, rg_cran_radio],
                        outputs=[rg_etat_md] + _panneau_sorties, **_cache)
    rg_test_btn.click(fn_refaire_test, inputs=[clone_dd, rg_version_dd],
                      outputs=[rg_progression_md] + _panneau_sorties, **_cache)
    rg_stop_btn.click(fn_arreter_test, outputs=[rg_progression_md], **_cache)

    # Met à jour la couverture émotionnelle quand on change de profil
    clone_dd.change(
        fn_select_clone_for_emotions,
        inputs=[clone_dd], outputs=[em_coverage_lbl], **_cache)
    clone_dd.change(
        lambda n: gr.update(value=n),
        inputs=[clone_dd], outputs=[em_profile_dd], **_cache)

    em_profile_dd.change(
        fn_select_clone_for_emotions,
        inputs=[em_profile_dd], outputs=[em_coverage_lbl], **_cache)

    em_add_btn.click(
        fn_add_emotion_and_test,
        inputs=[em_profile_dd, em_emotion_dd, em_audio_in, em_text_box],
        outputs=[em_status_lbl, em_coverage_lbl, clone_dd, rg_progression_md] + _panneau_sorties, **_cache)

    em_del_btn.click(
        fn_remove_emotion_sample,
        inputs=[em_profile_dd, em_emotion_dd],
        outputs=[em_status_lbl, em_coverage_lbl], **_cache)
    em_del_btn.click(lambda n: fn_panneau(n, ""), inputs=[clone_dd], outputs=_panneau_sorties, **_cache)

    clone_btn.click(
        fn_create_clone,
        inputs=[clone_audio_in, clone_ref_text_box, clone_name_box],
        outputs=[clone_status_lbl, clone_dd, tts_clone_dd, read_clone_dd, rg_progression_md]
        + _panneau_sorties, **_cache)
    # Sync des autres listes de voix après création / suppression
    clone_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[em_profile_dd], **_cache)

    clone_del_btn.click(
        fn_delete_clone,
        inputs=[clone_dd],
        outputs=[clone_status_lbl, clone_dd, tts_clone_dd, read_clone_dd], **_cache)
    clone_del_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[em_profile_dd], **_cache)

    # ── Lire un texte ──────────────────────────────────────────────────────
    tts_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[tts_use_clone_chk], outputs=[tts_clone_dd], **_cache)
    tts_synth_btn.click(
        fn_tts_direct,
        inputs=[tts_text_in, tts_voice_dd, tts_use_clone_chk, tts_clone_dd],
        outputs=[tts_audio_out], **_cache)

    # ── Redire un enregistrement ───────────────────────────────────────────
    s2s_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[s2s_use_clone_chk], outputs=[s2s_clone_dd], **_cache)
    s2s_transcribe_btn.click(
        fn_s2s_transcribe,
        inputs=[s2s_audio_in],
        outputs=[s2s_text_box, s2s_emotion_dd, s2s_note_lbl, s2s_status_lbl], **_cache)
    s2s_synth_btn.click(
        fn_s2s_resynthesize,
        inputs=[s2s_text_box, s2s_emotion_dd, s2s_voice_dd,
                s2s_use_clone_chk, s2s_clone_dd],
        outputs=[s2s_audio_out, s2s_result_lbl], **_cache)
    clone_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[s2s_clone_dd], **_cache)
    clone_del_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[s2s_clone_dd], **_cache)

    emotion_dd.change(fn_set_emotion, inputs=[emotion_dd], **_cache)
    language_dd.change(fn_set_language, inputs=[language_dd], **_cache)

    prev_btn.click(fn_preview_voice, inputs=[voice_dd], outputs=[preview_out], **_cache)
    refresh_btn.click(fn_refresh_models, outputs=[model_dd], **_cache)
    clear_btn.click(fn_clear, outputs=[chatbot, audio_out], **_cache)
    export_btn.click(fn_export, inputs=[chatbot], outputs=[export_dl], **_cache)
    def fn_arreter_ui():
        fn_shutdown_server()
        return gr.update(value="Arrêt en cours…", interactive=False)

    shutdown_btn.click(fn_arreter_ui, outputs=[shutdown_btn], **_cache)

    # ── Dictionnaire : ajout / suppression ────────────────────────────────
    lex_add_btn.click(
        fn_add_lexicon_entry,
        inputs=[lex_mot_box, lex_pron_box],
        outputs=[pron_status_lbl, lexicon_df, lex_mot_box, lex_pron_box], **_cache)
    lex_pron_box.submit(
        fn_add_lexicon_entry,
        inputs=[lex_mot_box, lex_pron_box],
        outputs=[pron_status_lbl, lexicon_df, lex_mot_box, lex_pron_box], **_cache)

    lex_del_btn.click(
        fn_delete_lexicon_entry,
        inputs=[lex_del_box],
        outputs=[pron_status_lbl, lexicon_df, lex_del_box], **_cache)

    # ── Silences + rechargement ────────────────────────────────────────────
    save_pron_btn.click(
        fn_save_lexicon,
        inputs=[lexicon_df, p_strong_nb, p_comma_nb, p_texcl_nb,
                p_dsemi_min_nb, p_dsemi_max_nb, p_dcolon_min_nb, p_dcolon_max_nb],
        outputs=[pron_status_lbl], **_cache)
    reload_pron_btn.click(
        fn_reload_pron,
        outputs=[pron_status_lbl, lexicon_df,
                 p_strong_nb, p_comma_nb, p_texcl_nb,
                 p_dsemi_min_nb, p_dsemi_max_nb, p_dcolon_min_nb, p_dcolon_max_nb], **_cache)


# ══════════════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════════════


def _watch_shutdown_request() -> None:
    """Observe la demande d'arrêt créée par l'UI ou le lanceur Windows."""
    while True:
        if _STOP_REQUEST_FILE.exists():
            _STOP_REQUEST_FILE.unlink(missing_ok=True)
            while not demo.is_running:
                time.sleep(0.1)
            print("[SERVER] Arrêt propre demandé.", flush=True)
            demo.close()
            return
        time.sleep(0.5)

if __name__ == "__main__":
    _STOP_REQUEST_FILE.unlink(missing_ok=True)
    threading.Thread(target=_watch_shutdown_request, daemon=True).start()
    print(f"\n{'='*60}")
    print(f"  Voix (botqwenClaude)  ·  Gradio {gr.__version__}")
    print(f"  Backend  ·  {_runtime.display_name}")
    print(f"  → sur cette machine : http://127.0.0.1:{_PORT}")
    if _RESEAU:
        for _ip in _adresses_reseau():
            print(f"  → sur le réseau local : http://{_ip}:{_PORT}")
        if _nom_bonjour():
            print(f"  → ou, sans se soucier de l'adresse IP : http://{_nom_bonjour()}:{_PORT}")
        print("    " + ("(mot de passe demandé, identifiant « voix »)" if _MOT_DE_PASSE else "(sans mot de passe)"))
        print(f"  → un lien qui parle : http://{_adresse_reseau()}:{_PORT}/dire?texte=Bonjour")
        print(f"  → pour vérifier qu'un téléphone atteint l'ordinateur : http://{_adresse_reseau()}:{_PORT}/ping")
        print("    (chaque appareil qui se connecte s'affiche ici : « 📱 Connexion depuis … »)")
    print("  Ctrl+C pour quitter")
    print(f"{'='*60}\n")

    _app, _, _ = demo.launch(
        server_name="0.0.0.0" if _RESEAU else "127.0.0.1",
        server_port=_PORT,
        share=False,                       # pas de lien public par Internet
        inbrowser=True,
        theme=THEME,
        css=CSS,
        auth=("voix", _MOT_DE_PASSE) if _MOT_DE_PASSE else None,
        footer_links=["api", "settings"],
        prevent_thread_lock=True,
        app_kwargs={"middleware": [__import__("starlette.middleware", fromlist=["Middleware"])
                                   .Middleware(_JournalConnexions)]},
    )
    _installer_adresses_simples(_app)
    try:
        demo.block_thread()
    except (KeyboardInterrupt, OSError):
        demo.close()
