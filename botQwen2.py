#!/usr/bin/env python3
"""
botQwen2.py · Interface Gradio 6.x — STT local · Ollama LLM · Qwen3-TTS
──────────────────────────────────────────────────────────────────────────────
Architecture :
  • Thread VAD continu
      → écoute micro en permanence, détecte la voix, STT, LLM, TTS
      → se tait pendant que le bot parle
  • Interface Gradio (navigateur)
      → conversation en temps réel
      → lecture autonome de fichiers .txt / .docx
      → clonage vocal local via Qwen3-TTS (sans clé API)
      → marqueurs [Ryan] [triste] [voix:Profil] directement dans le manuscrit

Installation :
    pip install gradio python-docx sounddevice soundfile librosa --break-system-packages

Lancement :
    python botQwen.py
"""

import json
import random
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from botqwen_runtime import create_runtime, infer_model_type

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
        return f"🌡 {TEMPERATURE_CLONE_DEFAUT} (pas encore testée au banc d'essai)"
    verdict = str(reglages.get("verdict", "")).split(" :")[0]
    return (f"🌡 {reglages.get('temperature')} · {reglages.get('mode_clone')}"
            + (f" · {verdict}" if verdict else ""))


def _clone_choices() -> list[str]:
    return list(_clone_profiles.keys())


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


def fn_create_clone(audio_file, ref_text: str, voice_name: str):
    """Ajoute un profil vocal à la bibliothèque.
    Copie le fichier audio dans voices/ pour garantir un chemin stable entre sessions."""
    global _clone_name, _clone_ref_audio, _clone_ref_text, _clone_status
    _dd_empty = gr.update()
    if _tts is None:
        return "❌ TTS pas encore chargé — patientez…", _dd_empty, _dd_empty, _dd_empty
    if not audio_file:
        return "❌ Aucun fichier audio fourni", _dd_empty, _dd_empty, _dd_empty
    if not ref_text.strip():
        return "❌ Transcription manquante", _dd_empty, _dd_empty, _dd_empty
    try:
        name       = voice_name.strip() or Path(str(audio_file)).stem
        stable_path = _copy_to_voices(str(audio_file), name)      # relatif
        _clone_profiles[name] = {"ref_audio": stable_path, "ref_text": ref_text.strip()}
        _clone_name      = name
        _clone_ref_audio = _resolve_audio(stable_path)             # absolu pour le TTS
        _clone_ref_text  = ref_text.strip()
        _save_clone_file()
        _clone_status = f"✅ Profil « {name} » ajouté · {len(_clone_profiles)} profil(s)"
        dd = gr.update(choices=_clone_choices(), value=name)
        return _clone_status, dd, dd, dd   # clone_dd, tts_clone_dd, read_clone_dd
    except Exception as exc:
        _clone_status = f"❌ {exc}"
        return _clone_status, _dd_empty, _dd_empty, _dd_empty


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
    status = f"✅ Sample « {emotion} » ajouté au profil « {profile_name} »"
    return status, _emotion_coverage(profile_name)


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


def synthesize(text: str, voice: str, out: str = "bot_answer.wav") -> str | None:
    """Synthèse TTS → fichier WAV. Utilise la voix clonée si activée."""
    if _use_clone and _clone_ref_audio:
        audio = _qwen_clone_synth(text, _active_emotion)
    else:
        audio = _qwen_synth(text, voice, _active_emotion)
    if audio is None:
        return None
    try:
        sf.write(out, audio, TTS_SR)
        return out
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

    return str(out_path) if out_path.exists() and out_path.stat().st_size > 0 else None


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
#  INTERFACE GRADIO
# ══════════════════════════════════════════════════════════════════════════

_ollama_models = get_ollama_models()

_BG      = "#0f0f1a"
_BLOCK   = "#1c1c2e"
_SURFACE = "#252538"
_BORDER  = "#3a3a5c"
_VIOLET  = "#7c6af7"
_WHITE   = "#f0f0ff"
_SUBDUED = "#a8a8cc"
_GHOST   = "#5a5a80"

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill=_BG,
    body_text_color=_WHITE,
    body_text_color_subdued=_SUBDUED,
    background_fill_primary=_BLOCK,
    background_fill_secondary=_SURFACE,
    border_color_primary=_BORDER,
    border_color_accent=_VIOLET,
    block_background_fill=_BLOCK,
    block_border_color=_BORDER,
    block_title_text_color=_WHITE,
    block_label_background_fill=_SURFACE,
    block_label_text_color=_SUBDUED,
    block_info_text_color=_SUBDUED,
    panel_background_fill=_BLOCK,
    panel_border_color=_BORDER,
    input_background_fill=_SURFACE,
    input_border_color=_BORDER,
    input_border_color_focus=_VIOLET,
    input_placeholder_color=_GHOST,
    checkbox_background_color=_SURFACE,
    checkbox_border_color=_BORDER,
    checkbox_border_color_focus=_VIOLET,
    checkbox_border_color_selected=_VIOLET,
    checkbox_label_background_fill=_SURFACE,
    checkbox_label_background_fill_hover=_BLOCK,
    checkbox_label_background_fill_selected=_VIOLET,
    checkbox_label_border_color=_BORDER,
    checkbox_label_text_color=_WHITE,
    checkbox_label_text_color_selected="#ffffff",
    accordion_text_color=_WHITE,
    table_text_color=_WHITE,
    table_even_background_fill=_BLOCK,
    table_odd_background_fill=_SURFACE,
    table_border_color=_BORDER,
    button_primary_background_fill=_VIOLET,
    button_primary_background_fill_hover="#6a5ae0",
    button_primary_text_color="#ffffff",
    button_primary_text_color_hover="#ffffff",
    button_secondary_background_fill="#2a2a44",
    button_secondary_background_fill_hover="#333355",
    button_secondary_text_color=_WHITE,
    button_secondary_text_color_hover="#ffffff",
    button_cancel_background_fill="#5c1a2a",
    button_cancel_background_fill_hover="#7a2030",
    button_cancel_text_color="#ffffff",
    button_cancel_text_color_hover="#ffffff",
    slider_color=_VIOLET,
    error_background_fill="#2a0a0a",
    error_border_color="#f87171",
    error_text_color="#f87171",
    code_background_fill=_SURFACE,
    link_text_color=_VIOLET,
    link_text_color_hover="#9d8eff",
    link_text_color_visited="#7c6af7",
)

# Charge la bibliothèque de profils AVANT de construire l'UI
# (pas besoin du TTS pour ça — lecture JSON uniquement)
_rebuild_clone_prompt()

with gr.Blocks(theme=THEME, title="Voix - Assistant vocal local") as demo:

    ctx_state = gr.State("")

    gr.Markdown("# Voix\n"
                f"*{_runtime.stt_name} · Ollama LLM · Qwen3-TTS · "
                f"{_runtime.display_name}*")

    with gr.Row(equal_height=False):

        # ════════════════════════════════════════════════════════════════
        #  RÉGLAGES (sidebar)
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=300):

            vad_btn = gr.Button("▶ Démarrer l'écoute", variant="primary", size="lg")
            vad_status_lbl = gr.Textbox(
                value=_vad_status, label="État du micro",
                interactive=False, max_lines=1)

            # ── Voix TTS ───────────────────────────────────────────────
            with gr.Group():
                gr.Markdown("### 🎙 Voix TTS — Qwen3")
                voice_dd = gr.Dropdown(
                    choices=VOICES, value="Ryan",
                    label="Locuteur")
                language_dd = gr.Dropdown(
                    choices=LANGUAGES, value="French",
                    label="🌐 Langue",
                    info="French recommandé pour les romans en français")
                emotion_dd = gr.Dropdown(
                    choices=EMOTIONS, value="neutral",
                    label="🎭 Ton émotionnel",
                    info="Traduit en instruction naturelle pour Qwen3-TTS. "
                         "Utilisable aussi via marqueurs [joyeux] dans le texte.")
                with gr.Row():
                    prev_btn    = gr.Button("▶ Tester", size="sm", scale=1)
                    preview_out = gr.Audio(autoplay=True, show_label=False, scale=3)

            # ── Modèle LLM ─────────────────────────────────────────────
            with gr.Group():
                gr.Markdown("### 🧠 Modèle LLM")
                model_dd    = gr.Dropdown(
                    choices=_ollama_models,
                    value=_ollama_models[0] if _ollama_models else "mistral-nemo",
                    label="Modèle", show_label=False)
                refresh_btn = gr.Button("↺ Actualiser", size="sm")

            # ── Longueur réponse ───────────────────────────────────────
            with gr.Group():
                gr.Markdown("### 📏 Longueur des réponses")
                length_dd = gr.Dropdown(
                    choices=list(LONGUEUR.keys()),
                    value="💬 Normal — 1 paragraphe",
                    label="Mode", show_label=False)

            # ── Fichier contexte ───────────────────────────────────────
            with gr.Group():
                gr.Markdown("### 📄 Fichier contexte / lecture")
                file_in = gr.File(
                    label="Déposer un .txt ou .docx",
                    file_types=[".txt", ".docx"],
                    file_count="single",
                    type="filepath")
                file_action = gr.Radio(
                    choices=[
                        "📌 Utiliser comme contexte",
                        "📝 Résumé automatique",
                        "🔊 Lire à voix haute",
                    ],
                    value="📌 Utiliser comme contexte",
                    show_label=False)
                load_btn   = gr.Button("Charger →", variant="primary", size="sm")
                ctx_status = gr.Textbox(
                    value="Aucun contexte", label="Contexte actif",
                    interactive=False, max_lines=2)
                strict_chk = gr.Checkbox(
                    label="Mode strict — contexte uniquement", value=False)

            # ── Avancé ─────────────────────────────────────────────────
            with gr.Accordion("🔧 Avancé", open=False):
                temp_sl = gr.Slider(0.0, 2.0, 0.7, step=0.05, label="Température")
                sys_box = gr.Textbox(
                    label="System prompt personnalisé",
                    placeholder="Laissez vide pour le prompt automatique…",
                    lines=4)

            models_lbl = gr.Textbox(
                value=_models_status, label="Modèles",
                interactive=False, max_lines=1)
            shutdown_btn = gr.Button(
                "■ Arrêter Voix",
                variant="stop",
                size="sm",
            )

        # ════════════════════════════════════════════════════════════════
        #  ONGLETS
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=2):
            with gr.Tabs():

                # ── Onglet 0 : TTS Direct ─────────────────────────────
                with gr.Tab("🔊 TTS Direct"):
                    _tts_caps = {
                        "base":   "⚠️ **Modèle Base détecté** — clonage vocal uniquement. "
                                  "Les voix nommées (Ryan, Vivian…) et les marqueurs d'émotion "
                                  "nécessitent le modèle **CustomVoice**. "
                                  "En mode standard, une voix neutre constante est générée.",
                        "custom": "✅ **Modèle CustomVoice** — voix nommées + émotions disponibles.",
                        "design": "🎨 **Modèle VoiceDesign** — décrivez la voix dans le champ instruct.",
                    }.get(_TTS_MODEL_TYPE, "")
                    gr.Markdown(_tts_caps)
                    gr.Markdown(
                        "Synthétisez n'importe quel texte — "
                        "marqueurs `[triste]` `[Ryan]` `[voix:NomDuProfil]` supportés selon le modèle.\n\n"
                        "⚠️ `[voix:Profil]` cible un profil de la bibliothèque de voix clonées. "
                        "Vous pouvez alterner entre plusieurs profils clonés dans un même texte, "
                        "mais **ne mélangez pas** `[voix:Profil]` et `[Ryan]`/etc. dans le même passage "
                        "— le modèle TTS chargé (Base ou CustomVoice) ne change pas automatiquement "
                        "en cours de synthèse.")
                    tts_text_in = gr.Textbox(
                        label="Texte à synthétiser",
                        placeholder=(
                            "Entrez votre texte ici…\n\n"
                            "Utilisez des marqueurs : [joyeux] Bonjour ! [triste] Au revoir…\n"
                            "Ou une voix clonée : [voix:Marielle] Bonjour…"),
                        lines=6)
                    with gr.Row():
                        tts_voice_dd = gr.Dropdown(
                            label="Voix standard",
                            choices=VOICES, value=VOICES[0], scale=2)
                        tts_use_clone_chk = gr.Checkbox(
                            label="🎤 Voix clonée", value=False, scale=1)
                        tts_clone_dd = gr.Dropdown(
                            label="Profil cloné",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            scale=2, visible=False)
                    tts_synth_btn = gr.Button(
                        "🔊 Synthétiser", variant="primary")
                    tts_audio_out = gr.Audio(
                        label="🎧 Résultat", autoplay=True)

                # ── Onglet 1 : Discussion ─────────────────────────────
                with gr.Tab("💬 Discussion"):
                    cfg_display = gr.Markdown(value=_cfg_badge())
                    chatbot = gr.Chatbot(
                        label="Conversation", height=440,
                        layout="bubble", render_markdown=True,
                        avatar_images=(None, "🤖"))
                    audio_out = gr.Audio(label="🔊 Réponse", autoplay=True)
                    with gr.Group():
                        gr.Markdown(
                            "#### ⌨️ Message texte\n"
                            "<small>En parallèle du micro — Entrée pour envoyer.</small>")
                        with gr.Row():
                            text_in  = gr.Textbox(
                                placeholder="Tapez votre message…",
                                show_label=False, scale=5, lines=1, max_lines=3)
                            send_btn = gr.Button(
                                "Envoyer ↑", variant="primary", scale=1, min_width=110)
                    with gr.Row():
                        clear_btn  = gr.Button("🗑 Effacer", variant="stop")
                        export_btn = gr.Button("💾 Exporter JSON")
                    export_dl = gr.File(label="Export", visible=False)

                # ── Onglet 2 : Lecture autonome ───────────────────────
                with gr.Tab("📖 Lecture autonome"):
                    gr.Markdown(
                        "Synthétisez un ou plusieurs fichiers `.txt` / `.docx` "
                        "intégralement, **sans interagir avec le bot**.\n\n"
                        "**🎭 Marqueurs dans le texte** — voix et émotion à la volée  \n"
                        "Insérez des marqueurs directement dans votre manuscrit :\n\n"
                        "**Voix standard** : `[Ryan]` `[Aiden]` `[Vivian]` `[Serena]` "
                        "`[Uncle_Fu]` `[Dylan]` `[Eric]` `[Ono_Anna]` `[Sohee]`  \n"
                        "**Voix clonée** : `[voix:NomDuProfil]` — cible un profil de "
                        "l'onglet 🎤 Voix clonée (ex : `[voix:Marielle]`). Vous pouvez "
                        "alterner entre plusieurs profils clonés dans un même texte.  \n"
                        "⚠️ Ne mélangez pas `[voix:Profil]` et les voix standard "
                        "(`[Ryan]`…) dans le même texte — le modèle TTS chargé ne "
                        "change pas à la volée entre les deux familles.  \n"
                        "**Émotion** : `[neutre]` `[joyeux]` `[triste]` `[excité]` "
                        "`[curieux]` `[colère]`  \n"
                        "*(versions EN : `[neutral]` `[happy]` `[sad]` `[excited]` "
                        "`[curious]` `[angry]`)*  \n\n"
                        "Exemple : `[Ryan] [triste]` puis le texte du personnage.  \n"
                        "Exemple voix clonée : `[voix:Marielle] [triste]` puis le texte.  \n\n"
                        "**⏱ Pauses** : insérez `::` (longue) ou `;;` (moyenne) dans le "
                        "texte pour ajouter un silence. Durées réglables dans l'onglet "
                        "**Dictionnaire**.")

                    read_files_in = gr.File(
                        label="Fichiers à synthétiser — sélection multiple possible",
                        file_types=[".txt", ".docx"],
                        file_count="multiple",
                        type="filepath", height=130)

                    out_dir_box = gr.Textbox(
                        value=str(Path.home() / "Downloads"),
                        label="📁 Dossier de sortie — les .wav seront enregistrés ici",
                        placeholder="/Users/toi/Documents",
                        interactive=True, max_lines=1)

                    with gr.Row():
                        read_voice_dd = gr.Dropdown(
                            label="Voix standard",
                            choices=VOICES, value=VOICES[0], scale=2)
                        read_use_clone_chk = gr.Checkbox(
                            label="🎤 Voix clonée", value=False, scale=1)
                        read_clone_dd = gr.Dropdown(
                            label="Profil cloné",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            scale=2, visible=False)
                    # clone_chk / clone_mode_lbl conservés pour les connexions héritées
                    clone_chk      = gr.Checkbox(visible=False, value=False)
                    clone_mode_lbl = gr.Textbox(visible=False, value="")

                    with gr.Row():
                        read_start_btn = gr.Button(
                            "▶ Synthétiser", variant="primary", scale=2)
                        read_stop_btn  = gr.Button(
                            "⏹ Arrêter", variant="stop", scale=1)

                    with gr.Group():
                        read_status_lbl = gr.Textbox(
                            value=_read_status, label="État",
                            interactive=False, max_lines=1)
                        read_progress_lbl = gr.Textbox(
                            value=_read_progress, label="Fichiers sauvegardés",
                            interactive=False, max_lines=4)

                # ── Onglet 3 : Speech-to-Speech ───────────────────────
                with gr.Tab("🎭 Speech-to-Speech"):
                    gr.Markdown(
                        "**⚠️ À lire avant utilisation**\n\n"
                        "Ceci n'est **pas** un vrai transfert de prosodie (voice "
                        f"conversion). Le pipeline transcrit votre audio "
                        f"({_runtime.stt_name}), "
                        "estime une **émotion globale approximative** à partir du "
                        "signal, puis resynthétise le **texte** avec la voix "
                        "choisie et cette émotion. Le phrasé exact de l'original "
                        "(timing, débit, pauses, accents) n'est **pas** préservé — "
                        "seuls le contenu et une tonalité approchée le sont.")
                    if not HAS_LIBROSA:
                        gr.Markdown(
                            "ℹ️ `librosa` n'est pas installé — la détection "
                            "d'émotion utilisera systématiquement « neutre ». "
                            "Installez-le avec :\n"
                            "`pip install librosa --break-system-packages`")

                    s2s_audio_in = gr.Audio(
                        label="1️⃣ Audio source — la voix/le phrasé à analyser",
                        sources=["upload", "microphone"], type="filepath")
                    s2s_transcribe_btn = gr.Button(
                        "🔍 Analyser (transcrire + estimer l'émotion)",
                        variant="primary")
                    s2s_status_lbl = gr.Textbox(
                        value="", label="Statut", interactive=False, max_lines=2)

                    with gr.Group():
                        gr.Markdown("### 2️⃣ Résultat — modifiable avant synthèse")
                        s2s_text_box = gr.Textbox(
                            label="Texte transcrit (corrigez si besoin)",
                            lines=5, placeholder="Le texte transcrit apparaîtra ici…")
                        s2s_emotion_dd = gr.Dropdown(
                            choices=EMOTIONS, value="neutral",
                            label="🎭 Émotion estimée (corrigez si besoin)")
                        s2s_note_lbl = gr.Textbox(
                            value="", label="Détail de l'estimation (indicatif)",
                            interactive=False, max_lines=2)

                    with gr.Group():
                        gr.Markdown("### 3️⃣ Resynthèse")
                        with gr.Row():
                            s2s_voice_dd = gr.Dropdown(
                                label="Voix standard",
                                choices=VOICES, value=VOICES[0], scale=2)
                            s2s_use_clone_chk = gr.Checkbox(
                                label="🎤 Voix clonée", value=False, scale=1)
                            s2s_clone_dd = gr.Dropdown(
                                label="Profil cloné",
                                choices=_clone_choices(),
                                value=_clone_name or None,
                                scale=2, visible=False)
                        s2s_synth_btn = gr.Button(
                            "🔊 Resynthétiser avec cette voix", variant="primary")
                        s2s_audio_out = gr.Audio(
                            label="🎧 Résultat", autoplay=True)
                        s2s_result_lbl = gr.Textbox(
                            value="", label="Statut synthèse",
                            interactive=False, max_lines=1)

                # ── Onglet 4 : Dictionnaire ───────────────────────────
                with gr.Tab("📖 Dictionnaire"):
                    gr.Markdown(
                        "Corrige la **prononciation** des mots / sigles et règle les "
                        "**pauses** insérées dans l'audio lors de la lecture.\n\n"
                        "S'applique à la **Lecture autonome** et au **TTS Direct**.")

                    with gr.Group():
                        gr.Markdown("#### 📖 Dictionnaire de prononciation")
                        pron_status_lbl = gr.Textbox(
                            value=_pron_status,
                            label=f"Statut · {PRON_FILE.name}",
                            interactive=False, max_lines=1)

                        # ── Ajout d'une entrée ─────────────────────────────
                        gr.Markdown("**➕ Ajouter / modifier une entrée**")
                        gr.Markdown(
                            "<small>Format : `mot original` = `prononciation` · "
                            "ex : `H.G.W.` = `Agé Vaï`&nbsp;&nbsp;|&nbsp;&nbsp;"
                            "`M.` = `Monsieur`&nbsp;&nbsp;|&nbsp;&nbsp;"
                            "`Jean-Luc` = `Jean Luc`</small>")
                        with gr.Row():
                            lex_mot_box  = gr.Textbox(
                                label="Mot original",
                                placeholder="ex : H.G.W.",
                                scale=2, max_lines=1)
                            lex_pron_box = gr.Textbox(
                                label="Prononciation",
                                placeholder="ex : Agé Vaï",
                                scale=2, max_lines=1)
                            lex_add_btn  = gr.Button(
                                "➕ Ajouter", variant="primary", scale=1, size="sm")

                        # ── Suppression ────────────────────────────────────
                        with gr.Row():
                            lex_del_box = gr.Textbox(
                                label="Supprimer un mot",
                                placeholder="Tapez le mot exact à supprimer",
                                scale=3, max_lines=1)
                            lex_del_btn = gr.Button(
                                "🗑 Supprimer", variant="stop", scale=1, size="sm")

                        # ── Tableau lecture seule ──────────────────────────
                        lexicon_df = gr.Dataframe(
                            value=_pron_data_as_rows(),
                            headers=["Mot original", "Prononciation"],
                            datatype=["str", "str"],
                            col_count=(2, "fixed"),
                            row_count=(8, "dynamic"),
                            type="array",
                            label="Lexique actuel",
                            interactive=False)

                    with gr.Group():
                        gr.Markdown("#### ⏱ Pauses sur ponctuation spéciale (ms)")
                        gr.Markdown(
                            "<small>**`::` et `;;` insèrent un vrai silence dans l'audio.** "
                            "La durée est tirée au hasard entre **min** et **max** à chaque "
                            "occurrence → respiration plus naturelle "
                            "(mets min = max pour une durée fixe). "
                            "Les marqueurs eux-mêmes ne sont pas prononcés.<br>"
                            "Les champs `. ! ?`, `,` et `!!!` sont enregistrés "
                            "mais **pas encore appliqués** à l'audio.</small>")

                        gr.Markdown("**Marqueurs actifs**")
                        with gr.Row():
                            p_dcolon_min_nb = gr.Number(
                                value=_pron.get("pause_double_colon_min", 1000),
                                label=":: min", precision=0, minimum=0, step=100)
                            p_dcolon_max_nb = gr.Number(
                                value=_pron.get("pause_double_colon_max", 3000),
                                label=":: max", precision=0, minimum=0, step=100)
                            p_dsemi_min_nb  = gr.Number(
                                value=_pron.get("pause_double_semi_min",  700),
                                label=";; min", precision=0, minimum=0, step=50)
                            p_dsemi_max_nb  = gr.Number(
                                value=_pron.get("pause_double_semi_max",  1500),
                                label=";; max", precision=0, minimum=0, step=50)

                        gr.Markdown("**Réglages enregistrés (non appliqués pour l'instant)**")
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
                                "💾 Enregistrer pauses", variant="primary", size="sm", scale=2)
                            reload_pron_btn = gr.Button(
                                "↺ Recharger", size="sm", scale=1)

                # ── Onglet 5 : Voix clonée (Qwen3-TTS local) ──────────
                with gr.Tab("🎤 Voix clonée"):
                    gr.Markdown(
                        "Clonez n'importe quelle voix à partir de quelques secondes "
                        "d'audio de référence — **100 % local, sans clé API**.\n\n"
                        "Chaque profil est sauvegardé dans `voices_qwen.json` "
                        "et rechargé automatiquement au prochain lancement.")

                    with gr.Group():
                        gr.Markdown("#### 📋 Bibliothèque de profils")
                        with gr.Row():
                            clone_dd = gr.Dropdown(
                                label="Profil actif",
                                choices=_clone_choices(),
                                value=_clone_name or None,
                                scale=3, interactive=True)
                            clone_del_btn = gr.Button(
                                "🗑 Supprimer", variant="stop",
                                scale=1, min_width=110)
                        clone_status_lbl = gr.Textbox(
                            value=_clone_status, label="Statut",
                            interactive=False, max_lines=2)

                    with gr.Group():
                        gr.Markdown("#### 🎙 Ajouter un profil vocal")
                        clone_audio_in = gr.File(
                            label="Audio de référence — ton neutre (.wav .mp3 .m4a)",
                            file_types=[".wav", ".mp3", ".m4a", ".ogg"],
                            file_count="single",
                            type="filepath")
                        clone_ref_text_box = gr.Textbox(
                            label="Transcription exacte de l'audio",
                            placeholder="Écrivez ici mot pour mot ce qui est dit dans l'audio…",
                            lines=3)
                        clone_name_box = gr.Textbox(
                            label="Nom du profil",
                            placeholder="ex. Narratrice-Hanta",
                            max_lines=1)
                        clone_btn = gr.Button(
                            "➕ Ajouter à la bibliothèque", variant="primary")

                    with gr.Accordion("🎭 Enrichir en émotions", open=False):
                        gr.Markdown(
                            "Le modèle Base **ignore `instruct`** quand un `ref_audio` est fourni.  \n"
                            "L'émotion sur voix clonée passe donc par un **sample de référence "
                            "de la même voix dans cet état émotionnel**.  \n\n"
                            "Pour chaque émotion, fournissez un extrait de 5–15 s de la voix "
                            "cible exprimant cette émotion, plus sa transcription exacte.  \n"
                            "Si un sample manque, l'émotion utilise le sample neutre (sans effet visible).")
                        em_profile_dd = gr.Dropdown(
                            label="Profil à enrichir",
                            choices=_clone_choices(),
                            value=_clone_name or None,
                            interactive=True)
                        em_coverage_lbl = gr.Textbox(
                            value=_emotion_coverage(_clone_name),
                            label="Couverture actuelle",
                            interactive=False, max_lines=3)
                        em_emotion_dd = gr.Dropdown(
                            label="Émotion",
                            choices=[e for e in EMOTIONS if e != "neutral"],
                            value="sad")
                        em_audio_in = gr.File(
                            label="Audio de référence pour cette émotion",
                            file_types=[".wav", ".mp3", ".m4a", ".ogg"],
                            file_count="single", type="filepath")
                        em_text_box = gr.Textbox(
                            label="Transcription exacte",
                            placeholder="Transcrivez mot pour mot l'audio ci-dessus…",
                            lines=3)
                        with gr.Row():
                            em_add_btn = gr.Button(
                                "✅ Ajouter / Remplacer", variant="primary", scale=2)
                            em_del_btn = gr.Button(
                                "🗑 Supprimer", variant="stop", scale=1)
                        em_status_lbl = gr.Textbox(
                            value="", label="Statut",
                            interactive=False, max_lines=2)

    # ── Timer ──────────────────────────────────────────────────────────────
    timer = gr.Timer(value=1.0)
    timer.tick(
        fn_poll,
        outputs=[chatbot, vad_status_lbl, models_lbl, cfg_display,
                 read_status_lbl, read_progress_lbl])

    # ══════════════════════════════════════════════════════════════════════
    #  CONNEXIONS
    # ══════════════════════════════════════════════════════════════════════

    _settings = [voice_dd, model_dd, temp_sl, length_dd, strict_chk, sys_box]

    for _w in _settings:
        _w.change(fn_sync_config, inputs=_settings)

    vad_btn.click(fn_toggle_vad, outputs=[vad_btn, vad_status_lbl])

    send_btn.click(
        fn_send_text, inputs=[text_in] + _settings,
        outputs=[chatbot, audio_out, text_in])
    text_in.submit(
        fn_send_text, inputs=[text_in] + _settings,
        outputs=[chatbot, audio_out, text_in])

    load_btn.click(
        fn_load_file,
        inputs=[file_in, file_action] + _settings,
        outputs=[ctx_state, ctx_status, chatbot, audio_out])

    read_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[read_use_clone_chk], outputs=[read_clone_dd])
    read_start_btn.click(
        fn_start_reading,
        inputs=[read_files_in, read_voice_dd, out_dir_box,
                read_use_clone_chk, read_clone_dd],
        outputs=[read_status_lbl, read_progress_lbl])
    read_stop_btn.click(fn_stop_reading, outputs=[read_status_lbl, read_progress_lbl])
    clone_chk.change(fn_toggle_clone, inputs=[clone_chk], outputs=[clone_mode_lbl])

    clone_dd.change(fn_select_clone, inputs=[clone_dd], outputs=[clone_status_lbl])

    # Met à jour la couverture émotionnelle quand on change de profil dans l'accordéon
    clone_dd.change(
        fn_select_clone_for_emotions,
        inputs=[clone_dd], outputs=[em_coverage_lbl])
    clone_dd.change(
        lambda n: gr.update(value=n),
        inputs=[clone_dd], outputs=[em_profile_dd])

    em_profile_dd.change(
        fn_select_clone_for_emotions,
        inputs=[em_profile_dd], outputs=[em_coverage_lbl])

    em_add_btn.click(
        fn_add_emotion_sample,
        inputs=[em_profile_dd, em_emotion_dd, em_audio_in, em_text_box],
        outputs=[em_status_lbl, em_coverage_lbl])

    em_del_btn.click(
        fn_remove_emotion_sample,
        inputs=[em_profile_dd, em_emotion_dd],
        outputs=[em_status_lbl, em_coverage_lbl])

    clone_btn.click(
        fn_create_clone,
        inputs=[clone_audio_in, clone_ref_text_box, clone_name_box],
        outputs=[clone_status_lbl, clone_dd, tts_clone_dd, read_clone_dd])
    # Sync dropdown enrichissement après création d'un profil
    clone_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[em_profile_dd])

    clone_del_btn.click(
        fn_delete_clone,
        inputs=[clone_dd],
        outputs=[clone_status_lbl, clone_dd, tts_clone_dd, read_clone_dd])
    clone_del_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[em_profile_dd])

    # ── TTS Direct ─────────────────────────────────────────────────────────
    tts_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[tts_use_clone_chk], outputs=[tts_clone_dd])
    tts_synth_btn.click(
        fn_tts_direct,
        inputs=[tts_text_in, tts_voice_dd, tts_use_clone_chk, tts_clone_dd],
        outputs=[tts_audio_out])

    # ── Speech-to-Speech ───────────────────────────────────────────────────
    s2s_use_clone_chk.change(
        fn_toggle_clone_visibility,
        inputs=[s2s_use_clone_chk], outputs=[s2s_clone_dd])
    s2s_transcribe_btn.click(
        fn_s2s_transcribe,
        inputs=[s2s_audio_in],
        outputs=[s2s_text_box, s2s_emotion_dd, s2s_note_lbl, s2s_status_lbl])
    s2s_synth_btn.click(
        fn_s2s_resynthesize,
        inputs=[s2s_text_box, s2s_emotion_dd, s2s_voice_dd,
                s2s_use_clone_chk, s2s_clone_dd],
        outputs=[s2s_audio_out, s2s_result_lbl])
    # Garde les profils clonés synchronisés avec les autres onglets
    clone_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[s2s_clone_dd])
    clone_del_btn.click(
        lambda: gr.update(choices=_clone_choices()),
        outputs=[s2s_clone_dd])

    emotion_dd.change(fn_set_emotion, inputs=[emotion_dd])
    language_dd.change(fn_set_language, inputs=[language_dd])

    prev_btn.click(fn_preview_voice, inputs=[voice_dd], outputs=[preview_out])
    refresh_btn.click(fn_refresh_models, outputs=[model_dd])
    clear_btn.click(fn_clear, outputs=[chatbot, audio_out])
    export_btn.click(fn_export, inputs=[chatbot], outputs=[export_dl])
    shutdown_btn.click(fn_shutdown_server)

    # ── Dictionnaire : ajout / suppression ────────────────────────────────
    lex_add_btn.click(
        fn_add_lexicon_entry,
        inputs=[lex_mot_box, lex_pron_box],
        outputs=[pron_status_lbl, lexicon_df, lex_mot_box, lex_pron_box])
    # Ajout aussi en appuyant Entrée dans le champ prononciation
    lex_pron_box.submit(
        fn_add_lexicon_entry,
        inputs=[lex_mot_box, lex_pron_box],
        outputs=[pron_status_lbl, lexicon_df, lex_mot_box, lex_pron_box])

    lex_del_btn.click(
        fn_delete_lexicon_entry,
        inputs=[lex_del_box],
        outputs=[pron_status_lbl, lexicon_df, lex_del_box])

    # ── Pauses + rechargement ──────────────────────────────────────────────
    save_pron_btn.click(
        fn_save_lexicon,
        inputs=[lexicon_df, p_strong_nb, p_comma_nb, p_texcl_nb,
                p_dsemi_min_nb, p_dsemi_max_nb, p_dcolon_min_nb, p_dcolon_max_nb],
        outputs=[pron_status_lbl])
    reload_pron_btn.click(
        fn_reload_pron,
        outputs=[pron_status_lbl, lexicon_df,
                 p_strong_nb, p_comma_nb, p_texcl_nb,
                 p_dsemi_min_nb, p_dsemi_max_nb, p_dcolon_min_nb, p_dcolon_max_nb])


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
    print(f"\n{'='*56}")
    print(f"  Voix     ·  Gradio {gr.__version__}")
    print(f"  Backend  ·  {_runtime.display_name}")
    print("  → http://127.0.0.1:7860")
    print("  Ctrl+C pour quitter")
    print(f"{'='*56}\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
