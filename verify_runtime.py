#!/usr/bin/env python3
"""Diagnostic leger du backend botQwen, sans telecharger les modeles."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import tempfile
from pathlib import Path

from botqwen_runtime import create_runtime


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-tts",
        action="store_true",
        help="Charge CustomVoice et genere une courte phrase de test.",
    )
    parser.add_argument(
        "--smoke-clone",
        metavar="PROFIL",
        help="Charge Base et genere un WAV avec ce profil de voices_qwen.json.",
    )
    args = parser.parse_args()
    if args.smoke_tts and args.smoke_clone:
        parser.error("utilisez --smoke-tts ou --smoke-clone, pas les deux")
    return args


def load_clone_profiles(root: Path) -> tuple[dict[str, dict], list[str]]:
    """Charge et valide les références vocales sans modifier le fichier utilisateur."""
    source = root / "voices_qwen.json"
    if not source.exists():
        return {}, ["voices_qwen.json absent"]
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"voices_qwen.json illisible : {exc}"]
    if "ref_audio" in data:
        data = {
            str(data.get("name") or "Profil 1"): {
                "ref_audio": data.get("ref_audio"),
                "ref_text": data.get("ref_text"),
            }
        }

    profiles: dict[str, dict] = {}
    warnings: list[str] = []
    for name, profile in data.items():
        if not isinstance(profile, dict):
            warnings.append(f"{name}: entrée JSON invalide")
            continue
        raw_audio = str(profile.get("ref_audio") or "").strip()
        ref_text = str(profile.get("ref_text") or "").strip()
        audio_path = Path(raw_audio)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        issues = []
        if not raw_audio or not audio_path.is_file():
            issues.append(f"audio introuvable ({raw_audio or 'chemin vide'})")
        if not ref_text:
            issues.append("transcription vide")
        if issues:
            warnings.append(f"{name}: {', '.join(issues)}")
            continue
        profiles[str(name)] = {
            **profile,
            "ref_audio": str(audio_path.resolve()),
            "ref_text": ref_text,
        }
    return profiles, warnings


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    runtime = create_runtime(root)
    print(f"Backend : {runtime.display_name}")
    print(f"STT     : {runtime.stt_name}")
    print(f"TTS CV  : {runtime.tts_source('custom')}")
    print(f"TTS Base: {runtime.tts_source('base')}")
    clone_profiles, clone_warnings = load_clone_profiles(root)
    print(f"Voix    : {len(clone_profiles)} profil(s) cloné(s) valide(s)")
    for warning in clone_warnings:
        print(f"AVERT.  : {warning}")

    if runtime.backend == "cuda":
        import torch
        import transformers
        from qwen_tts import Qwen3TTSModel

        if not torch.cuda.is_available():
            print("ERREUR  : torch.cuda.is_available() == False")
            return 1
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU     : {props.name} ({vram_gb:.1f} Go)")
        print(f"PyTorch : {torch.__version__} / CUDA {torch.version.cuda}")
        print(f"HF      : transformers {transformers.__version__}")
        print(f"qwen-tts: {package_version('qwen-tts')}")
        print(f"API TTS : {Qwen3TTSModel.__name__}")
        if package_version("qwen-tts") == "absent":
            return 1
    else:
        print(f"mlx-audio: {package_version('mlx-audio')}")
        if package_version("mlx-audio") == "absent":
            return 1

    if args.smoke_tts:
        source = runtime.tts_source("custom")
        if source is None:
            print("ERREUR  : modele CustomVoice introuvable")
            return 1
        print(f"Smoke TTS: chargement de {source}")
        model = runtime.load_tts(source)
        with tempfile.TemporaryDirectory(prefix="botqwen_smoke_") as temp_dir:
            audio = runtime.generate_audio(
                model=model,
                model_type="custom",
                temp_dir=temp_dir,
                kwargs={
                    "text": "Ceci est un test de synthese vocale.",
                    "language": "French",
                    "voice": "Ryan",
                },
            )
        if audio is None or audio.size == 0:
            print("ERREUR  : aucun signal audio genere")
            return 1
        duration = audio.size / runtime.sample_rate
        print(f"Smoke TTS: OK ({duration:.2f} s)")

    if args.smoke_clone:
        selected_name = next(
            (
                name
                for name in clone_profiles
                if name.casefold() == args.smoke_clone.casefold()
            ),
            None,
        )
        if selected_name is None:
            choices = ", ".join(clone_profiles) or "aucun"
            print(f"ERREUR  : profil cloné introuvable : {args.smoke_clone}")
            print(f"Profils : {choices}")
            return 1
        profile = clone_profiles[selected_name]
        source = runtime.tts_source("base")
        print(f"Smoke clone: chargement de {source}")
        model = runtime.load_tts(source)
        with tempfile.TemporaryDirectory(prefix="botqwen_clone_") as temp_dir:
            audio = runtime.generate_audio(
                model=model,
                model_type="base",
                temp_dir=temp_dir,
                kwargs={
                    "text": "Bonjour, ceci est un test de ma voix clonée en français.",
                    "language": "French",
                    "ref_audio": profile["ref_audio"],
                    "ref_text": profile["ref_text"],
                },
            )
        if audio is None or audio.size == 0:
            print("ERREUR  : aucun signal cloné généré")
            return 1
        import soundfile as sf

        output_dir = root / "outputs" / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", selected_name.lower()).strip("-") or "clone"
        output = output_dir / f"clone-{slug}.wav"
        sf.write(output, audio, runtime.sample_rate)
        duration = audio.size / runtime.sample_rate
        print(f"Smoke clone: OK ({duration:.2f} s) -> {output}")

    print("Diagnostic runtime : OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
