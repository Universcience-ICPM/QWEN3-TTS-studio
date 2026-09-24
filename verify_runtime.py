#!/usr/bin/env python3
"""Diagnostic leger du backend botQwen, sans telecharger les modeles."""

from __future__ import annotations

import argparse
import importlib.metadata
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    runtime = create_runtime(root)
    print(f"Backend : {runtime.display_name}")
    print(f"STT     : {runtime.stt_name}")
    print(f"TTS CV  : {runtime.tts_source('custom')}")
    print(f"TTS Base: {runtime.tts_source('base')}")

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

    print("Diagnostic runtime : OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
