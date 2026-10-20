#!/usr/bin/env python3
"""Backends d'inference pour botQwen2 (MLX Apple Silicon et PyTorch CUDA)."""

from __future__ import annotations

import gc
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf


SUPPORTED_BACKENDS = {"mlx", "cuda"}


def detect_backend(
    requested: str | None = None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Selectionne le backend sans importer MLX ou PyTorch."""
    choice = (requested or os.getenv("BOTQWEN_BACKEND", "auto")).strip().lower()
    if choice != "auto":
        if choice not in SUPPORTED_BACKENDS:
            allowed = ", ".join(sorted(SUPPORTED_BACKENDS | {"auto"}))
            raise ValueError(f"BOTQWEN_BACKEND={choice!r} invalide ({allowed})")
        return choice

    host_system = (system or platform.system()).lower()
    host_machine = (machine or platform.machine()).lower()
    if host_system == "darwin" and host_machine in {"arm64", "aarch64"}:
        return "mlx"
    if host_system == "windows":
        return "cuda"
    raise RuntimeError(
        "Plateforme non detectee. Definissez BOTQWEN_BACKEND=mlx ou "
        "BOTQWEN_BACKEND=cuda."
    )


def infer_model_type(source: str | Path) -> str:
    name = str(source).lower()
    if "customvoice" in name or "custom_voice" in name:
        return "custom"
    if "voicedesign" in name or "voice_design" in name:
        return "design"
    return "base"


class BaseRuntime:
    backend = ""
    display_name = ""
    stt_name = ""
    sample_rate = 24_000
    release_before_tts_load = False

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.models_dir = self.project_dir / "models"

    def tts_source(self, model_type: str) -> str | Path | None:
        raise NotImplementedError

    def preferred_tts_source(self) -> str | Path:
        for model_type in ("custom", "design", "base"):
            source = self.tts_source(model_type)
            if source is not None:
                return source
        raise FileNotFoundError("Aucun modele Qwen3-TTS disponible")

    def load_stt(self) -> Any:
        raise NotImplementedError

    def load_tts(self, source: str | Path) -> Any:
        raise NotImplementedError

    def release_tts(self) -> None:
        gc.collect()

    def generate_audio(
        self,
        model: Any,
        model_type: str,
        temp_dir: str,
        kwargs: dict[str, Any],
    ) -> np.ndarray | None:
        raise NotImplementedError

    @staticmethod
    def source_label(source: str | Path) -> str:
        source_path = Path(str(source))
        return source_path.name if source_path.name else str(source)


class MLXRuntime(BaseRuntime):
    backend = "mlx"
    display_name = "MLX / Apple Silicon"
    stt_name = "Voxtral MLX"

    def __init__(self, project_dir: Path):
        super().__init__(project_dir)
        self.stt_model = os.getenv(
            "BOTQWEN_STT_MODEL",
            "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
        )

    def tts_source(self, model_type: str) -> Path | None:
        keywords = {
            "custom": "CustomVoice",
            "design": "VoiceDesign",
            "base": "Base",
        }
        keyword = keywords[model_type]
        candidates = sorted(self.models_dir.glob(f"*{keyword}*"))
        return candidates[0] if candidates else None

    def preferred_tts_source(self) -> Path:
        try:
            return super().preferred_tts_source()
        except FileNotFoundError:
            return self.models_dir / "Qwen3-TTS-12Hz-1.7B-Base-8bit"

    def load_stt(self) -> Any:
        from mlx_audio.stt.utils import load

        return load(self.stt_model)

    def load_tts(self, source: str | Path) -> Any:
        from mlx_audio.tts.utils import load_model

        return load_model(str(source))

    def generate_audio(
        self,
        model: Any,
        model_type: str,
        temp_dir: str,
        kwargs: dict[str, Any],
    ) -> np.ndarray | None:
        from mlx_audio.tts.generate import generate_audio

        mlx_kwargs = dict(kwargs)
        language = mlx_kwargs.pop("language", None)
        if language:
            mlx_kwargs["lang_code"] = str(language).lower()

        wav_path = Path(temp_dir) / "audio_000.wav"
        wav_path.unlink(missing_ok=True)
        generate_audio(model=model, output_path=temp_dir, **mlx_kwargs)
        if not wav_path.exists():
            return None
        data, _ = sf.read(str(wav_path), dtype="float32")
        wav_path.unlink(missing_ok=True)
        return np.asarray(data, dtype=np.float32)


class TransformersWhisperSTT:
    """Adapte une pipeline Whisper a l'interface .generate(...).text existante."""

    def __init__(self, pipe: Any, language: str = "french"):
        self.pipe = pipe
        self.language = language

    def generate(self, audio_path: str) -> SimpleNamespace:
        result = self.pipe(
            str(audio_path),
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        return SimpleNamespace(text=(result.get("text") or "").strip())


class CudaRuntime(BaseRuntime):
    backend = "cuda"
    display_name = "PyTorch / NVIDIA CUDA"
    stt_name = "Whisper CUDA"
    release_before_tts_load = True

    def __init__(self, project_dir: Path):
        super().__init__(project_dir)
        model_size = os.getenv("BOTQWEN_TTS_SIZE", "1.7B")
        self.sources = {
            "custom": os.getenv(
                "BOTQWEN_CUDA_CUSTOM_MODEL",
                f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice",
            ),
            "design": os.getenv(
                "BOTQWEN_CUDA_DESIGN_MODEL",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            ),
            "base": os.getenv(
                "BOTQWEN_CUDA_BASE_MODEL",
                f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base",
            ),
        }
        self.stt_model = os.getenv(
            "BOTQWEN_STT_MODEL",
            "openai/whisper-small",
        )

    def tts_source(self, model_type: str) -> str:
        return self.sources[model_type]

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch absent. Lancez install_windows_nvidia.ps1."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA indisponible. Verifiez le pilote NVIDIA et l'installation "
                "de la version CUDA de PyTorch."
            )
        return torch

    def load_stt(self) -> TransformersWhisperSTT:
        torch = self._torch()
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        dtype = torch.float16
        processor = AutoProcessor.from_pretrained(self.stt_model)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.stt_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda:0")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=0,
        )
        return TransformersWhisperSTT(
            pipe,
            language=os.getenv("BOTQWEN_STT_LANGUAGE", "french"),
        )

    def load_tts(self, source: str | Path) -> Any:
        torch = self._torch()
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return Qwen3TTSModel.from_pretrained(
            str(source),
            device_map="cuda:0",
            dtype=dtype,
            attn_implementation=os.getenv("BOTQWEN_CUDA_ATTENTION", "sdpa"),
        )

    def release_tts(self) -> None:
        gc.collect()
        try:
            torch = self._torch()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def generate_audio(
        self,
        model: Any,
        model_type: str,
        temp_dir: str,
        kwargs: dict[str, Any],
    ) -> np.ndarray | None:
        del temp_dir
        text = kwargs["text"]
        language = kwargs.get("language") or "Auto"
        generation_kwargs = {
            key: kwargs[key]
            for key in (
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
                "max_new_tokens",
            )
            if kwargs.get(key) is not None
        }
        if kwargs.get("max_tokens") is not None:
            generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]

        if model_type == "custom":
            wavs, _ = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=kwargs.get("voice") or "Ryan",
                instruct=kwargs.get("instruct"),
                **generation_kwargs,
            )
        elif model_type == "base":
            ref_audio = kwargs.get("ref_audio")
            if not ref_audio:
                raise ValueError("Le modele Base exige un audio de reference")
            wavs, _ = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=kwargs.get("ref_text"),
                **generation_kwargs,
            )
        elif model_type == "design":
            wavs, _ = model.generate_voice_design(
                text=text,
                language=language,
                instruct=kwargs.get("instruct")
                or "A natural, clear and expressive French-speaking voice.",
                **generation_kwargs,
            )
        else:
            raise ValueError(f"Type de modele TTS inconnu : {model_type}")

        if not wavs:
            return None
        return np.asarray(wavs[0], dtype=np.float32)


def create_runtime(project_dir: str | Path, requested: str | None = None) -> BaseRuntime:
    backend = detect_backend(requested)
    root = Path(project_dir)
    if backend == "mlx":
        return MLXRuntime(root)
    return CudaRuntime(root)
