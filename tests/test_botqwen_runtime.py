from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from botqwen_runtime import (
    CudaRuntime,
    MLXRuntime,
    TransformersWhisperSTT,
    detect_backend,
    infer_model_type,
)


class FakeQwenModel:
    def __init__(self):
        self.call = None

    def generate_custom_voice(self, **kwargs):
        self.call = ("custom", kwargs)
        return [np.array([0.1, -0.1], dtype=np.float32)], 24_000

    def generate_voice_clone(self, **kwargs):
        self.call = ("base", kwargs)
        return [np.array([0.2, -0.2], dtype=np.float32)], 24_000


class RuntimeSelectionTests(unittest.TestCase):
    def test_auto_selects_mlx_on_apple_silicon(self):
        self.assertEqual(
            detect_backend("auto", system="Darwin", machine="arm64"),
            "mlx",
        )

    def test_auto_selects_cuda_on_windows(self):
        self.assertEqual(
            detect_backend("auto", system="Windows", machine="AMD64"),
            "cuda",
        )

    def test_explicit_backend_wins(self):
        self.assertEqual(
            detect_backend("cuda", system="Darwin", machine="arm64"),
            "cuda",
        )

    def test_model_type_detection(self):
        self.assertEqual(infer_model_type("foo-CustomVoice"), "custom")
        self.assertEqual(infer_model_type("foo-VoiceDesign"), "design")
        self.assertEqual(infer_model_type("foo-Base"), "base")


class RuntimeModelTests(unittest.TestCase):
    def test_mlx_discovers_local_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models = Path(temp_dir) / "models"
            (models / "Qwen-CustomVoice-8bit").mkdir(parents=True)
            runtime = MLXRuntime(Path(temp_dir))
            self.assertEqual(
                runtime.tts_source("custom").resolve(),
                (models / "Qwen-CustomVoice-8bit").resolve(),
            )

    def test_cuda_model_size_can_be_configured(self):
        with patch.dict(os.environ, {"BOTQWEN_TTS_SIZE": "0.6B"}, clear=False):
            runtime = CudaRuntime(Path("."))
        self.assertIn("0.6B-CustomVoice", runtime.tts_source("custom"))
        self.assertIn("0.6B-Base", runtime.tts_source("base"))

    def test_cuda_loads_qwen_with_sdpa(self):
        calls = {}

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_bf16_supported():
                return True

        fake_torch = SimpleNamespace(
            cuda=FakeCuda(),
            bfloat16="bf16",
            float16="fp16",
        )

        class FakeQwen3TTSModel:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls["source"] = source
                calls["kwargs"] = kwargs
                return "loaded-model"

        fake_qwen = SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel)
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "qwen_tts": fake_qwen},
        ):
            runtime = CudaRuntime(Path("."))
            model = runtime.load_tts("Qwen/Test-CustomVoice")

        self.assertEqual(model, "loaded-model")
        self.assertEqual(calls["kwargs"]["device_map"], "cuda:0")
        self.assertEqual(calls["kwargs"]["dtype"], "bf16")
        self.assertEqual(calls["kwargs"]["attn_implementation"], "sdpa")

    def test_cuda_custom_voice_dispatch(self):
        runtime = CudaRuntime(Path("."))
        model = FakeQwenModel()
        audio = runtime.generate_audio(
            model,
            "custom",
            "",
            {
                "text": "Bonjour",
                "language": "French",
                "voice": "Ryan",
                "instruct": "Speak joyfully.",
            },
        )
        self.assertEqual(model.call[0], "custom")
        self.assertEqual(model.call[1]["speaker"], "Ryan")
        np.testing.assert_allclose(audio, [0.1, -0.1])

    def test_cuda_clone_dispatch(self):
        runtime = CudaRuntime(Path("."))
        model = FakeQwenModel()
        audio = runtime.generate_audio(
            model,
            "base",
            "",
            {
                "text": "Bonjour",
                "language": "French",
                "ref_audio": "voice.wav",
                "ref_text": "Texte de reference",
            },
        )
        self.assertEqual(model.call[0], "base")
        self.assertEqual(model.call[1]["ref_audio"], "voice.wav")
        np.testing.assert_allclose(audio, [0.2, -0.2])

    def test_whisper_adapter_exposes_generate_text(self):
        calls = {}

        def fake_pipe(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return {"text": "  Bonjour le monde.  "}

        stt = TransformersWhisperSTT(fake_pipe, language="french")
        result = stt.generate("sample.wav")
        self.assertEqual(result.text, "Bonjour le monde.")
        self.assertEqual(calls["path"], "sample.wav")
        self.assertEqual(
            calls["kwargs"]["generate_kwargs"]["language"],
            "french",
        )


if __name__ == "__main__":
    unittest.main()
