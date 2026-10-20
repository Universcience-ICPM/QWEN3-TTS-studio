from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sfx_runtime import (
    StableAudioSFXRuntime,
    detect_sfx_backend,
    enrich_prompt,
    safe_slug,
)


class SFXBackendTests(unittest.TestCase):
    def test_auto_selects_mlx_on_apple_silicon(self):
        self.assertEqual(
            detect_sfx_backend("auto", system="Darwin", machine="arm64"),
            "mlx",
        )

    def test_auto_selects_tflite_on_windows(self):
        self.assertEqual(
            detect_sfx_backend("auto", system="Windows", machine="AMD64"),
            "tflite",
        )

    def test_prompt_enrichment_keeps_track_intent(self):
        self.assertIn("Continuous environmental ambience", enrich_prompt("Rain", "ambience"))
        self.assertIn("Isolated sound effect", enrich_prompt("Door slam", "sound_effect"))

    def test_slug_is_portable(self):
        self.assertEqual(safe_slug("Porte claquee !"), "porte-claquee")


class SFXGenerationTests(unittest.TestCase):
    def test_generation_uses_official_cli_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = root / ".stable-audio-3" / "optimized" / "mlx"
            engine.mkdir(parents=True)
            launcher = engine / "sa3"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")

            calls = []

            def fake_runner(command, **kwargs):
                calls.append((command, kwargs))
                output = Path(command[command.index("--out") + 1])
                output.write_bytes(b"RIFF-fake")
                return SimpleNamespace(returncode=0, stdout="saved", stderr="")

            runtime = StableAudioSFXRuntime(root, "mlx", runner=fake_runner)
            outputs, records, _ = runtime.generate_batch(
                prompt="Footsteps on wet gravel",
                track_type="sound_effect",
                duration=4,
                variants=2,
                seed=100,
                steps=8,
                cfg=1,
            )

            self.assertEqual(len(outputs), 2)
            self.assertEqual([record.seed for record in records], [100, 101])
            self.assertIn("sm-sfx", calls[0][0])
            self.assertEqual(
                Path(calls[0][1]["cwd"]).resolve(),
                engine.resolve(),
            )
            manifest = json.loads((outputs[0].parent / "manifest.json").read_text())
            self.assertEqual(len(manifest["generations"]), 2)

    def test_missing_engine_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = StableAudioSFXRuntime(Path(temp_dir), "mlx")
            with self.assertRaisesRegex(RuntimeError, "install_sfx_macos.sh"):
                runtime.generate_batch(
                    prompt="Thunder",
                    track_type="sound_effect",
                    duration=3,
                )

    def test_environment_can_override_engine_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "custom-engine"
            with patch.dict(os.environ, {"BOTQWEN_SFX_ENGINE": str(custom)}):
                runtime = StableAudioSFXRuntime(Path(temp_dir), "tflite")
            self.assertEqual(runtime.engine_dir, custom.resolve())


if __name__ == "__main__":
    unittest.main()
