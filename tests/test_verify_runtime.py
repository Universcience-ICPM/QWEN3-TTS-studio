from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from verify_runtime import load_clone_profiles


class CloneProfileDiagnosticTests(unittest.TestCase):
    def test_validates_relative_voice_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            voices = root / "voices"
            voices.mkdir()
            audio = voices / "narrateur.wav"
            audio.write_bytes(b"RIFF-test")
            (root / "voices_qwen.json").write_text(
                json.dumps(
                    {
                        "Narrateur": {
                            "ref_audio": "voices/narrateur.wav",
                            "ref_text": "Texte exact de la référence.",
                        },
                        "Invalide": {
                            "ref_audio": "voices/absent.wav",
                            "ref_text": "Texte.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            profiles, warnings = load_clone_profiles(root)

            self.assertEqual(list(profiles), ["Narrateur"])
            self.assertEqual(profiles["Narrateur"]["ref_audio"], str(audio.resolve()))
            self.assertTrue(any("Invalide" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
