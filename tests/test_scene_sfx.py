from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scene_sfx import (
    DocumentChapter,
    OllamaSceneAnalyzer,
    build_stable_audio_prompt_document,
    estimate_excerpt_position,
    mix_wav_files_to_voice_timeline,
    normalise_chapter_plan,
    plan_to_rows,
    rows_to_plans,
    sanitize_sound_prompt,
    split_text_chapters,
    stitch_wav_files,
    strip_dialogue,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class SceneParsingTests(unittest.TestCase):
    def test_plain_text_is_split_only_at_chapter_headings(self):
        chapters = split_text_chapters(
            "Chapitre 1\nLa pluie tombe.\n;;\nUne porte claque.\n"
            "Chapitre 2\nLe jour se leve.",
            "Roman",
        )
        self.assertEqual([chapter.title for chapter in chapters], ["Chapitre 1", "Chapitre 2"])
        self.assertIn(";;", chapters[0].text)

    def test_dialogue_is_removed_but_actions_remain(self):
        source = (
            "Elle posa le pot sur la table.\n"
            "- Vous partirez demain.\n"
            "\\- Cette ligne vient du Markdown.\n"
            "Le telephone vibra."
        )
        cleaned = strip_dialogue(source)
        self.assertNotIn("partirez", cleaned)
        self.assertNotIn("Markdown", cleaned)
        self.assertIn("posa le pot", cleaned)
        self.assertIn("telephone vibra", cleaned)

    def test_ollama_receives_narration_without_dialogue(self):
        calls = []

        def requester(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(
                {
                    "message": {
                        "content": """{
                          "summary": "Cuisine matinale",
                          "beats": [{
                            "label": "Pot",
                            "track_type": "sound_effect",
                            "duration_seconds": 5,
                            "position_percent": 25,
                            "prompt_en": "Glass jar lid resisting in a quiet kitchen, no voices",
                            "source_excerpt": "Le couvercle resistait"
                          }]
                        }"""
                    }
                }
            )

        analyzer = OllamaSceneAnalyzer("test-model", requester=requester)
        plan = analyzer.analyze_chapter(
            DocumentChapter(1, "Chapitre 1", "Le pot resiste.\n- Bonjour madame."),
            max_beats=4,
            default_duration=6,
        )
        user_prompt = calls[0][1]["json"]["messages"][1]["content"]
        self.assertNotIn("Bonjour madame", user_prompt)
        self.assertEqual(plan.beats[0].position_ratio, 0.25)
        self.assertEqual(calls[0][1]["json"]["format"], "json")

    def test_plan_rows_round_trip_positions(self):
        chapter = DocumentChapter(1, "Chapitre 1", "Une porte claque.")
        plan = normalise_chapter_plan(
            chapter,
            {
                "beats": [
                    {
                        "label": "Porte",
                        "type": "sound_effect",
                        "duration_seconds": 4,
                        "position_percent": 72,
                        "prompt_en": "A heavy door slam, no voice",
                    }
                ]
            },
            max_beats=3,
            default_duration=8,
        )
        restored = rows_to_plans(plan_to_rows([plan]), [chapter])
        self.assertAlmostEqual(restored[0].beats[0].position_ratio, 0.72)
        self.assertEqual(
            restored[0].beats[0].llm_prompt_en,
            "A heavy door slam, no voice",
        )

        report = build_stable_audio_prompt_document([plan], "roman.txt")
        self.assertIn("Prompt brut produit par le LLM", report)
        self.assertIn("A heavy door slam, no voice", report)
        self.assertIn("Prompt positif exact transmis a Stable Audio", report)
        self.assertIn("Isolated sound effect", report)
        self.assertIn("speech, dialogue, narration, voice, vocals, music", report)

    def test_long_chapter_plan_keeps_many_long_beats(self):
        chapter = DocumentChapter(1, "Chapitre long", "Une longue succession de sons.")
        payload = {
            "beats": [
                {
                    "label": f"Plan {index}",
                    "type": "ambience",
                    "duration_seconds": 90,
                    "position_percent": index,
                    "prompt_en": f"Environmental sound number {index}",
                }
                for index in range(1, 41)
            ]
        }

        plan = normalise_chapter_plan(
            chapter,
            payload,
            max_beats=40,
            default_duration=8,
        )
        self.assertEqual(len(plan.beats), 40)
        self.assertTrue(all(beat.duration == 90 for beat in plan.beats))

        rows = plan_to_rows([plan])
        rows[0][4] = 999
        restored = rows_to_plans(rows, [chapter])
        self.assertEqual(restored[0].beats[0].duration, 120)

    def test_vocal_clauses_are_removed_from_audio_prompts(self):
        prompt = sanitize_sound_prompt(
            "Phone ringing, then a short pause before the caller speaks in English, "
            "occasional grunts of effort"
        )
        self.assertIn("Phone ringing", prompt)
        self.assertNotIn("speaks in English", prompt)
        self.assertNotIn("grunts", prompt)
        self.assertIn("No speech", prompt)

    def test_excerpt_position_overrides_a_rough_llm_percentage(self):
        position = estimate_excerpt_position(
            "Une porte claque. Puis le telephone vibre sur la table.",
            "le telephone vibre",
            0.1,
        )
        self.assertGreater(position, 0.25)


class SceneAssemblyTests(unittest.TestCase):
    def _write_tone(self, path: Path, duration: float, sample_rate: int = 1000):
        frames = int(duration * sample_rate)
        signal = np.full((frames, 2), 0.1, dtype=np.float32)
        sf.write(path, signal, sample_rate)

    def test_sequential_stitch_uses_crossfade(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.wav"
            second = root / "second.wav"
            output = root / "joined.wav"
            self._write_tone(first, 1)
            self._write_tone(second, 1)
            result = stitch_wav_files([first, second], output, crossfade_seconds=0.1)
            self.assertAlmostEqual(result.duration, 1.9, places=3)
            self.assertEqual(result.start_times, (0.0, 0.9))

    def test_synced_track_matches_voice_duration_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "clip.wav"
            voice = root / "voice.wav"
            output = root / "synced.wav"
            self._write_tone(clip, 1, sample_rate=1000)
            self._write_tone(voice, 5, sample_rate=800)
            result = mix_wav_files_to_voice_timeline(
                [clip],
                [0.5],
                voice,
                output,
            )
            info = sf.info(output)
            self.assertEqual(info.frames, 5000)
            self.assertEqual(info.samplerate, 1000)
            self.assertAlmostEqual(result.duration, 5.0)
            self.assertAlmostEqual(result.start_times[0], 2.5)


if __name__ == "__main__":
    unittest.main()
