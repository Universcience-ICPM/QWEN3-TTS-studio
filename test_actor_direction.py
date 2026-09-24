#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from botqwen_runtime import create_runtime


ROOT = Path(__file__).resolve().parent
RUNTIME = create_runtime(ROOT)
CUSTOM_MODEL = RUNTIME.tts_source("custom")
BASE_MODEL = RUNTIME.tts_source("base")
VOICE_DB = ROOT / "voices_qwen.json"
OUT_ROOT = ROOT / "outputs" / "actor_direction_tests"

MAIN_TEXT = "Tu étais où ? Je t'ai appelé trois fois. Trois fois, Rakoto. Et toi... rien."
PAUSE_MARKER_RE = re.compile(r"\s*;;\s*")
NATIVE_SPEAKERS = [
    "Ryan",
    "Aiden",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
]


@dataclass(frozen=True)
class Direction:
    label: str
    display_name: str
    text: str
    instruct: str
    clone_text: str


DIRECTIONS: dict[str, Direction] = {
    "neutral": Direction(
        label="neutral",
        display_name="neutre",
        text=MAIN_TEXT,
        instruct=(
            "Speak naturally in a calm, neutral voice. "
            "No strong emotion. Medium speaking rate."
        ),
        clone_text=MAIN_TEXT,
    ),
    "restrained_anger": Direction(
        label="restrained_anger",
        display_name="colere contenue",
        text=MAIN_TEXT,
        instruct=(
            "Speak with restrained anger. The voice is tense but controlled. "
            "Keep the pitch slightly low at first. Use clipped articulation. "
            "Use short pauses. Stress 'trois fois' and 'rien'. "
            "The speaker is trying very hard not to shout. "
            "Let the anger become more audible toward the end."
        ),
        clone_text=(
            "Tu étais où ? ;; Je t'ai appelé trois fois. ;; "
            "Trois fois, Rakoto. ;; Et toi... rien."
        ),
    ),
    "cold_anger": Direction(
        label="cold_anger",
        display_name="colere froide",
        text=MAIN_TEXT,
        instruct=(
            "Speak with cold restrained anger. Keep the voice low, dry, and precise. "
            "Use very controlled energy and short silences. Do not shout. "
            "Let disappointment and contempt sit underneath the words."
        ),
        clone_text=(
            "Tu étais où. ;; Je t'ai appelé trois fois. ;; "
            "Trois fois, Rakoto. ;; Et toi... rien."
        ),
    ),
    "masked_worry": Direction(
        label="masked_worry",
        display_name="inquietude masquee",
        text=MAIN_TEXT,
        instruct=(
            "Speak as if the speaker is trying to hide worry behind irritation. "
            "Keep the voice controlled, but let small breathless hesitations appear. "
            "Do not sound openly afraid. The concern should leak through near the end."
        ),
        clone_text=(
            "Tu étais où ?... ;; Je t'ai appelé trois fois. ;; "
            "Trois fois, Rakoto... ;; Et toi... rien."
        ),
    ),
    "explosive_anger": Direction(
        label="explosive_anger",
        display_name="forte colere",
        text="Non ! Ça suffit maintenant ! Tu m'entends ? Maintenant, tu sors.",
        instruct=(
            "Speak with explosive anger. Strong projection. Sharp consonants. "
            "Fast breathing. Increase vocal intensity rapidly. Stress the accusations "
            "strongly. Raise the voice significantly on the final words. "
            "The speaker has just lost control."
        ),
        clone_text=(
            "Non ! ;; Ça suffit maintenant ! ;; "
            "Tu m'entends ? ;; Maintenant, tu sors !"
        ),
    ),
    "warm_joy": Direction(
        label="warm_joy",
        display_name="joie chaleureuse",
        text="Tu es là ! Je n'arrive pas à y croire. C'est une merveilleuse nouvelle.",
        instruct=(
            "Speak with genuine warm happiness. Smile naturally while speaking. "
            "The smile should be clearly audible in the voice. Use relaxed articulation "
            "and light melodic pitch variations. The speaker is deeply pleased, "
            "not childish or overexcited."
        ),
        clone_text=(
            "Tu es là ! ;; Je n'arrive pas à y croire. ;; "
            "C'est une merveilleuse nouvelle !"
        ),
    ),
    "tenderness": Direction(
        label="tenderness",
        display_name="tendresse",
        text="Viens là. Tout va bien maintenant.",
        instruct=(
            "Speak with quiet tenderness. Keep the voice soft, close, and reassuring. "
            "Use a gentle rhythm and warm articulation. Do not become sentimental."
        ),
        clone_text="Viens là. ;; Tout va bien maintenant.",
    ),
    "fatigue": Direction(
        label="fatigue",
        display_name="fatigue",
        text="Je vais bien. J'ai juste besoin d'une minute.",
        instruct=(
            "Speak with physical fatigue. Keep the energy low and the rhythm slow. "
            "Let the breath feel slightly heavy. Do not sound sad or dramatic."
        ),
        clone_text="Je vais bien. ;; J'ai juste besoin d'une minute.",
    ),
    "shame": Direction(
        label="shame",
        display_name="honte",
        text="Je sais. Je n'aurais pas dû te mentir.",
        instruct=(
            "Speak with quiet shame. The voice is lowered and hesitant. "
            "Use restrained emotion, small pauses, and little eye-contact in the voice. "
            "Do not cry."
        ),
        clone_text="Je sais. ;; Je n'aurais pas dû te mentir.",
    ),
    "dry_contempt": Direction(
        label="dry_contempt",
        display_name="mepris discret",
        text="Évidemment. Quelle excellente idée.",
        instruct=(
            "Speak with discreet contempt and dry irony. Keep the delivery controlled "
            "and intelligent. Do not laugh. Let the emphasis make the sarcasm clear."
        ),
        clone_text="Évidemment. ;; Quelle excellente idée.",
    ),
    "nervous_relief": Direction(
        label="nervous_relief",
        display_name="soulagement nerveux",
        text="Mon Dieu... tu es vivant.",
        instruct=(
            "Speak with nervous relief after intense fear. Start tense and breathless. "
            "Gradually release the voice. The final words should become softer and warmer."
        ),
        clone_text="Mon Dieu... ;; tu es vivant.",
    ),
    "fragile_joy": Direction(
        label="fragile_joy",
        display_name="joie fragile",
        text="Tu es là. Je n'arrive pas à y croire.",
        instruct=(
            "Speak with fragile joy. A small smile should be audible, but the voice "
            "is delicate and almost disbelieving. Keep the happiness restrained, "
            "as if the speaker is afraid the moment may disappear."
        ),
        clone_text="Tu es là. ;; Je n'arrive pas à y croire.",
    ),
}

QUICK_LABELS = ["neutral", "restrained_anger", "cold_anger", "masked_worry"]
CONTRAST_LABELS = ["neutral", "explosive_anger", "warm_joy"]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned or "item"


def load_clone_profile(profile_name: str) -> tuple[Path, str]:
    data = json.loads(VOICE_DB.read_text(encoding="utf-8"))
    if profile_name not in data:
        available = ", ".join(sorted(data))
        raise SystemExit(f"Unknown clone profile: {profile_name}\nAvailable: {available}")

    profile = data[profile_name]
    ref_audio = ROOT / profile["ref_audio"]
    ref_text = profile.get("ref_text", "").strip()
    if not ref_audio.exists():
        raise SystemExit(f"Missing neutral ref_audio for {profile_name}: {ref_audio}")
    if not ref_text:
        raise SystemExit(f"Missing ref_text for {profile_name}")
    return ref_audio, ref_text


def load_tts_model(model_source: str | Path):
    return RUNTIME.load_tts(model_source)


def seed_runtime(seed: int) -> None:
    if RUNTIME.backend == "mlx":
        import mlx.core as mx

        mx.random.seed(seed)
        return

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_runtime_cache() -> None:
    if RUNTIME.backend == "mlx":
        import mlx.core as mx

        mx.clear_cache()
    else:
        RUNTIME.release_tts()

def silence(sample_rate: int, seconds: float) -> np.ndarray:
    return np.zeros(max(0, int(sample_rate * seconds)), dtype=np.float32)


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return round(float(len(audio)) / float(sample_rate), 3)


def generate_one_segment(
    model,
    *,
    model_type: str,
    text: str,
    sample_rate_hint: int = 24_000,
    **kwargs,
):
    with tempfile.TemporaryDirectory(prefix="actor_direction_") as temp_dir:
        audio = RUNTIME.generate_audio(
            model=model,
            model_type=model_type,
            temp_dir=temp_dir,
            kwargs={"text": text, **kwargs},
        )
    if audio is None or audio.size == 0:
        raise RuntimeError("No audio returned by the active backend")
    return audio, sample_rate_hint, {}


def generate_with_pause_markers(
    model,
    *,
    model_type: str,
    text: str,
    pause_seconds: float,
    **kwargs,
):
    parts: list[np.ndarray] = []
    sample_rate = 24_000
    stats: dict = {}

    segments = [part.strip() for part in PAUSE_MARKER_RE.split(text) if part.strip()]
    for idx, segment in enumerate(segments):
        audio, sample_rate, stats = generate_one_segment(
            model,
            model_type=model_type,
            text=segment,
            sample_rate_hint=sample_rate,
            **kwargs,
        )
        parts.append(audio)
        if idx < len(segments) - 1:
            parts.append(silence(sample_rate, pause_seconds))

    if not parts:
        raise RuntimeError("No text segment to generate")
    return np.concatenate(parts), sample_rate, stats, len(segments)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sample_rate)


def run_native(
    *,
    out_dir: Path,
    labels: list[str],
    seed: int,
    speakers: list[str],
    temperature: float,
) -> list[dict]:
    print(f"[load] native model: {CUSTOM_MODEL}")
    model = load_tts_model(CUSTOM_MODEL)
    rows: list[dict] = []

    for speaker in speakers:
        for direction in [DIRECTIONS[label] for label in labels]:
            seed_runtime(seed)
            file_name = f"native_{slug(speaker)}_{direction.label}_seed{seed}.wav"
            wav_path = out_dir / file_name
            print(f"[native] {speaker} / {direction.display_name} -> {file_name}")
            started = time.time()
            audio, sample_rate, stats = generate_one_segment(
                model,
                model_type="custom",
                text=direction.text,
                voice=speaker,
                instruct=direction.instruct,
                language="French",
                temperature=temperature,
                top_k=50,
                top_p=1.0,
                max_tokens=1200,
            )
            write_wav(wav_path, audio, sample_rate)
            rows.append(
                {
                    "engine": "native_custom_voice",
                    "voice": speaker,
                    "clone_profile": "",
                    "direction": direction.label,
                    "display_name": direction.display_name,
                    "seed": seed,
                    "text": direction.text,
                    "instruct": direction.instruct,
                    "control_strategy": "CustomVoice speaker + instruct",
                    "file": str(wav_path),
                    "segments": 1,
                    "sample_rate": sample_rate,
                    "output_duration_seconds": duration_seconds(audio, sample_rate),
                    "seconds_wall": round(time.time() - started, 3),
                    **stats,
                }
            )

    del model
    clear_runtime_cache()
    return rows


def run_clone(
    *,
    out_dir: Path,
    labels: list[str],
    seed: int,
    clone_profile: str,
    temperature: float,
    pause_seconds: float,
) -> list[dict]:
    ref_audio, ref_text = load_clone_profile(clone_profile)
    print(f"[load] clone model: {BASE_MODEL}")
    model = load_tts_model(BASE_MODEL)
    rows: list[dict] = []

    for direction in [DIRECTIONS[label] for label in labels]:
        seed_runtime(seed)
        file_name = f"clone_{slug(clone_profile)}_{direction.label}_seed{seed}.wav"
        wav_path = out_dir / file_name
        print(f"[clone] {direction.display_name} -> {file_name}")
        started = time.time()
        audio, sample_rate, stats, segment_count = generate_with_pause_markers(
            model,
            model_type="base",
            text=direction.clone_text,
            pause_seconds=pause_seconds,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            language="French",
            temperature=temperature,
            top_k=50,
            top_p=1.0,
            max_tokens=1200,
        )
        write_wav(wav_path, audio, sample_rate)
        rows.append(
            {
                "engine": "clone_base_neutral_ref",
                "voice": "",
                "clone_profile": clone_profile,
                "direction": direction.label,
                "display_name": direction.display_name,
                "seed": seed,
                "text": direction.clone_text,
                "instruct": "",
                "control_strategy": (
                    "Base voice clone with neutral ref_audio only; "
                    "no artificial emotion sample; punctuation/segmentation only"
                ),
                "neutral_ref_audio": str(ref_audio),
                "file": str(wav_path),
                "segments": segment_count,
                "pause_seconds": pause_seconds,
                "sample_rate": sample_rate,
                "output_duration_seconds": duration_seconds(audio, sample_rate),
                "seconds_wall": round(time.time() - started, 3),
                **stats,
            }
        )

    del model
    clear_runtime_cache()
    return rows


def write_manifest(run_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {
            "native": str(CUSTOM_MODEL),
            "clone": str(BASE_MODEL),
        },
        "backend": RUNTIME.display_name,
        "rule_for_clones": (
            "Clone tests use only the neutral base voice sample from voices_qwen.json. "
            "Artificial emotion variants in voices/ are deliberately ignored."
        ),
        "args": vars(args),
        "items": rows,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rating_fields = [
        "engine",
        "voice",
        "clone_profile",
        "direction",
        "seed",
        "file",
        "emotion_correctness",
        "emotion_intensity",
        "naturalness",
        "voice_identity",
        "prosody",
        "rhythm",
        "pauses",
        "french_pronunciation",
        "notes",
    ]
    with (run_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rating_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in rating_fields})


def parse_labels(value: str) -> list[str]:
    if value == "quick":
        return QUICK_LABELS
    if value == "contrast":
        return CONTRAST_LABELS
    if value == "all":
        return list(DIRECTIONS)
    labels = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [label for label in labels if label not in DIRECTIONS]
    if unknown:
        raise SystemExit(f"Unknown direction label(s): {', '.join(unknown)}")
    return labels


def parse_native_speakers(value: str) -> list[str]:
    if value == "all":
        return NATIVE_SPEAKERS
    speakers = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [speaker for speaker in speakers if speaker not in NATIVE_SPEAKERS]
    if unknown:
        available = ", ".join(NATIVE_SPEAKERS)
        raise SystemExit(f"Unknown native speaker(s): {', '.join(unknown)}\nAvailable: {available}")
    return speakers


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS actor direction test harness.")
    parser.add_argument(
        "--suite",
        default="quick",
        help="'quick', 'contrast', 'all', or comma-separated labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-speaker", default="Ryan", help="'all' or comma-separated native speakers")
    parser.add_argument("--clone-profile", default="Homme Podalydes")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--pause-seconds", type=float, default=0.45)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--clone-only", action="store_true")
    args = parser.parse_args()

    labels = parse_labels(args.suite)
    native_speakers = parse_native_speakers(args.native_speaker)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"[run] {run_dir}")
    print(f"[directions] {', '.join(labels)}")
    if not args.clone_only:
        print(f"[native speakers] {', '.join(native_speakers)}")

    rows: list[dict] = []
    if not args.clone_only:
        rows.extend(
            run_native(
                out_dir=run_dir,
                labels=labels,
                seed=args.seed,
                speakers=native_speakers,
                temperature=args.temperature,
            )
        )
    if not args.native_only:
        rows.extend(
            run_clone(
                out_dir=run_dir,
                labels=labels,
                seed=args.seed,
                clone_profile=args.clone_profile,
                temperature=args.temperature,
                pause_seconds=args.pause_seconds,
            )
        )

    write_manifest(run_dir, rows, args)
    print(f"[done] wrote {len(rows)} wav files")
    print(f"[manifest] {run_dir / 'manifest.json'}")
    print(f"[ratings] {run_dir / 'ratings.csv'}")


if __name__ == "__main__":
    main()
