#!/usr/bin/env python3
"""Chapter-aware sound design for novels using Ollama and Stable Audio."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import requests
import soundfile as sf

from sfx_runtime import SFXGeneration, StableAudioSFXRuntime, enrich_prompt, safe_slug


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
MIN_SCENE_CLIP_DURATION = 2.0
MAX_SCENE_CLIP_DURATION = 120.0
SCENE_NEGATIVE_PROMPT = "speech, dialogue, narration, voice, vocals, music"
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapitre|chapter|partie|livre|prologue|epilogue|épilogue)\b.*$",
    re.IGNORECASE,
)
DIRECT_SPEECH_RE = re.compile(r"«[^»]*»|(?<!\w)\"[^\"]+\"")
DIALOGUE_LINE_RE = re.compile(r"^\s*\\?[-–—]\s+")
VOCAL_PROMPT_RE = re.compile(
    r"\b(?:speech|speak|speaks|speaking|spoken|voice|voices|dialogue|conversation|"
    r"talk|talks|talking|whisper|whispers|whispering|narration|narrator|vocals?|"
    r"grunt|grunts|grunting|groan|groans|groaning|sigh|sighs|sighing|laugh|laughs|"
    r"laughing|cry|cries|crying|cough|coughs|coughing|sneeze|sneezes|sneezing|"
    r"breath|breaths|breathing|gasp|gasps|gasping)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentChapter:
    index: int
    title: str
    text: str


@dataclass(frozen=True)
class SceneBeat:
    order: int
    label: str
    track_type: str
    llm_prompt_en: str
    prompt_en: str
    duration: float
    position_ratio: float
    source_excerpt: str


@dataclass(frozen=True)
class ChapterSoundPlan:
    chapter: DocumentChapter
    summary: str
    beats: tuple[SceneBeat, ...]


@dataclass(frozen=True)
class StitchResult:
    sample_rate: int
    channels: int
    duration: float
    start_times: tuple[float, ...]


@dataclass(frozen=True)
class DocumentRenderResult:
    run_dir: Path
    chapter_tracks: tuple[Path, ...]
    clip_paths: tuple[Path, ...]
    manifest_path: Path
    prompt_document_path: Path


def strip_dialogue(text: str) -> str:
    """Remove direct speech while preserving surrounding narrative action."""
    kept: list[str] = []
    for line in text.splitlines():
        if DIALOGUE_LINE_RE.match(line):
            continue
        cleaned = DIRECT_SPEECH_RE.sub(" ", line)
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
        if cleaned or ";;" in line:
            kept.append(cleaned)
    return "\n".join(kept).strip()


def split_text_chapters(text: str, default_title: str = "Scene") -> list[DocumentChapter]:
    """Split plain text only on explicit chapter-like headings."""
    text = text.strip()
    if not text:
        return []

    sections: list[tuple[str, list[str]]] = []
    title = default_title
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and CHAPTER_HEADING_RE.match(line):
            if any(item.strip() for item in lines):
                sections.append((title, lines))
            title = line
            lines = []
        else:
            lines.append(raw_line)
    if any(item.strip() for item in lines):
        sections.append((title, lines))

    return [
        DocumentChapter(index=index, title=section_title, text="\n".join(body).strip())
        for index, (section_title, body) in enumerate(sections, start=1)
    ]


def read_scene_document(path: str | Path) -> list[DocumentChapter]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".txt":
        text = source.read_text(encoding="utf-8", errors="replace")
        return split_text_chapters(text, source.stem)
    if suffix != ".docx":
        raise ValueError("Seuls les fichiers .txt et .docx sont acceptes.")

    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx est requis pour lire un fichier Word.") from exc

    document = Document(str(source))
    sections: list[tuple[str, list[str]]] = []
    title = source.stem
    paragraphs: list[str] = []
    for paragraph in document.paragraphs:
        value = paragraph.text.strip()
        if not value:
            continue
        style_name = (paragraph.style.name or "").lower()
        is_heading = style_name.startswith(("heading", "titre")) or bool(
            CHAPTER_HEADING_RE.match(value)
        )
        if is_heading:
            if paragraphs:
                sections.append((title, paragraphs))
            title = value
            paragraphs = []
        else:
            paragraphs.append(value)
    if paragraphs:
        sections.append((title, paragraphs))

    return [
        DocumentChapter(index=index, title=section_title, text="\n".join(body))
        for index, (section_title, body) in enumerate(sections, start=1)
    ]


def list_ollama_models(timeout: float = 3.0) -> list[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", []) if item.get("name")]
    except Exception:
        return []


def _extract_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("Ollama n'a pas renvoye de plan JSON.")
        result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(result, dict):
        raise ValueError("Le plan Ollama doit etre un objet JSON.")
    return result


def _normalise_track_type(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key in {"ambience", "ambiance", "atmosphere", "room tone"}:
        return "ambience"
    return "sound_effect"


def sanitize_sound_prompt(value: str) -> str:
    """Remove clauses that ask the audio model to synthesize vocal content."""
    clauses = re.split(r"(?<=[.!?])\s+|[,;]", value.strip())
    kept = [clause.strip() for clause in clauses if clause.strip() and not VOCAL_PROMPT_RE.search(clause)]
    base = ", ".join(kept).rstrip(" .")
    if not base:
        base = "Subtle environmental room tone"
    return f"{base}. No speech, no dialogue, no voices, no vocals, no narration, no music."


def estimate_excerpt_position(
    chapter_text: str,
    source_excerpt: str,
    fallback: float,
) -> float:
    narration = re.sub(r"\s+", " ", strip_dialogue(chapter_text)).strip().casefold()
    excerpt = re.sub(r"\s+", " ", source_excerpt).strip().casefold()
    if excerpt and narration:
        offset = narration.find(excerpt)
        if offset >= 0:
            return min(1.0, max(0.0, offset / max(1, len(narration))))
    return min(1.0, max(0.0, fallback))


def build_stable_audio_prompt_document(
    plans: Iterable[ChapterSoundPlan],
    document_name: str,
) -> str:
    """Build a readable audit of LLM output and exact Stable Audio inputs."""
    lines = [
        "# Document transmis a Stable Audio",
        "",
        f"Document source : {document_name}",
        "",
        (
            "Ce rapport compare le prompt brut produit par le LLM au prompt positif "
            "exact transmis a Stable Audio apres filtrage des contenus vocaux."
        ),
    ]
    for plan in plans:
        lines.extend(["", f"## Chapitre {plan.chapter.index} - {plan.chapter.title}"])
        for beat in plan.beats:
            effective_prompt = enrich_prompt(beat.prompt_en, beat.track_type)
            lines.extend(
                [
                    "",
                    f"### Sequence {beat.order} - {beat.label}",
                    "",
                    f"- Type : {beat.track_type}",
                    f"- Duree demandee : {beat.duration:g} s",
                    f"- Position dans le chapitre : {beat.position_ratio * 100:.1f} %",
                    f"- Extrait source : {beat.source_excerpt or '(non fourni)'}",
                    "",
                    "Prompt brut produit par le LLM :",
                    "",
                    beat.llm_prompt_en,
                    "",
                    "Prompt positif exact transmis a Stable Audio :",
                    "",
                    effective_prompt,
                    "",
                    "Negative prompt exact transmis a Stable Audio :",
                    "",
                    SCENE_NEGATIVE_PROMPT,
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_stable_audio_prompt_document(
    plans: Iterable[ChapterSoundPlan],
    document_name: str,
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_stable_audio_prompt_document(plans, document_name),
        encoding="utf-8",
    )
    return target


def normalise_chapter_plan(
    chapter: DocumentChapter,
    payload: dict[str, Any],
    *,
    max_beats: int,
    default_duration: float,
) -> ChapterSoundPlan:
    raw_beats = payload.get("beats") or payload.get("sequences") or []
    if not isinstance(raw_beats, list):
        raise ValueError("Le plan sonore ne contient pas de liste de sequences.")

    beats: list[SceneBeat] = []
    selected_beats = raw_beats[:max_beats]
    for item in selected_beats:
        if not isinstance(item, dict):
            continue
        llm_prompt = str(
            item.get("prompt_en") or item.get("prompt") or item.get("sound_prompt") or ""
        ).strip()
        if not llm_prompt:
            continue
        prompt = sanitize_sound_prompt(llm_prompt)
        try:
            duration = float(item.get("duration_seconds", default_duration))
        except (TypeError, ValueError):
            duration = float(default_duration)
        duration = min(
            MAX_SCENE_CLIP_DURATION,
            max(MIN_SCENE_CLIP_DURATION, duration),
        )
        order = len(beats) + 1
        try:
            raw_position = float(item.get("position_percent", item.get("position", -1)))
        except (TypeError, ValueError):
            raw_position = -1
        if raw_position < 0:
            denominator = max(1, len(selected_beats) - 1)
            position_ratio = min(0.95, (order - 1) / denominator)
        else:
            position_ratio = raw_position / 100 if raw_position > 1 else raw_position
            position_ratio = min(1.0, max(0.0, position_ratio))
        source_excerpt = str(item.get("source_excerpt") or "").strip()[:300]
        position_ratio = estimate_excerpt_position(
            chapter.text,
            source_excerpt,
            position_ratio,
        )
        beats.append(
            SceneBeat(
                order=order,
                label=str(item.get("label") or f"Plan {order}").strip(),
                track_type=_normalise_track_type(item.get("track_type") or item.get("type")),
                llm_prompt_en=llm_prompt,
                prompt_en=prompt,
                duration=duration,
                position_ratio=position_ratio,
                source_excerpt=source_excerpt,
            )
        )
    if not beats:
        raise ValueError(f"Aucun plan sonore exploitable pour {chapter.title}.")
    summary = str(payload.get("summary") or payload.get("scene_summary") or chapter.title).strip()
    return ChapterSoundPlan(chapter=chapter, summary=summary, beats=tuple(beats))


class OllamaSceneAnalyzer:
    def __init__(
        self,
        model: str,
        *,
        chat_url: str = OLLAMA_CHAT_URL,
        requester: Callable[..., Any] = requests.post,
    ):
        self.model = model.strip()
        self.chat_url = chat_url
        self.requester = requester

    def analyze_chapter(
        self,
        chapter: DocumentChapter,
        *,
        max_beats: int = 8,
        default_duration: float = 8.0,
    ) -> ChapterSoundPlan:
        narration = strip_dialogue(chapter.text)
        if not narration:
            raise ValueError(f"{chapter.title} ne contient aucune narration exploitable.")
        if not self.model:
            raise ValueError("Selectionnez un modele Ollama.")

        system_prompt = (
            "You are a film sound editor adapting French literary narration into a "
            "chronological sound-only storyboard. Ignore every spoken line, dialogue, "
            "narrator voice, thought and named character voice. Never generate speech, "
            "whispers, vocal reactions, music or narration. Keep only audible places, "
            "objects, weather, movement and concrete actions. A marker ;; is a strong "
            "possible beat boundary, never a mandatory sound. Return strict JSON only."
        )
        user_prompt = f"""
Create at most {int(max_beats)} chronological sound beats for this chapter.
Each beat must be a coherent standalone audio scene lasting roughly
{float(default_duration):g} seconds. Omit silent or purely psychological details.
Use as many beats as the audible action requires, up to the requested maximum.
Do not merge distinct places, ambience changes or concrete sound events merely
to keep the plan short.
Use track_type "ambience" for continuous places and "sound_effect" for concrete events.
Write prompt_en in precise English suitable for Stable Audio Small-SFX.
Descriptions must explicitly exclude speech, voices, vocals and music.
Set position_percent from 0 to 100 to locate the event in the chapter chronology.

Return exactly this shape:
{{
  "summary": "short French summary of the sound design",
  "beats": [
    {{
      "label": "short French label",
      "track_type": "ambience or sound_effect",
      "duration_seconds": {float(default_duration):g},
      "position_percent": 0,
      "prompt_en": "English sound description",
      "source_excerpt": "short French narrative excerpt, never dialogue"
    }}
  ]
}}

Chapter: {chapter.title}
<narration_without_dialogue>
{narration}
</narration_without_dialogue>
""".strip()
        try:
            response = self.requester(
                self.chat_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama est inaccessible. Lancez Ollama puis actualisez les modeles."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Le modele Ollama {self.model} n'a pas pu produire le plan JSON : {exc}"
            ) from exc
        payload = _extract_json_object(content)
        return normalise_chapter_plan(
            chapter,
            payload,
            max_beats=max_beats,
            default_duration=default_duration,
        )

    def analyze_document(
        self,
        chapters: Iterable[DocumentChapter],
        *,
        max_beats: int = 8,
        default_duration: float = 8.0,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[ChapterSoundPlan]:
        items = list(chapters)
        plans: list[ChapterSoundPlan] = []
        for position, chapter in enumerate(items, start=1):
            if progress:
                progress(position, len(items), chapter.title)
            plans.append(
                self.analyze_chapter(
                    chapter,
                    max_beats=max_beats,
                    default_duration=default_duration,
                )
            )
        return plans


def plan_to_rows(plans: Iterable[ChapterSoundPlan]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for plan in plans:
        for beat in plan.beats:
            rows.append(
                [
                    plan.chapter.index,
                    plan.chapter.title,
                    beat.order,
                    "Ambiance" if beat.track_type == "ambience" else "Bruitage",
                    beat.duration,
                    round(beat.position_ratio * 100, 1),
                    beat.llm_prompt_en,
                    beat.prompt_en,
                    beat.source_excerpt,
                ]
            )
    return rows


def rows_to_plans(rows: Any, chapters: Iterable[DocumentChapter]) -> list[ChapterSoundPlan]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    rows = rows or []
    chapter_map = {chapter.index: chapter for chapter in chapters}
    grouped: dict[int, list[SceneBeat]] = {}
    titles: dict[int, str] = {}
    for row in rows:
        if not row or len(row) < 9:
            continue
        try:
            chapter_index = int(float(row[0]))
            order = int(float(row[2]))
            duration = min(
                MAX_SCENE_CLIP_DURATION,
                max(MIN_SCENE_CLIP_DURATION, float(row[4])),
            )
            position_ratio = min(1.0, max(0.0, float(row[5]) / 100))
        except (TypeError, ValueError):
            continue
        llm_prompt = str(row[6] or "").strip()
        prompt = str(row[7] or "").strip()
        if not prompt:
            continue
        prompt = sanitize_sound_prompt(prompt)
        title = str(row[1] or f"Chapitre {chapter_index}").strip()
        titles[chapter_index] = title
        grouped.setdefault(chapter_index, []).append(
            SceneBeat(
                order=order,
                label=f"Plan {order}",
                track_type=_normalise_track_type(row[3]),
                duration=duration,
                position_ratio=position_ratio,
                llm_prompt_en=llm_prompt or prompt,
                prompt_en=prompt,
                source_excerpt=str(row[8] or "").strip()[:300],
            )
        )

    plans: list[ChapterSoundPlan] = []
    for chapter_index in sorted(grouped):
        chapter = chapter_map.get(
            chapter_index,
            DocumentChapter(chapter_index, titles[chapter_index], ""),
        )
        ordered = sorted(grouped[chapter_index], key=lambda beat: beat.order)
        beats = tuple(
            SceneBeat(
                order=index,
                label=beat.label,
                track_type=beat.track_type,
                llm_prompt_en=beat.llm_prompt_en,
                prompt_en=beat.prompt_en,
                duration=beat.duration,
                position_ratio=beat.position_ratio,
                source_excerpt=beat.source_excerpt,
            )
            for index, beat in enumerate(ordered, start=1)
        )
        plans.append(ChapterSoundPlan(chapter, chapter.title, beats))
    if not plans:
        raise ValueError("Le plan sonore est vide.")
    return plans


def stitch_wav_files(
    paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    crossfade_seconds: float = 0.25,
) -> StitchResult:
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("Aucun clip a assembler.")
    if not 0 <= float(crossfade_seconds) <= 2:
        raise ValueError("Le fondu doit etre compris entre 0 et 2 secondes.")

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    for path in files:
        audio, current_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = int(current_rate)
        elif current_rate != sample_rate:
            raise ValueError("Tous les clips doivent avoir la meme frequence.")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        arrays.append(audio)

    assert sample_rate is not None
    combined = arrays[0]
    starts = [0.0]
    requested_overlap = int(round(float(crossfade_seconds) * sample_rate))
    for audio in arrays[1:]:
        overlap = min(requested_overlap, len(combined), len(audio))
        starts.append((len(combined) - overlap) / sample_rate)
        if overlap:
            fade_in = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)[:, None]
            mixed = combined[-overlap:] * (1.0 - fade_in) + audio[:overlap] * fade_in
            combined = np.concatenate((combined[:-overlap], mixed, audio[overlap:]), axis=0)
        else:
            combined = np.concatenate((combined, audio), axis=0)

    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.999:
        combined *= 0.999 / peak
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), combined, sample_rate, subtype="PCM_16")
    return StitchResult(
        sample_rate=sample_rate,
        channels=combined.shape[1],
        duration=len(combined) / sample_rate,
        start_times=tuple(starts),
    )


def mix_wav_files_to_voice_timeline(
    paths: Iterable[str | Path],
    positions: Iterable[float],
    voice_path: str | Path,
    output_path: str | Path,
) -> StitchResult:
    """Place clips on a track whose duration exactly matches a voice reference."""
    files = [Path(path) for path in paths]
    position_values = [min(1.0, max(0.0, float(value))) for value in positions]
    if not files or len(files) != len(position_values):
        raise ValueError("Les clips et leurs positions ne correspondent pas.")

    voice_info = sf.info(str(voice_path))
    voice_duration = float(voice_info.frames) / float(voice_info.samplerate)
    first_info = sf.info(str(files[0]))
    sample_rate = int(first_info.samplerate)
    frame_count = max(1, int(round(voice_duration * sample_rate)))
    combined = np.zeros((frame_count, 2), dtype=np.float32)
    starts: list[float] = []

    for path, position in zip(files, position_values):
        audio, current_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if current_rate != sample_rate:
            raise ValueError("Tous les clips doivent avoir la meme frequence.")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        desired_start = int(round(position * frame_count))
        start = min(desired_start, max(0, frame_count - len(audio)))
        end = min(frame_count, start + len(audio))
        clip = audio[: end - start].copy()
        fade_frames = min(int(0.02 * sample_rate), len(clip) // 2)
        if fade_frames:
            fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, None]
            clip[:fade_frames] *= fade
            clip[-fade_frames:] *= fade[::-1]
        combined[start:end] += clip
        starts.append(start / sample_rate)

    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.999:
        combined *= 0.999 / peak
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), combined, sample_rate, subtype="PCM_16")
    return StitchResult(
        sample_rate=sample_rate,
        channels=2,
        duration=frame_count / sample_rate,
        start_times=tuple(starts),
    )


class SceneSoundtrackRenderer:
    def __init__(self, project_dir: str | Path, runtime: StableAudioSFXRuntime):
        self.project_dir = Path(project_dir).resolve()
        self.runtime = runtime

    def render_document(
        self,
        plans: Iterable[ChapterSoundPlan],
        *,
        source_text: str,
        document_name: str,
        crossfade_seconds: float = 0.25,
        seed: int = -1,
        steps: int = 8,
        cfg: float = 1.0,
        voice_paths: Iterable[str | Path] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> DocumentRenderResult:
        chapter_plans = list(plans)
        if not chapter_plans:
            raise ValueError("Le document ne contient aucun chapitre a generer.")
        voice_files = [Path(path) for path in (voice_paths or [])]
        if voice_files and len(voice_files) != len(chapter_plans):
            raise ValueError(
                "Le nombre de pistes voix doit correspondre au nombre de chapitres."
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.project_dir / "outputs" / "sfx_scenes" / (
            f"{timestamp}_{safe_slug(document_name)}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "source.txt").write_text(source_text, encoding="utf-8")
        prompt_document_path = write_stable_audio_prompt_document(
            chapter_plans,
            document_name,
            run_dir / "stable_audio_prompts.md",
        )

        total_beats = sum(len(plan.beats) for plan in chapter_plans)
        base_seed = int(seed) if int(seed) >= 0 else random.SystemRandom().randrange(0, 2**31)
        generated_count = 0
        all_records: list[dict[str, Any]] = []
        chapter_tracks: list[Path] = []
        clip_paths: list[Path] = []

        for chapter_position, plan in enumerate(chapter_plans):
            chapter_slug = safe_slug(plan.chapter.title)
            chapter_dir = run_dir / f"chapter_{plan.chapter.index:03d}_{chapter_slug}"
            clips_dir = chapter_dir / "clips"
            chapter_records: list[SFXGeneration] = []
            chapter_clips: list[Path] = []
            for beat in plan.beats:
                generated_count += 1
                if progress:
                    progress(generated_count, total_beats, f"{plan.chapter.title} - {beat.label}")
                current_seed = base_seed + generated_count - 1
                output = clips_dir / (
                    f"{beat.order:02d}_{safe_slug(beat.label)}_seed-{current_seed}.wav"
                )
                record, _ = self.runtime.generate_file(
                    prompt=beat.prompt_en,
                    track_type=beat.track_type,
                    output_path=output,
                    duration=beat.duration,
                    seed=current_seed,
                    steps=steps,
                    cfg=cfg,
                    negative_prompt=SCENE_NEGATIVE_PROMPT,
                )
                chapter_records.append(record)
                chapter_clips.append(output)
                clip_paths.append(output)

            final_path = chapter_dir / f"chapter_{plan.chapter.index:03d}_{chapter_slug}.wav"
            if voice_files:
                stitch = mix_wav_files_to_voice_timeline(
                    chapter_clips,
                    [beat.position_ratio for beat in plan.beats],
                    voice_files[chapter_position],
                    final_path,
                )
                sync_mode = "voice_timeline"
                voice_reference = str(voice_files[chapter_position].resolve())
            else:
                stitch = stitch_wav_files(
                    chapter_clips,
                    final_path,
                    crossfade_seconds=crossfade_seconds,
                )
                sync_mode = "sequential"
                voice_reference = None
            chapter_tracks.append(final_path)
            all_records.append(
                {
                    "chapter_index": plan.chapter.index,
                    "chapter_title": plan.chapter.title,
                    "summary": plan.summary,
                    "final_track": str(final_path),
                    "duration": stitch.duration,
                    "sample_rate": stitch.sample_rate,
                    "channels": stitch.channels,
                    "sync_mode": sync_mode,
                    "voice_reference": voice_reference,
                    "beats": [
                        {
                            "plan": asdict(beat),
                            "generation": asdict(record),
                            "start_time": stitch.start_times[index],
                        }
                        for index, (beat, record) in enumerate(
                            zip(plan.beats, chapter_records)
                        )
                    ],
                }
            )

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "document_name": document_name,
            "engine": self.runtime.display_name,
            "crossfade_seconds": float(crossfade_seconds),
            "chapters": all_records,
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return DocumentRenderResult(
            run_dir=run_dir,
            chapter_tracks=tuple(chapter_tracks),
            clip_paths=tuple(clip_paths),
            manifest_path=manifest_path,
            prompt_document_path=prompt_document_path,
        )
