#!/usr/bin/env python3
"""Standalone Gradio prototype for Stable Audio 3 Small-SFX."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Let Gradio try consecutive ports when the preferred one is already occupied.
SFX_START_PORT = (
    os.getenv("BOTQWEN_SFX_PORT")
    or os.getenv("GRADIO_SERVER_PORT")
    or "7861"
)
os.environ["GRADIO_SERVER_PORT"] = SFX_START_PORT

import gradio as gr

from scene_sfx import (
    OllamaSceneAnalyzer,
    SceneSoundtrackRenderer,
    build_stable_audio_prompt_document,
    list_ollama_models,
    plan_to_rows,
    read_scene_document,
    rows_to_plans,
    split_text_chapters,
    write_stable_audio_prompt_document,
)
from sfx_runtime import StableAudioSFXRuntime, safe_slug


PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME = StableAudioSFXRuntime(PROJECT_DIR)
SCENE_RENDERER = SceneSoundtrackRenderer(PROJECT_DIR, RUNTIME)
TRACK_TYPES = {
    "Bruitage": "sound_effect",
    "Ambiance": "ambience",
}
PLAN_HEADERS = [
    "Chapitre #",
    "Chapitre",
    "Ordre",
    "Type",
    "Duree",
    "Position dans le chapitre (%)",
    "Prompt brut LLM (anglais)",
    "Prompt Stable Audio nettoye (anglais)",
    "Passage narratif",
]


def generate_sfx(
    track_label: str,
    prompt: str,
    duration: float,
    variants: int,
    seed: int,
    steps: int,
    cfg: float,
    negative_prompt: str,
):
    try:
        outputs, records, _ = RUNTIME.generate_batch(
            prompt=prompt,
            track_type=TRACK_TYPES[track_label],
            duration=duration,
            variants=variants,
            seed=seed,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt,
        )
    except Exception as exc:
        return None, None, f"Erreur : {exc}", seed

    seeds = ", ".join(str(record.seed) for record in records)
    status = (
        f"Generation terminee : {len(outputs)} piste(s), {duration:g} s, "
        f"seeds {seeds}. Dossier : {outputs[0].parent}"
    )
    return str(outputs[0]), [str(path) for path in outputs], status, records[0].seed


def _chapters_from_inputs(file_path, scene_text: str):
    if file_path:
        return read_scene_document(str(file_path))
    return split_text_chapters(scene_text or "", "Scene collee")


def load_scene_file(file_path):
    if not file_path:
        return "", "Aucun document charge"
    try:
        chapters = read_scene_document(str(file_path))
    except Exception as exc:
        return "", f"Erreur : {exc}"
    if not chapters:
        return "", "Erreur : le document est vide"
    preview_parts = []
    for chapter in chapters:
        heading = f"{chapter.title}\n" if len(chapters) > 1 else ""
        preview_parts.append(f"{heading}{chapter.text}")
    text = "\n\n".join(preview_parts)
    return text, f"{len(chapters)} chapitre(s), {len(text):,} caracteres"


def refresh_ollama_models():
    models = list_ollama_models()
    if not models:
        return gr.update(choices=["mistral-nemo:latest"], value="mistral-nemo:latest")
    preferred = next(
        (model for model in models if model.lower().startswith("mistral-nemo")),
        models[0],
    )
    return gr.update(choices=models, value=preferred)


def _analyse_inputs(
    file_path,
    scene_text: str,
    model: str,
    max_beats: int,
    default_duration: float,
    progress=None,
):
    chapters = _chapters_from_inputs(file_path, scene_text)
    if not chapters:
        raise ValueError("Chargez un fichier .txt/.docx ou collez une scene.")
    analyzer = OllamaSceneAnalyzer(model)

    def report(current, total, title):
        if progress:
            progress((current - 1, total), desc=f"Analyse : {title}")

    plans = analyzer.analyze_document(
        chapters,
        max_beats=int(max_beats),
        default_duration=float(default_duration),
        progress=report,
    )
    summaries = "\n".join(
        f"Chapitre {plan.chapter.index} - {plan.chapter.title}: {plan.summary}"
        for plan in plans
    )
    rows = plan_to_rows(plans)
    document_name = Path(str(file_path)).stem if file_path else "scene_collee"
    prompt_text = build_stable_audio_prompt_document(plans, document_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_path = write_stable_audio_prompt_document(
        plans,
        document_name,
        PROJECT_DIR
        / "outputs"
        / "sfx_plans"
        / f"{timestamp}_{safe_slug(document_name)}"
        / "stable_audio_prompts.md",
    )
    return chapters, plans, rows, summaries, prompt_text, prompt_path


def analyze_scene_document(
    file_path,
    scene_text: str,
    model: str,
    max_beats: int,
    default_duration: float,
    progress=gr.Progress(),
):
    try:
        chapters, _, rows, summaries, prompt_text, prompt_path = _analyse_inputs(
            file_path,
            scene_text,
            model,
            max_beats,
            default_duration,
            progress,
        )
    except Exception as exc:
        return [], "", f"Erreur : {exc}", "", None
    return (
        rows,
        summaries,
        f"Plan pret : {len(rows)} sequence(s), {len(chapters)} piste(s) finale(s).",
        prompt_text,
        str(prompt_path),
    )


def _render_scene_rows(
    file_path,
    scene_text: str,
    rows,
    voice_paths,
    crossfade: float,
    seed: int,
    steps: int,
    cfg: float,
    progress=None,
):
    chapters = _chapters_from_inputs(file_path, scene_text)
    plans = rows_to_plans(rows, chapters)
    document_name = Path(str(file_path)).stem if file_path else "scene_collee"
    source_text = "\n\n".join(chapter.text for chapter in chapters)

    def report(current, total, title):
        if progress:
            progress((current - 1, total), desc=f"Generation : {title}")

    return SCENE_RENDERER.render_document(
        plans,
        source_text=source_text,
        document_name=document_name,
        crossfade_seconds=float(crossfade),
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        voice_paths=voice_paths or [],
        progress=report,
    )


def generate_scene_document(
    file_path,
    scene_text: str,
    rows,
    voice_paths,
    crossfade: float,
    seed: int,
    steps: int,
    cfg: float,
    progress=gr.Progress(),
):
    try:
        result = _render_scene_rows(
            file_path,
            scene_text,
            rows,
            voice_paths,
            crossfade,
            seed,
            steps,
            cfg,
            progress,
        )
    except Exception as exc:
        return None, None, None, None, "", None, f"Erreur : {exc}"
    prompt_text = result.prompt_document_path.read_text(encoding="utf-8")
    return (
        str(result.chapter_tracks[0]),
        [str(path) for path in result.chapter_tracks],
        [str(path) for path in result.clip_paths],
        str(result.manifest_path),
        prompt_text,
        str(result.prompt_document_path),
        f"Termine : {len(result.chapter_tracks)} piste(s) de chapitre dans {result.run_dir}",
    )


def analyze_and_generate_scene(
    file_path,
    scene_text: str,
    model: str,
    max_beats: int,
    default_duration: float,
    voice_paths,
    crossfade: float,
    seed: int,
    steps: int,
    cfg: float,
    progress=gr.Progress(),
):
    try:
        chapters, _, rows, summaries, _, _ = _analyse_inputs(
            file_path,
            scene_text,
            model,
            max_beats,
            default_duration,
            progress,
        )
        result = _render_scene_rows(
            file_path,
            scene_text,
            rows,
            voice_paths,
            crossfade,
            seed,
            steps,
            cfg,
            progress,
        )
    except Exception as exc:
        return (
            [],
            "",
            f"Erreur : {exc}",
            None,
            None,
            None,
            None,
            "",
            None,
            f"Erreur : {exc}",
        )
    prompt_text = result.prompt_document_path.read_text(encoding="utf-8")
    return (
        rows,
        summaries,
        f"Plan utilise : {len(rows)} sequence(s), {len(chapters)} chapitre(s).",
        str(result.chapter_tracks[0]),
        [str(path) for path in result.chapter_tracks],
        [str(path) for path in result.clip_paths],
        str(result.manifest_path),
        prompt_text,
        str(result.prompt_document_path),
        f"Termine : {len(result.chapter_tracks)} piste(s) dans {result.run_dir}",
    )


_ollama_models = list_ollama_models()
if not _ollama_models:
    _ollama_models = ["mistral-nemo:latest"]
_default_ollama_model = next(
    (model for model in _ollama_models if model.lower().startswith("mistral-nemo")),
    _ollama_models[0],
)


with gr.Blocks(title="Studio SFX") as demo:
    gr.Markdown("# Studio SFX")
    engine_status = gr.Textbox(
        value=RUNTIME.status(),
        label="Moteur",
        interactive=False,
    )

    with gr.Tabs():
        with gr.Tab("Son unique"):
            with gr.Row():
                with gr.Column(scale=3):
                    track_type = gr.Radio(
                        choices=list(TRACK_TYPES),
                        value="Bruitage",
                        label="Type de piste",
                    )
                    prompt = gr.Textbox(
                        label="Description",
                        lines=4,
                        placeholder="Heavy wooden door slamming in a stone corridor, close perspective",
                    )
                    with gr.Row():
                        duration = gr.Slider(1, 120, value=8, step=1, label="Duree (secondes)")
                        variants = gr.Slider(1, 4, value=1, step=1, label="Variantes")
                    generate_button = gr.Button("Generer", variant="primary")

                with gr.Column(scale=2):
                    preview = gr.Audio(label="Ecoute", type="filepath")
                    files = gr.File(label="Pistes WAV", file_count="multiple")
                    status = gr.Textbox(label="Etat", interactive=False, lines=3)

            with gr.Accordion("Reglages avances", open=False):
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatoire)")
                    steps = gr.Slider(1, 32, value=8, step=1, label="Etapes")
                    cfg = gr.Slider(0, 10, value=1, step=0.1, label="CFG")
                negative_prompt = gr.Textbox(
                    value="",
                    label="Description negative",
                    placeholder="speech, voice, vocals, music",
                )

            generate_button.click(
                fn=generate_sfx,
                inputs=[track_type, prompt, duration, variants, seed, steps, cfg, negative_prompt],
                outputs=[preview, files, status, seed],
            )

        with gr.Tab("Scene de roman"):
            with gr.Row():
                with gr.Column(scale=3):
                    scene_file = gr.File(
                        label="Document .txt ou .docx",
                        file_types=[".txt", ".docx"],
                        type="filepath",
                    )
                    scene_text = gr.Textbox(
                        label="Texte narratif",
                        lines=14,
                        placeholder="Collez une scene ou chargez un document...",
                    )
                    scene_file_status = gr.Textbox(
                        label="Document",
                        value="Aucun document charge",
                        interactive=False,
                    )
                    voice_chapter_files = gr.File(
                        label="Pistes voix WAV ou MP3 par chapitre, dans le meme ordre (facultatif)",
                        file_types=[".wav", ".mp3"],
                        file_count="multiple",
                        type="filepath",
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        ollama_model = gr.Dropdown(
                            choices=_ollama_models,
                            value=_default_ollama_model,
                            label="Modele Ollama",
                            scale=4,
                        )
                        refresh_models = gr.Button("Actualiser", scale=1)
                    max_beats = gr.Slider(
                        1,
                        120,
                        value=12,
                        step=1,
                        label="Sequences maximum par chapitre",
                    )
                    default_duration = gr.Slider(
                        2,
                        120,
                        value=8,
                        step=1,
                        label="Duree indicative par sequence",
                    )
                    analyze_button = gr.Button("Analyser seulement")
                    auto_button = gr.Button("Analyser et generer", variant="primary")

            analysis_status = gr.Textbox(label="Analyse", interactive=False)
            chapter_summaries = gr.Textbox(
                label="Intentions sonores par chapitre",
                interactive=False,
                lines=4,
            )
            plan_table = gr.Dataframe(
                headers=PLAN_HEADERS,
                datatype=[
                    "number",
                    "str",
                    "number",
                    "str",
                    "number",
                    "number",
                    "str",
                    "str",
                    "str",
                ],
                type="array",
                interactive=True,
                wrap=True,
                label="Plan sonore editable",
            )

            with gr.Accordion("Document transmis a Stable Audio", open=True):
                prompt_document_preview = gr.Textbox(
                    label="Prompts bruts et prompts exacts",
                    interactive=False,
                    lines=16,
                )
                prompt_document_file = gr.File(
                    label="Telecharger stable_audio_prompts.md",
                )

            with gr.Accordion("Montage et generation", open=True):
                with gr.Row():
                    scene_crossfade = gr.Slider(
                        0,
                        2,
                        value=0.25,
                        step=0.05,
                        label="Fondu entre sequences (secondes)",
                    )
                    scene_seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed de depart (-1 = aleatoire)",
                    )
                    scene_steps = gr.Slider(1, 32, value=8, step=1, label="Etapes")
                    scene_cfg = gr.Slider(0, 10, value=1, step=0.1, label="CFG")
                generate_scene_button = gr.Button(
                    "Generer le plan edite",
                    variant="primary",
                )

            with gr.Row():
                with gr.Column(scale=2):
                    scene_preview = gr.Audio(
                        label="Ecoute du premier chapitre",
                        type="filepath",
                    )
                    chapter_files = gr.File(
                        label="Pistes finales par chapitre",
                        file_count="multiple",
                    )
                    scene_manifest = gr.File(label="Manifeste JSON")
                with gr.Column(scale=3):
                    generation_status = gr.Textbox(
                        label="Generation",
                        interactive=False,
                        lines=4,
                    )
                    with gr.Accordion("Clips intermediaires", open=False):
                        scene_clips = gr.File(
                            label="Sequences WAV",
                            file_count="multiple",
                        )

            scene_file.change(
                fn=load_scene_file,
                inputs=[scene_file],
                outputs=[scene_text, scene_file_status],
            )
            refresh_models.click(
                fn=refresh_ollama_models,
                outputs=[ollama_model],
            )
            analyze_button.click(
                fn=analyze_scene_document,
                inputs=[
                    scene_file,
                    scene_text,
                    ollama_model,
                    max_beats,
                    default_duration,
                ],
                outputs=[
                    plan_table,
                    chapter_summaries,
                    analysis_status,
                    prompt_document_preview,
                    prompt_document_file,
                ],
            )
            generate_scene_button.click(
                fn=generate_scene_document,
                inputs=[
                    scene_file,
                    scene_text,
                    plan_table,
                    voice_chapter_files,
                    scene_crossfade,
                    scene_seed,
                    scene_steps,
                    scene_cfg,
                ],
                outputs=[
                    scene_preview,
                    chapter_files,
                    scene_clips,
                    scene_manifest,
                    prompt_document_preview,
                    prompt_document_file,
                    generation_status,
                ],
            )
            auto_button.click(
                fn=analyze_and_generate_scene,
                inputs=[
                    scene_file,
                    scene_text,
                    ollama_model,
                    max_beats,
                    default_duration,
                    voice_chapter_files,
                    scene_crossfade,
                    scene_seed,
                    scene_steps,
                    scene_cfg,
                ],
                outputs=[
                    plan_table,
                    chapter_summaries,
                    analysis_status,
                    scene_preview,
                    chapter_files,
                    scene_clips,
                    scene_manifest,
                    prompt_document_preview,
                    prompt_document_file,
                    generation_status,
                ],
            )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    print(
        "Studio SFX : recherche d'un port disponible a partir de "
        f"http://127.0.0.1:{SFX_START_PORT}"
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        inbrowser=True,
    )
