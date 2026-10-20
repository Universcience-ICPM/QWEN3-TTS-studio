#!/usr/bin/env python3
"""Isolated Stable Audio 3 Small-SFX subprocess adapter."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


SFX_BACKENDS = {"mlx", "tflite"}


def detect_sfx_backend(
    requested: str | None = None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    choice = (requested or os.getenv("BOTQWEN_SFX_BACKEND", "auto")).strip().lower()
    if choice != "auto":
        if choice not in SFX_BACKENDS:
            allowed = ", ".join(sorted(SFX_BACKENDS | {"auto"}))
            raise ValueError(f"BOTQWEN_SFX_BACKEND={choice!r} invalide ({allowed})")
        return choice

    host_system = (system or platform.system()).lower()
    host_machine = (machine or platform.machine()).lower()
    if host_system == "darwin" and host_machine in {"arm64", "aarch64"}:
        return "mlx"
    if host_system == "windows":
        return "tflite"
    return "tflite"


def safe_slug(value: str, max_length: int = 48) -> str:
    ascii_value = value.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return (slug or "audio")[:max_length].rstrip("-")


def enrich_prompt(prompt: str, track_type: str) -> str:
    clean = prompt.strip().rstrip(".")
    if not clean:
        raise ValueError("La description du son est vide.")
    suffixes = {
        "sound_effect": "Isolated sound effect, clean recording, no speech, no voice, no music.",
        "ambience": "Continuous environmental ambience, no speech, no voice, no music.",
    }
    if track_type not in suffixes:
        raise ValueError(f"Type de piste inconnu : {track_type}")
    return f"{clean}. {suffixes[track_type]}"


@dataclass(frozen=True)
class SFXGeneration:
    track_type: str
    prompt: str
    effective_prompt: str
    negative_prompt: str
    duration: float
    steps: int
    cfg: float
    seed: int
    backend: str
    output_path: str


class StableAudioSFXRuntime:
    def __init__(
        self,
        project_dir: str | Path,
        backend: str | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.backend = detect_sfx_backend(backend)
        self.runner = runner
        configured = os.getenv("BOTQWEN_SFX_ENGINE")
        if configured:
            self.engine_dir = Path(configured).expanduser().resolve()
        else:
            optimized = "mlx" if self.backend == "mlx" else "tflite"
            self.engine_dir = (
                self.project_dir / ".stable-audio-3" / "optimized" / optimized
            )

    @property
    def launcher(self) -> Path:
        return self.engine_dir / ("sa3.bat" if self.backend == "tflite" and os.name == "nt" else "sa3")

    @property
    def display_name(self) -> str:
        return "Stable Audio 3 Small-SFX / MLX" if self.backend == "mlx" else "Stable Audio 3 Small-SFX / TFLite CPU"

    def is_installed(self) -> bool:
        return self.launcher.is_file()

    def installation_hint(self) -> str:
        script = "install_sfx_macos.sh" if self.backend == "mlx" else "install_sfx_windows.ps1"
        prefix = "./" if self.backend == "mlx" else ".\\"
        return f"Moteur absent. Lancez {prefix}{script}."

    def status(self) -> str:
        if self.is_installed():
            return f"Pret : {self.display_name}"
        return self.installation_hint()

    def _command(self, generation: SFXGeneration) -> list[str]:
        args = [
            "--prompt",
            generation.effective_prompt,
            "--dit",
            "sm-sfx",
            "--decoder",
            "same-s",
            "--seconds",
            f"{generation.duration:g}",
            "--steps",
            str(generation.steps),
            "--seed",
            str(generation.seed),
            "--cfg",
            f"{generation.cfg:g}",
            "--out",
            generation.output_path,
        ]
        if generation.negative_prompt.strip():
            args.extend(["--negative-prompt", generation.negative_prompt.strip()])

        if os.name == "nt" and self.launcher.suffix.lower() == ".bat":
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            command_line = subprocess.list2cmdline([str(self.launcher), *args])
            return [comspec, "/d", "/s", "/c", command_line]
        return [str(self.launcher), *args]

    @staticmethod
    def _validate_parameters(
        *,
        duration: float,
        variants: int = 1,
        steps: int,
        cfg: float,
    ) -> None:
        if not 1 <= float(duration) <= 120:
            raise ValueError("La duree doit etre comprise entre 1 et 120 secondes.")
        if not 1 <= int(variants) <= 4:
            raise ValueError("Le nombre de variantes doit etre compris entre 1 et 4.")
        if not 1 <= int(steps) <= 32:
            raise ValueError("Le nombre d'etapes doit etre compris entre 1 et 32.")
        if not 0 <= float(cfg) <= 10:
            raise ValueError("Le CFG doit etre compris entre 0 et 10.")

    def generate_file(
        self,
        *,
        prompt: str,
        track_type: str,
        output_path: str | Path,
        duration: float,
        seed: int,
        steps: int = 8,
        cfg: float = 1.0,
        negative_prompt: str = "",
    ) -> tuple[SFXGeneration, str]:
        """Generate one clip at a caller-provided path."""
        if not self.is_installed():
            raise RuntimeError(self.installation_hint())
        self._validate_parameters(
            duration=duration,
            steps=steps,
            cfg=cfg,
        )
        if int(seed) < 0:
            raise ValueError("La seed d'un fichier individuel doit etre positive.")

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        record = SFXGeneration(
            track_type=track_type,
            prompt=prompt.strip(),
            effective_prompt=enrich_prompt(prompt, track_type),
            negative_prompt=negative_prompt.strip(),
            duration=float(duration),
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            backend=self.backend,
            output_path=str(target),
        )
        completed = self.runner(
            self._command(record),
            cwd=str(self.engine_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        combined_log = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        if completed.returncode != 0:
            tail = "\n".join(combined_log.splitlines()[-12:])
            raise RuntimeError(
                f"Stable Audio a echoue (code {completed.returncode}).\n{tail}"
            )
        if not target.is_file():
            raise RuntimeError(
                f"Stable Audio n'a pas cree le fichier attendu : {target}"
            )
        return record, combined_log

    def generate_batch(
        self,
        *,
        prompt: str,
        track_type: str,
        duration: float,
        variants: int = 1,
        seed: int = -1,
        steps: int = 8,
        cfg: float = 1.0,
        negative_prompt: str = "",
    ) -> tuple[list[Path], list[SFXGeneration], str]:
        if not self.is_installed():
            raise RuntimeError(self.installation_hint())
        self._validate_parameters(
            duration=duration,
            variants=variants,
            steps=steps,
            cfg=cfg,
        )

        base_seed = int(seed) if int(seed) >= 0 else random.SystemRandom().randrange(0, 2**31)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.project_dir / "outputs" / "sfx" / f"{timestamp}_{safe_slug(prompt)}"
        run_dir.mkdir(parents=True, exist_ok=False)

        outputs: list[Path] = []
        records: list[SFXGeneration] = []
        logs: list[str] = []
        prefix = "ambience" if track_type == "ambience" else "sfx"

        for index in range(int(variants)):
            current_seed = base_seed + index
            output_path = (run_dir / f"{prefix}_{index + 1:02d}_seed-{current_seed}.wav").resolve()
            record, combined_log = self.generate_file(
                prompt=prompt,
                track_type=track_type,
                output_path=output_path,
                duration=duration,
                seed=current_seed,
                steps=steps,
                cfg=cfg,
                negative_prompt=negative_prompt,
            )
            logs.append(combined_log)
            outputs.append(output_path)
            records.append(record)

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": self.display_name,
            "generations": [asdict(record) for record in records],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return outputs, records, "\n\n".join(logs)


def command_as_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])
