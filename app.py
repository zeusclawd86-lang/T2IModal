"""
Nova Anime IL v5.5 on Modal — punto de entrada principal.

Este archivo es el entry point para `modal run` y `modal deploy`.
Importa y re-exporta todos los componentes registrados en la app Modal:
  - NovaAnimeModel  (clase GPU con predict/predict_one/get_timing_report)
  - fastapi_app     (web function con /predict, /timing-report, /health)

Uso:
  modal run  app.py --prompt "..."    # prueba local
  modal deploy app.py                 # producción
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Re-exportar infraestructura Modal para que `modal` la descubra ---
from config.infrastructure import app  # noqa: F401

# --- Re-exportar componentes registrados en la app ---
from model.nova_anime import NovaAnimeModel  # noqa: F401
from api.endpoints import fastapi_app  # noqa: F401

# --- Constantes para el entrypoint local ---
from config.constants import (
    DEFAULT_CLIP_SKIP,
    DEFAULT_GUIDANCE,
    DEFAULT_GUIDANCE_RESCALE,
    DEFAULT_STEPS,
)

# Carpeta por defecto para guardar resultados (project root / outputs)
_DEFAULT_OUTPUTS_DIR = str(Path(__file__).resolve().parent / "outputs")


@app.local_entrypoint()
def main(
    prompt: str = "street, 1girl, dark-purple short hair, purple eyes, medium breasts, cleavage, casual clothes, smile",
    prepend_preprompt: bool = True,
    negative_prompt: Optional[str] = "nsfw, naked",
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_GUIDANCE,
    guidance_rescale: float = DEFAULT_GUIDANCE_RESCALE,
    clip_skip: Optional[int] = DEFAULT_CLIP_SKIP,
    seed: Optional[int] = None,
    save_dir: str = _DEFAULT_OUTPUTS_DIR,
    show_report: bool = True,
) -> None:
    """Entrypoint local: genera una imagen y la guarda en disco."""
    t0_total = time.perf_counter()

    out_bytes, timings = NovaAnimeModel().predict_one.remote(
        prompt,
        prepend_preprompt=prepend_preprompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        guidance_rescale=guidance_rescale,
        clip_skip=clip_skip,
        seed=seed,
    )

    total_seconds = time.perf_counter() - t0_total
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = path / f"output_{ts}.png"
    out_path.write_bytes(out_bytes)

    print(f"Guardado en {out_path}")
    print(
        f"Tiempos: cold_start={timings['cold_start_seconds']}s  "
        f"inferencia={timings['inference_seconds']}s  "
        f"total_cliente={total_seconds:.2f}s  "
        f"request#{timings['request_number']}"
    )

    if show_report:
        report = NovaAnimeModel().get_timing_report.remote()
        print(report)
