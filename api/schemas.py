"""Esquemas Pydantic para la API HTTP (estilo Replicate)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config.constants import (
    DEFAULT_CLIP_SKIP,
    DEFAULT_GUIDANCE,
    DEFAULT_GUIDANCE_RESCALE,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
)


class PredictInput(BaseModel):
    """Esquema de entrada para /predict.

    Los campos son equivalentes a la API de Replicate (nova-anime-ilxl-v5.5).
    """

    prompt: str = Field(
        ...,
        description="Prompt de generación; acepta sintaxis Compel (word:1.2)",
    )
    negative_prompt: Optional[str] = Field(
        None, description="Prompt negativo (cosas que no quieres en la imagen)"
    )
    cfg_scale: float = Field(
        DEFAULT_GUIDANCE,
        ge=1.0,
        le=50.0,
        description="CFG scale (1 = desactivado)",
    )
    guidance_rescale: float = Field(
        DEFAULT_GUIDANCE_RESCALE,
        ge=0.0,
        le=5.0,
        description="Rescale CFG noise; 0 = CFG estándar (recomendado)",
    )
    pag_scale: float = Field(
        1.5, ge=0.0, le=50.0, description="PAG scale; 0 = desactivado, 1.5 = default"
    )
    clip_skip: Optional[int] = Field(
        DEFAULT_CLIP_SKIP, ge=1, description="Capas de CLIP a saltar"
    )
    width: int = Field(DEFAULT_WIDTH, ge=1, le=4096)
    height: int = Field(DEFAULT_HEIGHT, ge=1, le=4096)
    prepend_preprompt: bool = Field(
        True,
        description="Anteponer preprompt Replicate (masterpiece, best quality, ...)",
    )
    scheduler: Optional[str] = Field("Euler a", description="Scheduler (Euler a)")
    steps: int = Field(DEFAULT_STEPS, ge=1, le=100)
    batch_size: int = Field(1, ge=1, le=4, description="Número de imágenes")
    seed: Optional[int] = Field(-1, description="-1 = aleatorio")
    face_yolov9c: bool = Field(True, description="Refinado de caras ADetailer")
    hand_yolov9c: bool = Field(False, description="ADetailer manos (aceptado, no implementado)")
    person_yolov8m_seg: bool = Field(False, description="ADetailer persona (aceptado, no implementado)")
