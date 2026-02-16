"""Clase Modal NovaAnimeModel: carga del pipeline SDXL, inferencia y refinado.

Encapsula todo el ciclo de vida del modelo en un contenedor GPU de Modal:
  1. @modal.enter → carga checkpoint, VAE, Compel, PAG, pipeline de inpainting.
  2. predict / predict_one → genera imágenes con Compel/chunking + PAG + face refine.
  3. get_timing_report → resumen de tiempos acumulados en el Volume.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import modal

from config.constants import (
    DEFAULT_CLIP_SKIP,
    DEFAULT_GUIDANCE,
    DEFAULT_GUIDANCE_RESCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    REPLICATE_POST_PROMPT,
    REPLICATE_PRE_NEGATIVE,
    REPLICATE_PREPROMPT,
)
from config.infrastructure import (
    CACHE_DIR,
    TIMING_REPORT_PATH,
    app,
    volume,
)
from core.checkpoint import get_checkpoint_path
from core.face_refiner import refine_faces
from core.prompt_encoder import encode_long_prompt_sdxl


@app.cls(
    gpu="A100-40GB",
    timeout=600,
    scaledown_window=30,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("nova-anime-checkpoint")],
)
class NovaAnimeModel:
    """Modelo Nova Anime IL (SDXL) en Modal.

    API compatible con Replicate: prompt, negative_prompt, steps,
    guidance_scale, seed, width, height, face_yolov9c, etc.
    """

    # ------------------------------------------------------------------
    # Carga del modelo (cold start)
    # ------------------------------------------------------------------
    @modal.enter(snap=True)
    def load(self) -> None:
        """Carga el pipeline SDXL, VAE, Compel, PAG e inpainting al arrancar el contenedor.

        Con GPU Memory Snapshot (snap=True), todo el estado de GPU se captura
        después de esta función. Los cold starts posteriores restauran la
        memoria directamente sin re-ejecutar esta función (~5-10s vs ~40-60s).
        """
        import torch
        from compel import Compel, ReturnedEmbeddingsType
        from diffusers import (
            AutoencoderKL,
            AutoPipelineForText2Image,
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLPipeline,
        )

        from config.constants import SDXL_VAE_HF_ID

        t0_load = time.perf_counter()
        checkpoint_path = get_checkpoint_path()

        # VAE fp16-fix evita borrosidad y NaN en decodificado SDXL
        vae = AutoencoderKL.from_pretrained(
            SDXL_VAE_HF_ID,
            torch_dtype=torch.float16,
        )

        # Pipeline principal txt2img
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            vae=vae,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

        # Compel: pesos de prompt (word:1.2), [word] estilo Replicate
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )

        # PAG (Perturbed-Attention Guidance): mejora estructura cuando pag_scale > 0
        self.pipe_pag = AutoPipelineForText2Image.from_pipe(
            self.pipe, enable_pag=True, pag_applied_layers=["mid"]
        )

        # Inpainting para refinado de caras (ADetailer-style)
        enable_face_refine = os.environ.get(
            "ENABLE_FACE_REFINEMENT", "1"
        ).strip().lower() in ("1", "true", "yes")
        self.pipe_inpaint = None
        if enable_face_refine:
            try:
                self.pipe_inpaint = StableDiffusionXLInpaintPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    vae=vae,
                )
                self.pipe_inpaint.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.pipe_inpaint.scheduler.config
                )
                self.pipe_inpaint = self.pipe_inpaint.to("cuda")
                self.pipe_inpaint.set_progress_bar_config(disable=True)
            except Exception:
                self.pipe_inpaint = None

        # Warmup: ejecutar un forward pass corto para calentar kernels CUDA.
        # Estos kernels quedan incluidos en el GPU snapshot, haciendo que la
        # primera inferencia real sea más rápida tras restaurar.
        import torch

        print("Warmup: ejecutando forward pass de prueba...")
        try:
            warmup_embeds, warmup_pooled = self.compel("warmup test")
            neg_embeds, neg_pooled = self.compel("bad quality")
            _ = self.pipe_pag(
                prompt_embeds=warmup_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=warmup_pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=2,
                guidance_scale=5.0,
                width=512,
                height=512,
                pag_scale=1.5,
            ).images[0]
            del warmup_embeds, warmup_pooled, neg_embeds, neg_pooled, _
            torch.cuda.empty_cache()
            print("Warmup completado.")
        except Exception as e:
            print(f"Warmup falló (no afecta funcionalidad): {e}")

        # Métricas de rendimiento
        self._cold_start_seconds = time.perf_counter() - t0_load
        self._request_count = 0
        self._inference_times: list[float] = []

        # Persistir cache en Volume para futuros cold starts
        try:
            modal.Volume.from_name("nova-anime-cache").commit()
        except Exception:
            pass

        print(f"Snapshot listo. Cold start: {self._cold_start_seconds:.2f}s")

    # ------------------------------------------------------------------
    # Lógica central de inferencia
    # ------------------------------------------------------------------
    def _run_predict(
        self,
        prompt: str,
        prepend_preprompt: Union[bool, str, None] = True,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        guidance_rescale: float = DEFAULT_GUIDANCE_RESCALE,
        clip_skip: Optional[int] = DEFAULT_CLIP_SKIP,
        pag_scale: float = 1.5,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: Optional[int] = None,
        num_outputs: int = 1,
        face_yolov9c: bool = True,
        **_: object,
    ) -> tuple[list[bytes], float, float]:
        """Genera imágenes: Compel/chunking + PAG + face refine.

        Returns:
            (lista_de_pngs_bytes, inference_seconds, cold_start_seconds)
        """
        import torch

        t0_infer = time.perf_counter()
        request_num = self._request_count
        is_first = request_num == 0

        num_outputs = max(1, min(4, num_outputs))

        # Construir prompts completos con preprompt + post-prompt si corresponde
        full_prompt = prompt
        full_negative = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE
        if prepend_preprompt is True:
            full_prompt = REPLICATE_PREPROMPT + prompt + ", " + REPLICATE_POST_PROMPT
            full_negative = REPLICATE_PRE_NEGATIVE + (negative_prompt or "")
        elif isinstance(prepend_preprompt, str) and prepend_preprompt.strip():
            full_prompt = prepend_preprompt.strip().rstrip(",") + ", " + prompt

        # Codificar: Compel (rápido) → fallback chunking (prompts >77 tokens)
        try:
            prompt_embeds, pooled_positive = self.compel(full_prompt)
            negative_embeds, pooled_negative = self.compel(full_negative)
        except Exception:
            prompt_embeds, negative_embeds, pooled_positive, pooled_negative = (
                encode_long_prompt_sdxl(
                    self.pipe, full_prompt, full_negative, clip_skip=clip_skip
                )
            )

        # Igualar longitud si difieren (Compel puede generar longitudes distintas)
        if prompt_embeds.shape[1] != negative_embeds.shape[1]:
            target_len = max(prompt_embeds.shape[1], negative_embeds.shape[1])
            if prompt_embeds.shape[1] < target_len:
                pad = torch.zeros(
                    prompt_embeds.shape[0],
                    target_len - prompt_embeds.shape[1],
                    prompt_embeds.shape[2],
                    device=prompt_embeds.device,
                    dtype=prompt_embeds.dtype,
                )
                prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
            if negative_embeds.shape[1] < target_len:
                pad = torch.zeros(
                    negative_embeds.shape[0],
                    target_len - negative_embeds.shape[1],
                    negative_embeds.shape[2],
                    device=negative_embeds.device,
                    dtype=negative_embeds.dtype,
                )
                negative_embeds = torch.cat([negative_embeds, pad], dim=1)

        # Batch: repetir embeddings para num_outputs
        prompt_embeds = prompt_embeds.repeat(num_outputs, 1, 1)
        negative_embeds = negative_embeds.repeat(num_outputs, 1, 1)
        pooled_positive = pooled_positive.repeat(num_outputs, 1)
        pooled_negative = pooled_negative.repeat(num_outputs, 1)

        generator = None
        if seed is not None and seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Seleccionar pipeline (PAG si pag_scale > 0)
        pipe_to_use = self.pipe_pag if pag_scale > 0 else self.pipe
        kwargs = dict(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            pooled_prompt_embeds=pooled_positive,
            negative_pooled_prompt_embeds=pooled_negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            guidance_rescale=guidance_rescale if guidance_rescale > 0 else 0.0,
            original_size=(height, width),
            target_size=(height, width),
        )
        if clip_skip is not None:
            kwargs["clip_skip"] = clip_skip
        if pag_scale > 0:
            kwargs["pag_scale"] = pag_scale

        images = pipe_to_use(**kwargs).images

        # Refinado de caras (ADetailer-style con inpainting)
        if face_yolov9c and self.pipe_inpaint is not None:
            images = [
                refine_faces(
                    self.pipe_inpaint,
                    img,
                    full_prompt,
                    full_negative,
                    num_inference_steps=min(20, num_inference_steps),
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                for img in images
            ]

        # Serializar a PNG bytes
        out: list[bytes] = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())

        inference_seconds = time.perf_counter() - t0_infer
        self._inference_times.append(inference_seconds)
        cold_start_for_request = self._cold_start_seconds if is_first else 0.0
        self._request_count = request_num + 1

        # Reporte de tiempos → archivo en el Volume
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(TIMING_REPORT_PATH, "a") as f:
                f.write(
                    f"{ts} request={self._request_count} "
                    f"cold_start_s={cold_start_for_request:.2f} "
                    f"inference_s={inference_seconds:.2f}\n"
                )
            modal.Volume.from_name("nova-anime-cache").commit()
        except Exception:
            pass

        return (out, inference_seconds, cold_start_for_request)

    # ------------------------------------------------------------------
    # Métodos remotos (endpoints Modal)
    # ------------------------------------------------------------------
    @modal.method()
    def predict(
        self,
        prompt: str,
        prepend_preprompt: Union[bool, str, None] = True,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        guidance_rescale: float = DEFAULT_GUIDANCE_RESCALE,
        clip_skip: Optional[int] = DEFAULT_CLIP_SKIP,
        pag_scale: float = 1.5,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: Optional[int] = None,
        num_outputs: int = 1,
        face_yolov9c: bool = True,
        hand_yolov9c: bool = False,
        person_yolov8m_seg: bool = False,
    ) -> tuple[list[bytes], dict]:
        """Genera N imágenes y devuelve (lista_bytes, dict_timings)."""
        images, inference_s, cold_s = self._run_predict(
            prompt=prompt,
            prepend_preprompt=prepend_preprompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            pag_scale=pag_scale,
            width=width,
            height=height,
            seed=seed,
            num_outputs=num_outputs,
            face_yolov9c=face_yolov9c,
        )
        timings = {
            "inference_seconds": round(inference_s, 2),
            "cold_start_seconds": round(cold_s, 2),
            "request_number": self._request_count,
        }
        return (images, timings)

    @modal.method()
    def predict_one(self, prompt: str, **kwargs) -> tuple[bytes, dict]:
        """Genera una sola imagen y devuelve (bytes, dict_timings)."""
        images, inference_s, cold_s = self._run_predict(
            prompt=prompt, num_outputs=1, **kwargs
        )
        timings = {
            "inference_seconds": round(inference_s, 2),
            "cold_start_seconds": round(cold_s, 2),
            "request_number": self._request_count,
        }
        return (images[0], timings)

    @modal.method()
    def get_timing_report(self) -> str:
        """Devuelve el reporte completo de tiempos (resumen + detalle por request)."""
        try:
            if not Path(TIMING_REPORT_PATH).exists():
                return "No hay registros aún. Ejecuta al menos una generación.\n"
            lines = Path(TIMING_REPORT_PATH).read_text().strip().split("\n")
            if not lines:
                return "Reporte vacío.\n"
            cold_starts: list[float] = []
            inference_times: list[float] = []
            for line in lines:
                if "cold_start_s=" in line and "inference_s=" in line:
                    try:
                        parts = line.split()
                        for p in parts:
                            if p.startswith("cold_start_s="):
                                cold_starts.append(float(p.split("=")[1]))
                            elif p.startswith("inference_s="):
                                inference_times.append(float(p.split("=")[1]))
                    except (ValueError, IndexError):
                        pass
            n = len(inference_times)
            total_cold = sum(cold_starts)
            avg_inf = sum(inference_times) / n if n else 0
            min_inf = min(inference_times) if inference_times else 0
            max_inf = max(inference_times) if inference_times else 0
            summary = (
                f"=== Reporte de tiempos Nova Anime ===\n"
                f"Total requests: {n}\n"
                f"Cold start (solo primer request): {total_cold:.2f}s\n"
                f"Inferencia: avg={avg_inf:.2f}s min={min_inf:.2f}s max={max_inf:.2f}s\n"
                f"--- Detalle por request ---\n"
            )
            return summary + "\n".join(lines) + "\n"
        except Exception as e:
            return f"Error leyendo reporte: {e}\n"
