"""Endpoints FastAPI expuestos como web function de Modal.

Rutas:
  POST /predict        → genera imagen(es)
  GET  /timing-report  → reporte de tiempos
  GET  /health         → healthcheck
"""
from __future__ import annotations

import modal

from config.infrastructure import app

# Importación lazy del modelo para evitar circular; se resuelve en runtime
# porque Modal ya lo tiene registrado en el mismo App.
from model.nova_anime import NovaAnimeModel  # noqa: F401


@app.function(
    timeout=60,
    secrets=[modal.Secret.from_name("nova-anime-checkpoint")],
)
@modal.asgi_app()
def fastapi_app():
    """Crea y devuelve la aplicación FastAPI.

    Los imports de fastapi y pydantic se hacen dentro de la función
    porque solo están disponibles dentro del contenedor Modal.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response

    from api.schemas import PredictInput

    app_fastapi = FastAPI(title="Nova Anime IL XL", version="1.0")

    @app_fastapi.post("/predict")
    async def predict(body: PredictInput):
        """Genera imagen(es) a partir del prompt y devuelve PNG o JSON con base64."""
        try:
            images, timings = NovaAnimeModel().predict.remote(
                prompt=body.prompt,
                prepend_preprompt=body.prepend_preprompt,
                negative_prompt=body.negative_prompt,
                num_inference_steps=body.steps,
                guidance_scale=body.cfg_scale,
                guidance_rescale=body.guidance_rescale,
                clip_skip=body.clip_skip,
                pag_scale=body.pag_scale,
                width=body.width,
                height=body.height,
                seed=body.seed if body.seed != -1 else None,
                num_outputs=body.batch_size,
                face_yolov9c=body.face_yolov9c,
                hand_yolov9c=body.hand_yolov9c,
                person_yolov8m_seg=body.person_yolov8m_seg,
            )
            headers = {
                "X-Inference-Seconds": str(timings["inference_seconds"]),
                "X-Cold-Start-Seconds": str(timings["cold_start_seconds"]),
                "X-Request-Number": str(timings["request_number"]),
            }
            # Una sola imagen → respuesta binaria PNG directa
            if len(images) == 1:
                return Response(
                    content=images[0], media_type="image/png", headers=headers
                )
            # Múltiples imágenes → JSON con base64
            import base64

            from fastapi.responses import JSONResponse

            return JSONResponse(
                content={
                    "images": [base64.b64encode(img).decode() for img in images],
                    "timings": timings,
                },
                headers=headers,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.get("/timing-report")
    async def timing_report():
        """Devuelve el reporte de tiempos acumulados (cold start, inferencia)."""
        try:
            report = NovaAnimeModel().get_timing_report.remote()
            return Response(content=report, media_type="text/plain; charset=utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.get("/health")
    async def health():
        """Healthcheck simple."""
        return {"status": "ok"}

    return app_fastapi
