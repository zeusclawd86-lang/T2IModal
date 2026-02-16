"""Refinado de caras con inpainting (estilo ADetailer).

Detecta caras con Haar cascade de OpenCV, genera una máscara elíptica y ejecuta
inpainting SDXL sobre las regiones detectadas para mejorar la calidad facial.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLInpaintPipeline


def refine_faces(
    pipe_inpaint: "StableDiffusionXLInpaintPipeline",
    image: "Image.Image",
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
) -> "Image.Image":
    """Refina caras detectadas en la imagen usando inpainting SDXL.

    Args:
        pipe_inpaint: Pipeline de inpainting SDXL ya cargado en GPU.
        image: Imagen PIL a refinar.
        prompt: Prompt positivo para guiar el inpainting.
        negative_prompt: Prompt negativo.
        num_inference_steps: Pasos de inferencia para el inpainting.
        guidance_scale: CFG scale para el inpainting.
        seed: Semilla (se usa seed+1 para diferenciar del generado original).

    Returns:
        Imagen PIL con caras refinadas (o la original si no se detectaron caras).
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image as PILImage

    w, h = image.size
    img_np = np.array(image)
    if img_np.ndim == 2:
        gray = img_np
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return image

    # Máscara combinada: expandir bbox ~35% para capturar mejor ojos y detalles
    mask_np = np.zeros((h, w), dtype=np.uint8)
    for (x, y, bw, bh) in faces:
        pad = max(4, int(0.35 * max(bw, bh)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - cx) ** 2 / (rx**2 + 1)) + ((yy - cy) ** 2 / (ry**2 + 1)) <= 1
        mask_np[ellipse] = 255

    mask_pil = PILImage.fromarray(mask_np).convert("L")
    if mask_np.max() == 0:
        return image

    gen = None
    if seed is not None and seed != -1:
        gen = torch.Generator(device="cuda").manual_seed(seed + 1)

    out = pipe_inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=0.45,
        generator=gen,
    ).images[0]
    return out
