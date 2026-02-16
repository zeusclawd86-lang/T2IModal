"""Simulador de llamadas remotas al modelo desplegado en Modal.

Simula el flujo de un bot de Telegram donde múltiples usuarios
solicitan generación de imágenes de forma continua.

Requisitos:
  1. Haber desplegado la app:  modal deploy app.py
  2. Tener el token de Modal configurado (modal token set ...)
  3. Activar el entorno virtual: source .venv/bin/activate (Linux/Mac)
      o: .venv\Scripts\activate (Windows)

Uso:
  python test/simulate_bot.py                           # ejecuta la simulación completa
  python test/simulate_bot.py --prompt "1girl, ..."     # una sola imagen
  python test/simulate_bot.py --delay 5                 # 5s entre peticiones
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import modal

# Nombre de la app tal como está definida en infrastructure.py
APP_NAME = "nova-anime-ilxl"
CLASS_NAME = "NovaAnimeModel"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def get_model():
    """Obtiene una referencia remota al modelo desplegado."""
    Model = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    return Model()


def generate_image(
    model,
    prompt: str,
    negative_prompt: str = "nsfw, naked",
    steps: int = 30,
    cfg_scale: float = 5.0,
    seed: int | None = None,
) -> tuple[bytes, dict, float]:
    """Genera una imagen y devuelve (bytes, timings, duración_total_cliente)."""
    t0 = time.perf_counter()

    image_bytes, timings = model.predict_one.remote(
        prompt=prompt,
        prepend_preprompt=True,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        seed=seed,
    )

    client_duration = time.perf_counter() - t0
    return image_bytes, timings, client_duration


def save_image(image_bytes: bytes, prefix: str = "bot_sim") -> Path:
    """Guarda la imagen en la carpeta outputs/ y devuelve el path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{prefix}_{ts}.png"
    out_path.write_bytes(image_bytes)
    return out_path


def run_single(prompt: str, **kwargs) -> None:
    """Ejecuta una única generación (simula un usuario del bot)."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    model = get_model()
    image_bytes, timings, client_duration = generate_image(model, prompt, **kwargs)

    out_path = save_image(image_bytes)

    print(f"Imagen guardada en: {out_path}")
    print(
        f"Cold start: {timings['cold_start_seconds']}s | "
        f"Inferencia: {timings['inference_seconds']}s | "
        f"Total cliente: {client_duration:.2f}s | "
        f"Request #{timings['request_number']}"
    )


def run_simulation(delay: float = 10.0) -> None:
    """Simula múltiples usuarios del bot haciendo peticiones consecutivas.

    Con container_idle_timeout=30, si delay < 30 el contenedor se reutiliza
    (warm start). Si delay > 30 verás un cold start en la siguiente petición.
    """
    prompts = [
        "street, 1girl, dark-purple short hair, purple eyes, medium breasts, cleavage, casual clothes, smile",
        "cyberpunk city, neon lights, rainy night, 1girl walking with umbrella, reflections",
        "beach sunset, 1girl, long blonde hair, white sundress, wind blowing hair, golden hour",
        "cozy cafe interior, 1girl reading a book, glasses, warm lighting, coffee cup on table",
        "1girl, red kimono, cherry blossoms falling, traditional japanese garden, serene expression",
    ]

    model = get_model()
    print(f"Simulación: {len(prompts)} peticiones con {delay}s de delay entre cada una")
    print(f"Container idle timeout: 30s (warm si delay < 30s, cold si delay > 30s)\n")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Petición {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")

        image_bytes, timings, client_duration = generate_image(model, prompt)
        out_path = save_image(image_bytes, prefix=f"bot_sim_{i}")

        results.append({
            "prompt": prompt,
            "client_duration": client_duration,
            **timings,
        })

        print(
            f"Cold start: {timings['cold_start_seconds']}s | "
            f"Inferencia: {timings['inference_seconds']}s | "
            f"Total cliente: {client_duration:.2f}s | "
            f"Request #{timings['request_number']}"
        )
        print(f"Guardada en: {out_path}")

        if i < len(prompts):
            print(f"\nEsperando {delay}s para la siguiente petición...")
            time.sleep(delay)

    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE LA SIMULACIÓN")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        cold = "COLD" if r["cold_start_seconds"] > 0 else "WARM"
        print(
            f"  [{cold}] #{i}: cliente={r['client_duration']:.2f}s "
            f"inferencia={r['inference_seconds']}s "
            f"cold_start={r['cold_start_seconds']}s"
        )

    durations = [r["client_duration"] for r in results]
    print(f"\n  Promedio cliente: {sum(durations)/len(durations):.2f}s")
    print(f"  Min: {min(durations):.2f}s  Max: {max(durations):.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Simulador de llamadas remotas al bot de Telegram (Modal deploy)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Si se indica, genera una sola imagen con este prompt",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Segundos entre cada petición en la simulación (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Pasos de inferencia (default: 30)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="CFG scale (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducibilidad (default: aleatorio)",
    )
    args = parser.parse_args()

    if args.prompt:
        run_single(
            args.prompt,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
    else:
        run_simulation(delay=args.delay)


if __name__ == "__main__":
    main()
