"""Infraestructura de Modal: imagen Docker, App, Volume, rutas del contenedor."""
from pathlib import Path

import modal

# --- Rutas internas del contenedor ---
LOCAL_CHECKPOINT_IN_IMAGE = "/opt/checkpoint/model.safetensors"
CACHE_DIR = "/cache"
MODEL_CACHE_DIR = "/cache/checkpoint"
VAE_CACHE_DIR = "/cache/vae"
TIMING_REPORT_PATH = "/cache/nova_anime_timing_report.txt"

# --- Imagen Docker con CUDA + dependencias ---
CUDA_VERSION = "12.4.0"
PYTHON_VERSION = "3.12"

cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
    add_python=PYTHON_VERSION,
).entrypoint([])

nova_image = (
    cuda_image.apt_install("git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "libgl1")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "diffusers>=0.31.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "safetensors>=0.4.4",
        "huggingface-hub>=0.25.0",
        "hf_transfer>=0.1.0",
        "compel>=2.0.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.0",
    )
    .env(
        {
            "HF_HUB_CACHE": "/cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root",  # Establecer PYTHONPATH para que Python encuentre los módulos
        }
    )
    # Agregar todos los archivos Python del proyecto al contenedor
    .add_local_file("app.py", "/root/app.py")
    .add_local_file("config/__init__.py", "/root/config/__init__.py")
    .add_local_file("config/constants.py", "/root/config/constants.py")
    .add_local_file("config/infrastructure.py", "/root/config/infrastructure.py")
    .add_local_file("api/__init__.py", "/root/api/__init__.py")
    .add_local_file("api/endpoints.py", "/root/api/endpoints.py")
    .add_local_file("api/schemas.py", "/root/api/schemas.py")
    .add_local_file("core/__init__.py", "/root/core/__init__.py")
    .add_local_file("core/checkpoint.py", "/root/core/checkpoint.py")
    .add_local_file("core/face_refiner.py", "/root/core/face_refiner.py")
    .add_local_file("core/prompt_encoder.py", "/root/core/prompt_encoder.py")
    .add_local_file("model/__init__.py", "/root/model/__init__.py")
    .add_local_file("model/nova_anime.py", "/root/model/nova_anime.py")
)

# Si existe un checkpoint local, se embebe en la imagen Docker
_local_ckpt = Path(__file__).resolve().parent.parent / "checkpoint" / "novaAnimeXL_ilV5b.safetensors"
if _local_ckpt.exists():
    nova_image = nova_image.add_local_file(str(_local_ckpt), LOCAL_CHECKPOINT_IN_IMAGE)

# --- App y Volume de Modal ---
app = modal.App("nova-anime-ilxl", image=nova_image)
volume = modal.Volume.from_name("nova-anime-cache", create_if_missing=True)
