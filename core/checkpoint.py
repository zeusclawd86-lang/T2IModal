"""Resolución del checkpoint: busca el modelo en imagen embebida, env vars, Volume o HuggingFace."""
from __future__ import annotations

import os
from pathlib import Path

from config.infrastructure import (
    LOCAL_CHECKPOINT_IN_IMAGE,
    MODEL_CACHE_DIR,
)


def get_checkpoint_path() -> str:
    """Devuelve la ruta al checkpoint del modelo.

    Orden de prioridad:
      1. Checkpoint embebido en la imagen Docker (más rápido).
      2. Variable de entorno CHECKPOINT_PATH (ruta explícita).
      3. Checkpoint en el Volume cache (/cache/checkpoint/model.safetensors).
      4. Descarga desde CHECKPOINT_URL (CivitAI con API key).
      5. Descarga desde HuggingFace (fallback público).
    """
    # 1) Embebido en imagen
    if Path(LOCAL_CHECKPOINT_IN_IMAGE).exists():
        return LOCAL_CHECKPOINT_IN_IMAGE

    # 2) Variable de entorno CHECKPOINT_PATH
    path = os.environ.get("CHECKPOINT_PATH")
    if path and Path(path).exists():
        return path

    # 3) Volume cache
    Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cached = f"{MODEL_CACHE_DIR}/model.safetensors"
    if Path(cached).exists():
        return cached

    # 4) Descarga desde CHECKPOINT_URL (CivitAI)
    url = os.environ.get("CHECKPOINT_URL")
    if url:
        if not Path(cached).exists():
            import urllib.request

            api_key = (
                os.environ.get("CIVITAI_API_KEY")
                or os.environ.get("CIVITAI_TOKEN")
                or ""
            ).strip()

            if api_key and "civitai.com" in url:
                import urllib.error
                from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

                parsed = list(urlparse(url))
                qs = parse_qs(parsed[4], keep_blank_values=True)
                qs["token"] = [api_key]
                parsed[4] = urlencode(qs, doseq=True)
                url_with_token = urlunparse(parsed)

                req = urllib.request.Request(
                    url_with_token,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        Path(cached).write_bytes(resp.read())
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        raise RuntimeError(
                            "CivitAI devolvió 403 Forbidden. Comprueba que CIVITAI_API_KEY "
                            "en el secret 'nova-anime-checkpoint' sea un token válido de "
                            "https://civitai.com/user/account (Create API Key). "
                            "Si el modelo requiere login, acepta los términos en la web."
                        ) from e
                    raise
            else:
                urllib.request.urlretrieve(url, cached)
        return cached

    # 5) Fallback HuggingFace
    hf_id = os.environ.get("NOVA_ANIME_HF_ID", "John6666/nova-anime-xl-il-v80-sdxl")
    hf_file = os.environ.get("NOVA_ANIME_HF_FILENAME")

    from huggingface_hub import hf_hub_download, list_repo_files

    if hf_file:
        dest = hf_hub_download(
            repo_id=hf_id,
            filename=hf_file,
            local_dir=MODEL_CACHE_DIR,
            local_dir_use_symlinks=False,
        )
        return dest

    files = list_repo_files(hf_id)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    if not safetensors:
        raise RuntimeError(
            f"No .safetensors in {hf_id}. Set CHECKPOINT_URL or NOVA_ANIME_HF_FILENAME."
        )

    dest = hf_hub_download(
        repo_id=hf_id,
        filename=safetensors[0],
        local_dir=MODEL_CACHE_DIR,
        local_dir_use_symlinks=False,
    )
    return dest
