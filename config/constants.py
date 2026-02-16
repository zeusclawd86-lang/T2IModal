"""Constantes de generación alineadas con CivitAI Illustrious (nova-anime-ilxl-v5.5).

Referencia: https://civitai.com/models/376130?modelVersionId=1500882
"""

# --- Parámetros de inferencia por defecto ---
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 5.0          # CFG scale; 4–6 recomendado para Illustrious
DEFAULT_GUIDANCE_RESCALE = 0.0  # 0.0 = CFG estándar (NO usar 1.0, anula el CFG y causa borroneo)
DEFAULT_CLIP_SKIP = 2
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# --- VAE ---
SDXL_VAE_HF_ID = "madebyollin/sdxl-vae-fp16-fix"
#TODO this sohuld be in telegramchat repo
# --- Preprompts CivitAI Illustrious ---
REPLICATE_PREPROMPT = (
    "masterpiece, best quality, amazing quality, very aesthetic, "
    "high resolution, ultra-detailed, absurdres, newest, scenery, "
)
REPLICATE_POST_PROMPT = "BREAK, depth of field, volumetric lighting"
REPLICATE_PRE_NEGATIVE = (
    "modern, recent, old, oldest, cartoon, graphic, text, painting, crayon, "
    "graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, "
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, "
    "cropped, very displeasing, (worst quality, bad quality:1.2), sketch, "
    "jpeg artifacts, signature, watermark, username, simple background, "
    "conjoined, bad ai-generated, "
)

# Negative por defecto si no se usa prepend (CivitAI-style)
DEFAULT_NEGATIVE = (
    "modern, recent, old, oldest, cartoon, graphic, text, painting, crayon, "
    "graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, "
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, "
    "cropped, very displeasing, (worst quality, bad quality:1.2), sketch, "
    "jpeg artifacts, signature, watermark, username, simple background, "
    "conjoined, bad ai-generated"
)

# --- Límites CLIP para prompts largos ---
CLIP_MAX_LENGTH = 77
CONTENT_TOKENS_PER_CHUNK = 75

# --- Identificadores CivitAI ---
CIVITAI_MODEL_ID = "376130"
CIVITAI_VERSION_ID = "1500882"
