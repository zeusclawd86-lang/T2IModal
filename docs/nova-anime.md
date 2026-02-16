# Nova Anime IL v5.5 en Modal

Arquitectura equivalente a [Replicate: aisha-ai-official/nova-anime-ilxl-v5.5](https://replicate.com/aisha-ai-official/nova-anime-ilxl-v5.5), desplegada en [Modal](https://modal.com) para obtener **calidad y latencia similares o mejores**.

## Mismo modelo que Replicate

Replicate ejecuta **exactamente** este checkpoint de CivitAI:

- **Modelo:** [Nova Anime XL - IL v5B](https://civitai.com/models/376130?modelVersionId=1500882) (original IL v5.5)
- **Tipo:** Checkpoint Merge SDXL, ~6.46 GB SafeTensor fp16
- **Recomendado:** Sampler Euler a, steps 20–30, CFG 5–7, negative prompts según CivitAI

## Comparación con Replicate

| Aspecto        | Replicate              | Este proyecto (Modal)        |
|----------------|------------------------|------------------------------|
| GPU            | L40S                   | A100 40GB (configurable)      |
| Latencia típica| ~12 s                  | Objetivo: ~10–12 s o menos   |
| Checkpoint     | CivitAI 376130 / v1500882 | El mismo (vía CHECKPOINT_URL o HF) |
| API            | REST                   | REST + `modal run`            |

### Paridad con la API de Replicate (versión ac582da6)

Se imitan los inputs de [Replicate](https://replicate.com/aisha-ai-official/nova-anime-ilxl-v5.5/versions/ac582da6619c6deb7ff561eefb5824324c9ff0f485ccb7867964bce7040b0568/api):

| Parámetro Replicate | Default | Implementado |
|---------------------|---------|--------------|
| `prompt` | (string) | ✅ |
| `negative_prompt` | "nsfw, naked" | ✅ |
| `cfg_scale` | 5 (1–50) | ✅ `guidance_scale` |
| `guidance_rescale` | 1 (0–5, 0/1=off) | ✅ |
| `pag_scale` | 0 (0=off) | ✅ Aceptado (PAG no aplicado) |
| `clip_skip` | 2 | ✅ |
| `width`, `height` | 1024 | ✅ |
| `prepend_preprompt` | **true** | ✅ bool: antepone "masterpiece, best quality, ..." y negative fijo |
| `scheduler` | Euler a | ✅ Fijo |
| `steps` | 30 | ✅ |
| `batch_size` | 1 (1–4) | ✅ `num_outputs` |
| `seed` | -1 (random) | ✅ |
| `face_yolov9c` | true | ✅ Aceptado (ADetailer no ejecutado) |
| `hand_yolov9c`, `person_yolov8m_seg` | false | ✅ Aceptados |

## Requisitos

- Cuenta en [Modal](https://modal.com) y `modal token` configurado.
- Checkpoint del modelo (ver sección **Checkpoint**).

## Instalación

Desde la raíz del proyecto (`TelegramChatbotModal/`):

```bash
pip install -r requirements.txt
modal setup
```

## Checkpoint (el mismo que Replicate)

El modelo que corre en Replicate es [CivitAI Nova Anime XL - IL v5B](https://civitai.com/models/376130?modelVersionId=1500882) (version ID **1500882**, ~6.46 GB SafeTensor). Tres formas de usarlo en Modal:

### 1. URL del checkpoint (`CHECKPOINT_URL`)

**Opción A:** Descarga el `.safetensors` desde CivitAI (botón Download) y hostéalo en una URL accesible (S3, GCS, R2, etc.). Crea en Modal un Secret con `CHECKPOINT_URL` = esa URL. En el primer arranque Modal descargará el archivo al volume y lo reutilizará.

**Opción B (mismo checkpoint que Replicate):** Con [CivitAI API token](https://civitai.com/user/account), en Modal crea un Secret con:
- `CHECKPOINT_URL` = `https://civitai.com/api/download/models/1500882`
- `CIVITAI_API_KEY` = tu token de CivitAI

El código descargará el checkpoint desde CivitAI en el primer arranque y lo cacheará en el volume.

### 2. Hugging Face (`NOVA_ANIME_HF_ID`)

Si subes el checkpoint a Hugging Face (un solo `.safetensors` en el repo):

- Crea un Secret en Modal con tu `HF_TOKEN` (nombre sugerido: `huggingface-secret`).
- Configura la variable de entorno `NOVA_ANIME_HF_ID` con el `repo_id` (ej: `tu-usuario/nova-anime-il-v55`).
- Opcional: `NOVA_ANIME_HF_FILENAME` si el archivo no se detecta automáticamente (ej: `model.safetensors`).

Por defecto, si no configuras nada, se usa el repo `John6666/nova-anime-xl-il-v80-sdxl` (IL v8.0; para v5.5 conviene usar tu propio repo o `CHECKPOINT_URL`).

### 3. Ruta local en la imagen (`CHECKPOINT_PATH`)

Si incluyes el `.safetensors` en la imagen Modal (p. ej. en un `run_commands` que lo descargue), define `CHECKPOINT_PATH` con la ruta dentro del contenedor.

## Uso

### Línea de comandos

Ejecutar desde la **raíz del proyecto** (donde está la carpeta `outputs/`):

```bash
modal run app.py
modal run app.py --prompt "score_9, masterpiece, best quality, 1girl, cherry blossoms"
modal run app.py --prompt "tu prompt" --seed 42 --save-dir ./salida
```

La imagen se guarda por defecto en la carpeta **`outputs/`** del proyecto con nombre `output_YYYYMMDD_HHMMSS.png`. Con `--save-dir` puedes indicar otra ruta (ej. `.\salida` en Windows).

### Desde Python

```python
import modal

NovaAnimeModel = modal.Cls.lookup("nova-anime-ilxl", "NovaAnimeModel")

# Una imagen (bytes PNG)
img_bytes = NovaAnimeModel().predict_one.remote(
    "score_9, masterpiece, best quality, 1girl, BREAK cherry blossoms",
    seed=12345,
    num_inference_steps=25,
    guidance_scale=6.0,
)

# Varias imágenes
images = NovaAnimeModel().predict.remote(
    prompt="...",
    num_outputs=4,
    negative_prompt="...",
)
```

### API HTTP (estilo Replicate)

Desde la raíz del proyecto:

```bash
modal serve app.py
```

Esto levanta la app **nova-anime-ilxl** con un ASGI (FastAPI). En la consola verás una URL tipo `https://xxx--nova-anime-ilxl-fastapi.modal.run`.

Ejemplo de llamada:

```bash
curl -X POST "https://TU-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "score_9, masterpiece, 1girl, cherry blossoms", "seed": 42}' \
  --output imagen.png
```

Parámetros del body (todos opcionales salvo `prompt`):

- `prompt`, `negative_prompt`
- `num_inference_steps` (default 25)
- `guidance_scale` (default 6.0)
- `width`, `height` (default 1024)
- `seed`, `num_outputs` (1–4)

## Ajuste de latencia y calidad

- **Pasos:** 20–30 (default 25). Menos pasos = más rápido, algo menos de calidad.
- **GPU:** Por defecto A100 40GB. Puedes cambiar en `app.py` a `modal.gpu.L40S()` o `modal.gpu.H100()` según disponibilidad y coste.
- **Calidad:** Sigue las recomendaciones de [CivitAI](https://civitai.com/models/376130) (Euler a, CFG 4–7, negative prompts sugeridos).

## Estructura

```
TelegramChatbotModal/
  app.py           # App Modal: NovaAnimeModel + optional FastAPI
  requirements.txt
  api/
  config/
  core/
  model/
  docs/
```

## Licencia del modelo

Nova Anime XL está bajo [NoobAI License](https://huggingface.co/Laxhar/noobai-XL-1.0/blob/main/README.md#model-license). Revisa los términos en CivitAI antes de uso comercial.
