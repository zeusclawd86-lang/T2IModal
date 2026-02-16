# Documentación - Diagramas de Arquitectura

Esta carpeta contiene diagramas de secuencia que ilustran los diferentes flujos de ejecución de la aplicación Nova Anime IL v5.5 en Modal.

## Diagramas Disponibles

1. **[modal-run-flow.md](./modal-run-flow.md)** - Flujo de ejecución de `modal run` (CLI local)
2. **[api-http-flow.md](./api-http-flow.md)** - Flujo de llamada HTTP a la API REST
3. **[cold-start-flow.md](./cold-start-flow.md)** - Proceso de cold start y carga del modelo
4. **[checkpoint-resolution-flow.md](./checkpoint-resolution-flow.md)** - Resolución del checkpoint del modelo
5. **[image-generation-flow.md](./image-generation-flow.md)** - Proceso completo de generación de imagen
6. **[architecture-overview.md](./architecture-overview.md)** - Visión general de la arquitectura

## Cómo visualizar los diagramas

Los diagramas están escritos en sintaxis **Mermaid**, que es compatible con:

- **GitHub** - Se renderizan automáticamente en archivos `.md`
- **VS Code** - Con extensiones como "Markdown Preview Mermaid Support"
- **Cursor** - Con preview de Markdown
- **Online** - [Mermaid Live Editor](https://mermaid.live/)

## Estructura de la aplicación

```
TelegramChatbotModal/
├── app.py                    # Entry point principal
├── config/
│   ├── infrastructure.py     # Configuración de Modal (app, volume, cache)
│   └── constants.py          # Constantes y valores por defecto
├── model/
│   └── nova_anime.py         # Clase NovaAnimeModel (GPU, inferencia)
├── core/
│   ├── checkpoint.py         # Resolución del checkpoint
│   ├── prompt_encoder.py     # Codificación de prompts largos
│   └── face_refiner.py       # Refinado de caras (ADetailer)
└── api/
    ├── endpoints.py          # FastAPI endpoints (/predict, /health)
    └── schemas.py            # Pydantic schemas
```

## Componentes principales

### Modal App
- **Nombre:** `nova-anime-ilxl`
- **GPU:** A100-40GB (configurable)
- **Volume:** Cache persistente para checkpoints y reportes
- **Secrets:** `nova-anime-checkpoint` (CHECKPOINT_URL, CIVITAI_API_KEY, HF_TOKEN)

### NovaAnimeModel (Clase GPU)
- Carga el pipeline SDXL con VAE, Compel, PAG
- Métodos remotos: `predict()`, `predict_one()`, `get_timing_report()`
- Refinado de caras con YOLOv9c + inpainting

### API REST (FastAPI)
- `POST /predict` - Generación de imágenes
- `GET /timing-report` - Reporte de tiempos acumulados
- `GET /health` - Healthcheck

## Flujos principales

### 1. Modal Run (Local CLI)
Usuario ejecuta `modal run` → Modal llama `main()` → `predict_one.remote()` → GPU genera imagen → Descarga a `outputs/`

### 2. API HTTP
Cliente HTTP → FastAPI endpoint → `predict.remote()` → GPU genera imagen → Respuesta PNG o JSON base64

### 3. Cold Start
Primera llamada → Carga checkpoint → Carga VAE → Carga Compel → Carga PAG → Carga pipeline inpaint → Commit volume cache

## Optimizaciones

- **Cache persistente:** Checkpoint y modelos se guardan en Modal Volume
- **Compel:** Codificación rápida de prompts con pesos
- **Chunking:** Fallback para prompts largos (>77 tokens)
- **PAG:** Mejora estructura cuando `pag_scale > 0`
- **Face refinement:** Inpainting automático con YOLOv9c
- **Timing reports:** Métricas persistentes en Volume

## Referencias

- [Modal Documentation](https://modal.com/docs)
- [Replicate Nova Anime IL v5.5](https://replicate.com/aisha-ai-official/nova-anime-ilxl-v5.5)
- [CivitAI Nova Anime XL](https://civitai.com/models/376130)
