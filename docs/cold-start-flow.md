# Flujo de Cold Start

Este diagrama muestra el proceso de inicialización del contenedor GPU (cold start) que ocurre en la primera llamada o cuando el contenedor se recicla.

## Diagrama de Secuencia

```mermaid
sequenceDiagram
    participant ModalCloud as Modal Cloud
    participant Container as GPU Container<br/>(A100-40GB)
    participant NovaModel as NovaAnimeModel<br/>@modal.enter()
    participant Checkpoint as checkpoint.py
    participant Volume as Modal Volume<br/>(/cache)
    participant CivitAI as CivitAI API
    participant HF as Hugging Face

    Note over ModalCloud: Primera llamada a predict()<br/>o contenedor reciclado
    
    ModalCloud->>Container: Crear contenedor GPU
    activate Container
    Note over Container: Provisionar A100-40GB<br/>Instalar dependencias<br/>Montar volume
    
    Container->>NovaModel: Ejecutar @modal.enter()
    activate NovaModel
    
    Note over NovaModel: Iniciar timer<br/>t0_load = time.perf_counter()
    
    NovaModel->>Checkpoint: get_checkpoint_path()
    activate Checkpoint
    
    Note over Checkpoint: Estrategia de resolución:<br/>1. Embebido en imagen<br/>2. CHECKPOINT_PATH<br/>3. Volume cache<br/>4. CHECKPOINT_URL<br/>5. HuggingFace
    
    Checkpoint->>Checkpoint: ¿Existe LOCAL_CHECKPOINT_IN_IMAGE?
    alt Checkpoint embebido en imagen
        Note over Checkpoint: /local/model.safetensors
        Checkpoint-->>NovaModel: Ruta local
    else No embebido
        Checkpoint->>Checkpoint: ¿Variable CHECKPOINT_PATH?
        alt CHECKPOINT_PATH definida
            Checkpoint-->>NovaModel: Ruta de env var
        else No definida
            Checkpoint->>Volume: ¿Existe /cache/checkpoint/model.safetensors?
            activate Volume
            
            alt Existe en cache
                Volume-->>Checkpoint: Ruta del cache
                deactivate Volume
                Checkpoint-->>NovaModel: /cache/checkpoint/model.safetensors
            else No existe en cache
                deactivate Volume
                
                Checkpoint->>Checkpoint: ¿Variable CHECKPOINT_URL?
                
                alt CHECKPOINT_URL definida
                    Note over Checkpoint: URL configurada<br/>(CivitAI o custom)
                    
                    Checkpoint->>Checkpoint: ¿URL contiene "civitai.com"?
                    
                    alt Es CivitAI
                        Checkpoint->>Checkpoint: Leer CIVITAI_API_KEY
                        Note over Checkpoint: Agregar token a URL<br/>y header Authorization
                        
                        Checkpoint->>CivitAI: GET /api/download/models/1500882<br/>?token=XXX<br/>Header: Bearer XXX
                        activate CivitAI
                        
                        alt Token válido
                            CivitAI-->>Checkpoint: Stream ~6.46 GB
                            Note over Checkpoint: Descarga: ~2-5 min<br/>(depende de red)
                        else Token inválido o modelo restringido
                            CivitAI-->>Checkpoint: 403 Forbidden
                            Checkpoint-->>NovaModel: RuntimeError:<br/>"Comprueba CIVITAI_API_KEY"
                            NovaModel-->>Container: Error fatal
                            Container-->>ModalCloud: Cold start fallido
                        end
                        deactivate CivitAI
                        
                    else URL custom (S3, GCS, R2, etc.)
                        Checkpoint->>Checkpoint: urllib.request.urlretrieve()
                        Note over Checkpoint: Descargar desde URL<br/>sin autenticación
                    end
                    
                    Checkpoint->>Volume: Escribir /cache/checkpoint/model.safetensors
                    activate Volume
                    Volume-->>Checkpoint: OK
                    deactivate Volume
                    Checkpoint-->>NovaModel: /cache/checkpoint/model.safetensors
                    
                else CHECKPOINT_URL no definida (fallback HuggingFace)
                    Note over Checkpoint: Usar HuggingFace<br/>NOVA_ANIME_HF_ID<br/>(default: John6666/nova-anime-xl-il-v80-sdxl)
                    
                    Checkpoint->>Checkpoint: ¿Variable NOVA_ANIME_HF_FILENAME?
                    
                    alt Filename especificado
                        Checkpoint->>HF: hf_hub_download(repo_id, filename)
                        activate HF
                        HF-->>Checkpoint: Descargar archivo específico
                        deactivate HF
                    else Filename no especificado
                        Checkpoint->>HF: list_repo_files(repo_id)
                        activate HF
                        HF-->>Checkpoint: Lista de archivos
                        deactivate HF
                        
                        Checkpoint->>Checkpoint: Filtrar *.safetensors
                        
                        alt Hay archivos .safetensors
                            Checkpoint->>HF: hf_hub_download(repo_id, primer_safetensor)
                            activate HF
                            HF-->>Checkpoint: Descargar primer .safetensors
                            deactivate HF
                        else No hay .safetensors
                            Checkpoint-->>NovaModel: RuntimeError:<br/>"No .safetensors en repo"
                        end
                    end
                    
                    Checkpoint->>Volume: Guardar en /cache/checkpoint/
                    activate Volume
                    Volume-->>Checkpoint: OK
                    deactivate Volume
                    Checkpoint-->>NovaModel: Ruta del checkpoint
                end
            end
        end
    end
    
    deactivate Checkpoint
    
    Note over NovaModel: checkpoint_path obtenido<br/>Iniciar carga de modelos
    
    NovaModel->>NovaModel: Cargar VAE fp16-fix
    Note over NovaModel: AutoencoderKL.from_pretrained()<br/>SDXL VAE (evita blur/NaN)<br/>~1-2s
    
    NovaModel->>NovaModel: Cargar pipeline SDXL principal
    Note over NovaModel: StableDiffusionXLPipeline<br/>.from_single_file(checkpoint_path)<br/>torch_dtype=float16<br/>~10-15s
    
    NovaModel->>NovaModel: Configurar scheduler
    Note over NovaModel: EulerAncestralDiscreteScheduler<br/>~0.1s
    
    NovaModel->>NovaModel: Mover pipeline a CUDA
    Note over NovaModel: pipe.to("cuda")<br/>~5-8s
    
    NovaModel->>NovaModel: Cargar Compel
    Note over NovaModel: Compel(tokenizer, text_encoder)<br/>Para prompt weighting<br/>~1-2s
    
    NovaModel->>NovaModel: Crear PAG pipeline
    Note over NovaModel: AutoPipelineForText2Image<br/>.from_pipe(enable_pag=True)<br/>~2-3s
    
    alt ENABLE_FACE_REFINEMENT = true
        NovaModel->>NovaModel: Cargar inpainting pipeline
        Note over NovaModel: StableDiffusionXLInpaintPipeline<br/>.from_single_file(checkpoint_path)<br/>~8-10s
        
        NovaModel->>NovaModel: Configurar scheduler inpaint
        NovaModel->>NovaModel: Mover inpaint a CUDA
    else Face refinement desactivado
        Note over NovaModel: self.pipe_inpaint = None<br/>Ahorro: ~8-10s
    end
    
    NovaModel->>NovaModel: Calcular cold_start_seconds
    Note over NovaModel: t_elapsed = time.perf_counter() - t0_load<br/>self._cold_start_seconds = t_elapsed
    
    NovaModel->>NovaModel: Inicializar métricas
    Note over NovaModel: self._request_count = 0<br/>self._inference_times = []
    
    NovaModel->>Volume: Commit volume cache
    activate Volume
    Note over Volume: Persistir checkpoint<br/>y estado para futuros cold starts
    Volume-->>NovaModel: Cache confirmado
    deactivate Volume
    
    Note over NovaModel: Cold start completado<br/>Total: ~30-60s
    
    deactivate NovaModel
    Container-->>ModalCloud: Contenedor ready
    deactivate Container
    
    Note over ModalCloud: Contenedor warm<br/>Listo para inferencia
```

## Fases del Cold Start

### 1. Provisión del Contenedor (Modal)
- **Duración:** ~5-10 segundos
- Crear contenedor con GPU A100-40GB
- Instalar imagen Docker con dependencias
- Montar Modal Volume en `/cache`

### 2. Resolución del Checkpoint
- **Duración:** Variable según estrategia

| Estrategia | Duración típica | Notas |
|------------|-----------------|-------|
| Embebido en imagen | ~0s | Checkpoint pre-incluido (aumenta tamaño de imagen) |
| CHECKPOINT_PATH | ~0s | Ruta directa en contenedor |
| Volume cache (hit) | ~0s | Checkpoint ya descargado previamente |
| CHECKPOINT_URL (CivitAI) | ~2-5 min | Descarga ~6.46 GB desde CivitAI |
| HuggingFace | ~2-5 min | Descarga ~6.46 GB desde HF |

### 3. Carga de Modelos
- **Duración:** ~20-35 segundos

| Componente | Duración | Propósito |
|------------|----------|-----------|
| VAE fp16-fix | ~1-2s | Evita blur y NaN en SDXL |
| Pipeline SDXL | ~10-15s | Modelo principal de diffusion |
| Scheduler | ~0.1s | Euler Ancestral |
| To CUDA | ~5-8s | Transferir a memoria GPU |
| Compel | ~1-2s | Prompt weighting y parsing |
| PAG pipeline | ~2-3s | Perturbed-Attention Guidance |
| Inpaint pipeline | ~8-10s | Face refinement (opcional) |

### 4. Persistencia de Cache
- **Duración:** ~1-2 segundos
- Commit del Volume para futuros cold starts
- Guardar checkpoint en `/cache/checkpoint/model.safetensors`

## Tiempos Totales de Cold Start

| Escenario | Tiempo total |
|-----------|--------------|
| **Checkpoint en cache** | ~30-40s |
| **Primera descarga desde CivitAI** | ~3-6 min |
| **Primera descarga desde HuggingFace** | ~3-6 min |
| **Con face refinement OFF** | -8-10s |

## Optimizaciones

### 1. Cache Persistente
- Checkpoint se descarga **solo una vez**
- Se guarda en Modal Volume (persistente)
- Futuros cold starts reutilizan el cache

### 2. Warm Containers
- Modal mantiene contenedores activos ~5 minutos
- Requests frecuentes → **0 segundos** de cold start
- Solo paga por tiempo de inferencia

### 3. Face Refinement Opcional
```bash
# Desactivar para cold starts más rápidos
export ENABLE_FACE_REFINEMENT=0
```
Ahorro: ~8-10 segundos

### 4. Checkpoint Embebido
Incluir checkpoint en la imagen Docker:
```python
# En infrastructure.py
app = modal.App(
    image=image.copy_local_file(
        "/path/to/model.safetensors",
        "/local/model.safetensors"
    )
)
```
**Ventaja:** Cold start ~30s
**Desventaja:** Imagen muy pesada (~6.5 GB)

## Variables de Entorno

### Checkpoint
```bash
# Opción 1: URL directa
CHECKPOINT_URL="https://civitai.com/api/download/models/1500882"
CIVITAI_API_KEY="tu_token_de_civitai"

# Opción 2: Hugging Face
NOVA_ANIME_HF_ID="usuario/repo"
NOVA_ANIME_HF_FILENAME="model.safetensors"  # Opcional
HF_TOKEN="tu_token_hf"  # Para repos privados

# Opción 3: Ruta local en imagen
CHECKPOINT_PATH="/local/model.safetensors"
```

### Face Refinement
```bash
# Activar/desactivar inpainting pipeline
ENABLE_FACE_REFINEMENT=1  # Default: activado
ENABLE_FACE_REFINEMENT=0  # Desactivar para cold starts rápidos
```

## Estrategia de Resolución del Checkpoint

El sistema busca el checkpoint en este orden:

1. **`LOCAL_CHECKPOINT_IN_IMAGE`** (`/local/model.safetensors`)
   - Checkpoint pre-incluido en imagen Docker
   - Más rápido pero aumenta tamaño de imagen

2. **`CHECKPOINT_PATH`** (variable de entorno)
   - Ruta explícita en el contenedor
   - Útil para montar desde otros volumes

3. **Volume cache** (`/cache/checkpoint/model.safetensors`)
   - Checkpoint descargado previamente
   - Se reutiliza en futuros cold starts

4. **`CHECKPOINT_URL`** (descarga desde URL)
   - CivitAI con autenticación
   - URLs custom (S3, GCS, R2, etc.)

5. **HuggingFace** (fallback)
   - `NOVA_ANIME_HF_ID` (repo)
   - `NOVA_ANIME_HF_FILENAME` (opcional)
   - Descarga el primer `.safetensors` encontrado

## Métricas de Cold Start

El sistema registra métricas en cada request:

```python
self._cold_start_seconds  # Tiempo de cold start (solo primer request)
self._request_count       # Número de request actual
self._inference_times     # Lista de tiempos de inferencia
```

Disponibles en:
- Headers HTTP: `X-Cold-Start-Seconds`
- Output de `modal run`: consola
- Endpoint `/timing-report`: estadísticas acumuladas

## Logs de Cold Start

Ejemplo de logs durante cold start:

```
[Modal] Provisioning GPU container (A100-40GB)...
[Modal] Mounting volume nova-anime-cache...
[NovaAnimeModel] Resolving checkpoint path...
[checkpoint.py] Checking volume cache: /cache/checkpoint/model.safetensors
[checkpoint.py] Cache miss. Downloading from CivitAI...
[checkpoint.py] GET https://civitai.com/api/download/models/1500882
[checkpoint.py] Downloaded 6.46 GB in 3m 12s
[checkpoint.py] Saved to /cache/checkpoint/model.safetensors
[NovaAnimeModel] Loading VAE fp16-fix...
[NovaAnimeModel] Loading StableDiffusionXLPipeline...
[NovaAnimeModel] Loading Compel...
[NovaAnimeModel] Loading PAG pipeline...
[NovaAnimeModel] Loading inpainting pipeline...
[NovaAnimeModel] Moving models to CUDA...
[NovaAnimeModel] Cold start completed in 215.3s
[Modal] Container ready. Waiting for requests...
```

## Troubleshooting

### Error: "CivitAI returned 403 Forbidden"
**Causa:** Token inválido o modelo requiere login
**Solución:** 
1. Verificar `CIVITAI_API_KEY` en Modal Secrets
2. Aceptar términos del modelo en CivitAI
3. Generar nuevo API token en https://civitai.com/user/account

### Error: "No .safetensors in HuggingFace repo"
**Causa:** Repo no contiene archivos `.safetensors`
**Solución:**
1. Especificar `NOVA_ANIME_HF_FILENAME` con el nombre correcto
2. Usar `CHECKPOINT_URL` en su lugar

### Cold Start muy lento (>5 min)
**Causa:** Descarga lenta desde CivitAI/HF
**Solución:**
1. Verificar conexión de red del datacenter Modal
2. Usar URL custom más cercana (S3 en misma región)
3. Embedder checkpoint en la imagen Docker

### Out of Memory (OOM) durante cold start
**Causa:** GPU A100-40GB insuficiente para cargar todos los modelos
**Solución:**
1. Desactivar face refinement (`ENABLE_FACE_REFINEMENT=0`)
2. Usar GPU con más memoria (A100-80GB, H100)
