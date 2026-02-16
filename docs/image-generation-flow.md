# Flujo de Generación de Imagen

Este diagrama muestra el proceso completo de inferencia para generar una imagen con el modelo Nova Anime IL v5.5.

## Diagrama de Secuencia

```mermaid
sequenceDiagram
    participant Caller as Caller<br/>(main/endpoint)
    participant NovaModel as NovaAnimeModel
    participant RunPredict as _run_predict()
    participant Compel as Compel<br/>(Prompt Encoder)
    participant Chunking as encode_long_prompt_sdxl()<br/>(Fallback)
    participant Pipeline as SDXL Pipeline<br/>(GPU)
    participant PAG as PAG Pipeline<br/>(Perturbed-Attention)
    participant FaceRefiner as Face Refiner<br/>(YOLOv9c + Inpaint)
    participant Volume as Modal Volume

    Caller->>NovaModel: predict_one.remote(prompt, **kwargs)<br/>o predict.remote(...)
    activate NovaModel
    
    NovaModel->>RunPredict: _run_predict(prompt, steps, seed, ...)
    activate RunPredict
    
    Note over RunPredict: Iniciar timer<br/>t0_infer = time.perf_counter()
    
    RunPredict->>RunPredict: Validar num_outputs (1-4)
    Note over RunPredict: num_outputs = max(1, min(4, num_outputs))
    
    RunPredict->>RunPredict: Incrementar contador<br/>request_num = self._request_count<br/>is_first = (request_num == 0)
    
    rect rgb(255, 250, 240)
    Note over RunPredict: Fase 1: Construcción de prompts
    
    RunPredict->>RunPredict: ¿prepend_preprompt?
    
    alt prepend_preprompt = True
        Note over RunPredict: Agregar preprompt de Replicate
        RunPredict->>RunPredict: full_prompt = REPLICATE_PREPROMPT + prompt
        Note over RunPredict: REPLICATE_PREPROMPT:<br/>"masterpiece, best quality,<br/>amazing quality, very aesthetic,<br/>absurdres, newest, ..."
        
        RunPredict->>RunPredict: full_negative = REPLICATE_PRE_NEGATIVE + negative_prompt
        Note over RunPredict: REPLICATE_PRE_NEGATIVE:<br/>"lowres, (bad), text, error,<br/>fewer, extra, missing, worst quality,<br/>jpeg artifacts, low quality, ..."
        
    else prepend_preprompt = string custom
        RunPredict->>RunPredict: full_prompt = custom_preprompt + prompt
        Note over RunPredict: Usuario especifica<br/>preprompt personalizado
        
    else prepend_preprompt = False
        RunPredict->>RunPredict: full_prompt = prompt<br/>full_negative = negative_prompt
        Note over RunPredict: Sin preprompt
    end
    end
    
    rect rgb(240, 248, 255)
    Note over RunPredict,Chunking: Fase 2: Codificación de prompts
    
    RunPredict->>Compel: Intentar codificación rápida
    activate Compel
    Note over Compel: Compel maneja:<br/>• Pesos: (word:1.2), [word]<br/>• Dual text encoders<br/>• Pooled embeddings
    
    alt Prompt corto (≤77 tokens)
        Compel->>Compel: Tokenizar prompt/negative
        Compel->>Compel: Encode con text_encoder + text_encoder_2
        Compel->>Compel: Generar pooled embeddings
        
        Compel-->>RunPredict: (prompt_embeds, pooled_positive)<br/>(negative_embeds, pooled_negative)
        deactivate Compel
        
        Note over RunPredict: ✓ Codificación rápida<br/>~50-100ms
        
    else Prompt largo (>77 tokens) o error
        Compel-->>RunPredict: Exception (truncation error)
        deactivate Compel
        
        Note over RunPredict: Fallback a chunking
        
        RunPredict->>Chunking: encode_long_prompt_sdxl(pipe, full_prompt, full_negative)
        activate Chunking
        
        Note over Chunking: Dividir prompt en chunks de 77 tokens
        Chunking->>Chunking: Tokenizar y dividir
        
        loop Para cada chunk
            Chunking->>Chunking: Encode chunk con text_encoder
            Chunking->>Chunking: Encode chunk con text_encoder_2
        end
        
        Chunking->>Chunking: Concatenar embeddings
        Chunking->>Chunking: Aplicar clip_skip si especificado
        Note over Chunking: clip_skip = omitir últimas N capas<br/>del text encoder
        
        Chunking-->>RunPredict: (prompt_embeds, negative_embeds,<br/>pooled_positive, pooled_negative)
        deactivate Chunking
        
        Note over RunPredict: ✓ Chunking completado<br/>~200-500ms
    end
    
    RunPredict->>RunPredict: Igualar longitud de embeddings
    Note over RunPredict: Si prompt_embeds.shape[1] ≠ negative_embeds.shape[1]<br/>→ padding con zeros
    
    RunPredict->>RunPredict: Repetir embeddings para batch
    Note over RunPredict: prompt_embeds.repeat(num_outputs, 1, 1)<br/>negative_embeds.repeat(num_outputs, 1, 1)<br/>pooled_positive.repeat(num_outputs, 1)<br/>pooled_negative.repeat(num_outputs, 1)
    end
    
    rect rgb(240, 255, 240)
    Note over RunPredict,Pipeline: Fase 3: Generación con diffusion
    
    RunPredict->>RunPredict: ¿seed especificado?
    
    alt seed is not None and seed != -1
        RunPredict->>RunPredict: generator = torch.Generator("cuda")<br/>.manual_seed(seed)
        Note over RunPredict: Generación determinística
    else seed = None o -1
        Note over RunPredict: generator = None<br/>Generación aleatoria
    end
    
    RunPredict->>RunPredict: ¿pag_scale > 0?
    
    alt pag_scale > 0
        Note over RunPredict: Usar PAG pipeline<br/>(Perturbed-Attention Guidance)
        
        RunPredict->>PAG: Ejecutar diffusion con PAG
        activate PAG
        
        Note over PAG: PAG mejora estructura<br/>perturbando atención en mid layers
        
        PAG->>PAG: Loop denoising (num_inference_steps)
        Note over PAG: Por cada step:<br/>1. Self-attention en mid block<br/>2. Perturbar atención<br/>3. Guidance con perturbación<br/>4. Denoising step
        
        PAG-->>RunPredict: Lista de PIL Images
        deactivate PAG
        
        Note over RunPredict: Inferencia con PAG<br/>~10-14s
        
    else pag_scale = 0
        Note over RunPredict: Usar pipeline estándar
        
        RunPredict->>Pipeline: Ejecutar diffusion estándar
        activate Pipeline
        
        Pipeline->>Pipeline: Construir kwargs
        Note over Pipeline: • prompt_embeds<br/>• negative_prompt_embeds<br/>• pooled_prompt_embeds<br/>• num_inference_steps<br/>• guidance_scale<br/>• guidance_rescale<br/>• width, height<br/>• generator<br/>• clip_skip
        
        Pipeline->>Pipeline: Loop denoising (num_inference_steps)
        Note over Pipeline: Euler Ancestral scheduler<br/>Por cada step:<br/>1. Predict noise<br/>2. CFG (classifier-free guidance)<br/>3. Scheduler step<br/>4. Rescale guidance (si > 0)
        
        Pipeline->>Pipeline: Decodificar latents con VAE
        Note over Pipeline: VAE fp16-fix evita blur/NaN
        
        Pipeline-->>RunPredict: Lista de PIL Images
        deactivate Pipeline
        
        Note over RunPredict: Inferencia estándar<br/>~8-12s
    end
    end
    
    rect rgb(255, 245, 245)
    Note over RunPredict,FaceRefiner: Fase 4: Refinado de caras (opcional)
    
    RunPredict->>RunPredict: ¿face_yolov9c = True?
    
    alt Face refinement activado
        Note over RunPredict: Y self.pipe_inpaint disponible
        
        loop Para cada imagen generada
            RunPredict->>FaceRefiner: refine_faces(pipe_inpaint, img, prompt, negative)
            activate FaceRefiner
            
            FaceRefiner->>FaceRefiner: Detectar caras con YOLOv9c
            Note over FaceRefiner: Modelo pre-entrenado<br/>para detección de rostros
            
            alt Caras detectadas
                FaceRefiner->>FaceRefiner: Para cada cara detectada:
                Note over FaceRefiner: 1. Extraer bounding box<br/>2. Crear máscara de inpainting<br/>3. Expandir área (padding)
                
                FaceRefiner->>FaceRefiner: Ejecutar inpainting pipeline
                Note over FaceRefiner: • Usar mismo prompt<br/>• Menos steps (~20)<br/>• Mismo guidance_scale<br/>• Solo área de la cara
                
                FaceRefiner->>FaceRefiner: Compositar cara refinada
                Note over FaceRefiner: Blend con imagen original<br/>usando máscara
                
                Note over FaceRefiner: +2-4s por cara detectada
                
            else No se detectaron caras
                Note over FaceRefiner: Retornar imagen sin cambios
            end
            
            FaceRefiner-->>RunPredict: Imagen refinada
            deactivate FaceRefiner
        end
        
        Note over RunPredict: Face refinement completado
        
    else Face refinement desactivado
        Note over RunPredict: Omitir refinado<br/>(face_yolov9c = False<br/>o pipe_inpaint = None)
    end
    end
    
    rect rgb(248, 248, 255)
    Note over RunPredict: Fase 5: Serialización y métricas
    
    loop Para cada PIL Image
        RunPredict->>RunPredict: Crear BytesIO buffer
        RunPredict->>RunPredict: img.save(buf, format="PNG")
        RunPredict->>RunPredict: Agregar bytes a lista
    end
    
    Note over RunPredict: Todas las imágenes → PNG bytes
    
    RunPredict->>RunPredict: Calcular tiempos
    Note over RunPredict: inference_seconds = time.perf_counter() - t0_infer<br/>self._inference_times.append(inference_seconds)<br/>cold_start_for_request = self._cold_start_seconds if is_first else 0
    
    RunPredict->>RunPredict: Incrementar contador
    Note over RunPredict: self._request_count = request_num + 1
    
    RunPredict->>Volume: Escribir timing report
    activate Volume
    Note over Volume: Formato:<br/>"YYYY-MM-DDTHH:MM:SSZ request=N<br/>cold_start_s=X.XX inference_s=Y.YY"
    Volume-->>RunPredict: OK
    deactivate Volume
    
    RunPredict->>Volume: Commit volume
    activate Volume
    Volume-->>RunPredict: Cache persistido
    deactivate Volume
    end
    
    RunPredict-->>NovaModel: (lista_bytes, inference_seconds, cold_start_seconds)
    deactivate RunPredict
    
    NovaModel->>NovaModel: Construir dict de timings
    Note over NovaModel: timings = {<br/>  "inference_seconds": round(inference_s, 2),<br/>  "cold_start_seconds": round(cold_s, 2),<br/>  "request_number": self._request_count<br/>}
    
    alt predict_one()
        NovaModel-->>Caller: (bytes_primera_imagen, timings)
    else predict()
        NovaModel-->>Caller: (lista_completa_bytes, timings)
    end
    
    deactivate NovaModel
```

## Fases del Proceso

### Fase 1: Construcción de Prompts (~1ms)

Transforma el prompt del usuario en prompt completo:

**Con prepend_preprompt=True (default):**
```python
# Input usuario
prompt = "1girl, cherry blossoms"

# Output procesado
full_prompt = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, ..., 1girl, cherry blossoms"
full_negative = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, ..., nsfw, naked"
```

**Formato Replicate:**
- Preprompt positivo: Mejora calidad general
- Preprompt negativo: Evita defectos comunes

### Fase 2: Codificación de Prompts (~50-500ms)

Convierte texto en embeddings para el modelo:

**Estrategia dual:**

1. **Compel (rápido):** Para prompts ≤77 tokens
   - Soporta pesos: `(word:1.2)`, `[word]`
   - Dual text encoders (SDXL)
   - Pooled embeddings para conditioning

2. **Chunking (fallback):** Para prompts largos
   - Divide en chunks de 77 tokens
   - Procesa cada chunk
   - Concatena embeddings
   - Aplica clip_skip si especificado

**Clip Skip:**
```python
clip_skip = 2  # Omitir últimas 2 capas del text encoder
# Útil para estilos de anime (menos fotorealismo)
```

**Batch Processing:**
```python
# Para num_outputs=4
prompt_embeds.repeat(4, 1, 1)  # [1, 77, 2048] → [4, 77, 2048]
```

### Fase 3: Generación con Diffusion (~8-14s)

Proceso de denoising iterativo:

**Pipeline Estándar:**
```python
# Euler Ancestral scheduler
for step in range(num_inference_steps):
    # 1. Predict noise
    noise_pred = unet(latents, timestep, prompt_embeds)
    
    # 2. Classifier-Free Guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # 3. Guidance Rescale (evita oversaturation)
    if guidance_rescale > 0:
        noise_pred = rescale_noise_cfg(noise_pred, guidance_rescale)
    
    # 4. Scheduler step (denoise)
    latents = scheduler.step(noise_pred, timestep, latents).prev_sample
```

**Con PAG (Perturbed-Attention Guidance):**
```python
# PAG mejora estructura perturbando atención
for step in range(num_inference_steps):
    # Self-attention en mid block
    attn_output = self_attention(hidden_states)
    
    # Perturbar atención
    perturbed_attn = perturb_attention(attn_output, pag_scale)
    
    # Guidance con perturbación
    noise_pred = unet_with_pag(latents, timestep, prompt_embeds, perturbed_attn)
    
    # Denoising step
    latents = scheduler.step(noise_pred, timestep, latents).prev_sample
```

**Parámetros clave:**

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `num_inference_steps` | 25 | Más steps = mejor calidad, más lento |
| `guidance_scale` | 6.0 | CFG strength (7-9 = más fiel al prompt) |
| `guidance_rescale` | 1.0 | Evita oversaturation (0=off) |
| `pag_scale` | 0.0 | PAG strength (0=off, 0.3-0.5 recomendado) |

### Fase 4: Face Refinement (~2-4s por cara)

Mejora detalles de rostros con inpainting:

**Proceso:**

1. **Detección con YOLOv9c:**
   ```python
   faces = detect_faces(image)  # [(x, y, w, h), ...]
   ```

2. **Para cada cara:**
   ```python
   # Extraer bounding box con padding
   x1, y1, x2, y2 = expand_bbox(face, padding=0.2)
   
   # Crear máscara
   mask = create_inpaint_mask(x1, y1, x2, y2)
   
   # Inpainting
   refined_face = inpaint_pipeline(
       image=image,
       mask_image=mask,
       prompt=full_prompt,
       negative_prompt=full_negative,
       num_inference_steps=20,  # Menos steps que generación inicial
       guidance_scale=guidance_scale,
       strength=0.4  # Cuánto cambiar (0=nada, 1=completamente nuevo)
   )
   
   # Compositar
   image = blend(image, refined_face, mask)
   ```

3. **Resultado:**
   - Caras más detalladas
   - Mejor anatomía facial
   - Corrección de artefactos

**Costo:**
- +2-4 segundos por cara detectada
- Depende de tamaño de cara y num_inference_steps

**Desactivar:**
```python
# En llamada
predict_one(prompt, face_yolov9c=False)

# Globalmente (variable de entorno)
ENABLE_FACE_REFINEMENT=0
```

### Fase 5: Serialización y Métricas (~100-200ms)

Conversión final y registro:

**Serialización:**
```python
for img in images:
    buf = BytesIO()
    img.save(buf, format="PNG")  # PIL Image → PNG bytes
    out.append(buf.getvalue())
```

**Métricas:**
```python
timings = {
    "inference_seconds": 10.23,      # Tiempo de generación
    "cold_start_seconds": 0.0,       # 0 si warm, >0 si primer request
    "request_number": 5              # Contador de requests del contenedor
}
```

**Persistencia:**
```python
# Escribir en /cache/timing_report.txt
timestamp = "2026-02-07T14:30:22Z"
line = f"{timestamp} request=5 cold_start_s=0.00 inference_s=10.23\n"
with open("/cache/timing_report.txt", "a") as f:
    f.write(line)
```

## Tiempos por Fase

| Fase | Tiempo típico | Configurable |
|------|---------------|--------------|
| 1. Construcción prompts | ~1ms | prepend_preprompt |
| 2. Codificación | ~50-500ms | - |
| 3. Diffusion estándar | ~8-12s | steps, guidance_scale |
| 3. Diffusion con PAG | ~10-14s | pag_scale |
| 4. Face refinement | +2-4s/cara | face_yolov9c |
| 5. Serialización | ~100-200ms | - |
| **Total** | **~8-18s** | - |

## Parámetros de Entrada

### Obligatorios
- `prompt` (str): Descripción de la imagen

### Opcionales (con defaults)

```python
prepend_preprompt: bool = True          # Agregar preprompt de calidad
negative_prompt: str = "nsfw, naked"    # Prompt negativo
num_inference_steps: int = 25           # Pasos de diffusion
guidance_scale: float = 6.0             # CFG scale
guidance_rescale: float = 1.0           # Rescale para evitar oversaturation
clip_skip: int = 2                      # Omitir capas del text encoder
pag_scale: float = 0.0                  # PAG strength (0=off)
width: int = 1024                       # Ancho en pixels
height: int = 1024                      # Alto en pixels
seed: int = None                        # Seed para reproducibilidad
num_outputs: int = 1                    # Número de imágenes (1-4)
face_yolov9c: bool = True              # Activar face refinement
hand_yolov9c: bool = False             # (Aceptado pero no implementado)
person_yolov8m_seg: bool = False       # (Aceptado pero no implementado)
```

## Salida

### predict_one()
```python
(bytes, timings) = predict_one(prompt)
```

**Retorna:**
- `bytes`: PNG bytes de una imagen
- `timings`: Dict con métricas

### predict()
```python
(lista_bytes, timings) = predict(prompt, num_outputs=4)
```

**Retorna:**
- `lista_bytes`: Lista de PNG bytes (1-4 imágenes)
- `timings`: Dict con métricas

## Optimizaciones

### Para velocidad máxima
```python
predict_one(
    prompt="...",
    num_inference_steps=20,      # Reducir steps
    face_yolov9c=False,          # Desactivar face refine
    pag_scale=0.0,               # Desactivar PAG
    prepend_preprompt=False      # Omitir preprompt (menos tokens)
)
# ~6-8 segundos
```

### Para calidad máxima
```python
predict_one(
    prompt="...",
    num_inference_steps=30,      # Más steps
    guidance_scale=7.0,          # CFG más alto
    pag_scale=0.3,               # Activar PAG
    face_yolov9c=True,           # Face refinement
    prepend_preprompt=True       # Preprompt de calidad
)
# ~14-20 segundos
```

### Para reproducibilidad
```python
predict_one(
    prompt="...",
    seed=42,                     # Seed fijo
    num_inference_steps=25,      # Steps fijos
    # ... otros parámetros fijos
)
# Misma imagen cada vez
```

## Troubleshooting

### Imagen borrosa
**Causa:** VAE issue o steps insuficientes
**Solución:**
- VAE fp16-fix está activado (automático)
- Aumentar `num_inference_steps` a 30
- Verificar `guidance_scale` (6-7 recomendado)

### Imagen sobre-saturada
**Causa:** `guidance_scale` muy alto
**Solución:**
- Reducir `guidance_scale` a 5-6
- Activar `guidance_rescale=1.0`

### Anatomía incorrecta
**Causa:** Diffusion insuficiente
**Solución:**
- Aumentar `num_inference_steps`
- Activar `pag_scale=0.3` (mejora estructura)
- Activar `face_yolov9c=True`

### Caras borrosas
**Causa:** Face refinement desactivado
**Solución:**
- Activar `face_yolov9c=True`
- Verificar que `ENABLE_FACE_REFINEMENT=1`

### Prompt no sigue instrucciones
**Causa:** `guidance_scale` muy bajo
**Solución:**
- Aumentar `guidance_scale` a 7-9
- Usar pesos en prompt: `(important word:1.3)`
- Verificar que preprompt no interfiere

### Generación muy lenta
**Causa:** Demasiados steps o face refinement
**Solución:**
- Reducir `num_inference_steps` a 20
- Desactivar `face_yolov9c=False`
- Desactivar `pag_scale=0.0`
