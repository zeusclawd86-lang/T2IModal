# Flujo de Ejecución: Modal Run (CLI Local)

Este diagrama muestra el flujo completo cuando un usuario ejecuta `modal run app.py --prompt "..."` desde la terminal.

## Diagrama de Secuencia

```mermaid
sequenceDiagram
    actor Usuario
    participant CLI as Modal CLI
    participant LocalEntry as app.main()
    participant ModalCloud as Modal Cloud
    participant NovaModel as NovaAnimeModel<br/>(GPU Container)
    participant Volume as Modal Volume<br/>(Cache)
    participant LocalFS as Sistema de Archivos<br/>(outputs/)

    Usuario->>CLI: modal run app.py --prompt "1girl..."
    activate CLI
    
    CLI->>LocalEntry: Ejecutar @local_entrypoint
    activate LocalEntry
    
    Note over LocalEntry: Parsea argumentos:<br/>prompt, steps, cfg_scale,<br/>seed, save_dir, etc.
    
    LocalEntry->>ModalCloud: Conectar con Modal
    activate ModalCloud
    
    ModalCloud->>NovaModel: predict_one.remote(prompt, **kwargs)
    activate NovaModel
    
    Note over NovaModel: Primera llamada?<br/>Si: ejecutar @modal.enter()
    
    alt Cold Start (Primera llamada)
        NovaModel->>Volume: Buscar checkpoint en cache
        activate Volume
        Volume-->>NovaModel: checkpoint.safetensors (o no existe)
        deactivate Volume
        
        alt Checkpoint no existe en cache
            NovaModel->>NovaModel: get_checkpoint_path()
            Note over NovaModel: Descargar desde:<br/>CHECKPOINT_URL o<br/>HuggingFace
            NovaModel->>Volume: Guardar checkpoint
            activate Volume
            Volume-->>NovaModel: OK
            deactivate Volume
        end
        
        NovaModel->>NovaModel: Cargar VAE (fp16-fix)
        NovaModel->>NovaModel: Cargar StableDiffusionXLPipeline
        NovaModel->>NovaModel: Cargar Compel (prompt encoder)
        NovaModel->>NovaModel: Cargar PAG pipeline
        NovaModel->>NovaModel: Cargar Inpaint pipeline (face refine)
        NovaModel->>Volume: Commit cache
        activate Volume
        Volume-->>NovaModel: Cache persistido
        deactivate Volume
        
        Note over NovaModel: Cold start: ~30-60s<br/>(solo primera vez)
    end
    
    Note over NovaModel: Iniciar inferencia
    
    NovaModel->>NovaModel: Construir prompt completo<br/>(prepend_preprompt?)
    NovaModel->>NovaModel: Codificar con Compel<br/>(o chunking si >77 tokens)
    NovaModel->>NovaModel: Generar embeddings positivos/negativos
    
    NovaModel->>NovaModel: Seleccionar pipeline<br/>(PAG si pag_scale > 0)
    NovaModel->>NovaModel: Ejecutar diffusion<br/>(num_inference_steps)
    
    Note over NovaModel: Inferencia: ~8-12s
    
    alt Face Refinement activado
        NovaModel->>NovaModel: Detectar caras (YOLOv9c)
        NovaModel->>NovaModel: Refinar con inpainting
        Note over NovaModel: +2-4s por cara detectada
    end
    
    NovaModel->>NovaModel: Convertir PIL Image → PNG bytes
    NovaModel->>NovaModel: Registrar timing en memoria
    NovaModel->>Volume: Escribir timing_report.txt
    activate Volume
    Volume-->>NovaModel: OK
    deactivate Volume
    
    NovaModel-->>ModalCloud: (bytes, timings_dict)
    deactivate NovaModel
    
    ModalCloud-->>LocalEntry: Recibir imagen + timings
    deactivate ModalCloud
    
    Note over LocalEntry: Crear timestamp:<br/>YYYYMMDD_HHMMSS
    
    LocalEntry->>LocalFS: Crear directorio outputs/ si no existe
    activate LocalFS
    LocalFS-->>LocalEntry: OK
    deactivate LocalFS
    
    LocalEntry->>LocalFS: Escribir output_YYYYMMDD_HHMMSS.png
    activate LocalFS
    LocalFS-->>LocalEntry: Archivo guardado
    deactivate LocalFS
    
    LocalEntry->>CLI: Imprimir resultado en consola
    Note over LocalEntry: "Guardado en outputs/output_20260207_143022.png"<br/>"Tiempos: cold_start=0s inferencia=10.5s"
    
    alt show_report=True
        LocalEntry->>ModalCloud: get_timing_report.remote()
        activate ModalCloud
        ModalCloud->>NovaModel: Leer timing_report.txt
        activate NovaModel
        NovaModel->>Volume: Leer /cache/timing_report.txt
        activate Volume
        Volume-->>NovaModel: Contenido completo
        deactivate Volume
        NovaModel-->>ModalCloud: Reporte formateado
        deactivate NovaModel
        ModalCloud-->>LocalEntry: Reporte de tiempos
        deactivate ModalCloud
        LocalEntry->>CLI: Imprimir reporte
        Note over CLI: "=== Reporte de tiempos ===<br/>Total requests: 5<br/>Inferencia: avg=10.2s"
    end
    
    deactivate LocalEntry
    CLI-->>Usuario: Comando completado ✓
    deactivate CLI
```

## Detalles del Flujo

### 1. Inicio del Comando
- Usuario ejecuta `modal run` con argumentos
- Modal CLI parsea los argumentos y los pasa a `main()`

### 2. Local Entrypoint
- Función decorada con `@app.local_entrypoint()`
- Recibe argumentos: `prompt`, `steps`, `cfg_scale`, `seed`, `save_dir`, etc.
- Convierte argumentos de CLI a parámetros del modelo

### 3. Llamada Remota
- `NovaAnimeModel().predict_one.remote()` ejecuta en GPU remota
- Modal serializa los argumentos y los envía al contenedor GPU

### 4. Cold Start (Solo primera vez)
- **Duración:** ~30-60 segundos
- Descarga checkpoint si no está en cache (~6.46 GB)
- Carga VAE, pipeline SDXL, Compel, PAG, inpaint
- Persiste cache en Modal Volume

### 5. Inferencia
- **Duración:** ~8-12 segundos
- Codificación de prompts con Compel
- Generación con diffusion pipeline
- Refinado de caras (opcional, +2-4s)

### 6. Descarga y Guardado
- Imagen se recibe como bytes PNG
- Se guarda en `outputs/output_YYYYMMDD_HHMMSS.png`
- Se imprimen métricas de tiempo

### 7. Reporte (Opcional)
- Si `--show-report`, obtiene estadísticas acumuladas
- Muestra cold start, tiempos promedio/min/max de inferencia

## Argumentos Disponibles

```bash
modal run app.py \
  --prompt "score_9, masterpiece, 1girl, cherry blossoms" \
  --prepend-preprompt true \
  --negative-prompt "nsfw, naked" \
  --steps 25 \
  --cfg-scale 6.0 \
  --guidance-rescale 1.0 \
  --clip-skip 2 \
  --seed 42 \
  --save-dir ./outputs \
  --show-report true
```

## Tiempos Típicos

| Fase | Primera ejecución | Ejecuciones siguientes |
|------|-------------------|------------------------|
| Cold start | ~30-60s | ~0s (contenedor warm) |
| Inferencia | ~8-12s | ~8-12s |
| Face refine | +2-4s por cara | +2-4s por cara |
| Descarga | ~0.5s | ~0.5s |
| **Total** | **~40-75s** | **~8-15s** |

## Optimizaciones

1. **Cache persistente:** Checkpoint se descarga solo una vez
2. **Warm containers:** Modal mantiene contenedores activos (~5 min)
3. **Compel:** Codificación rápida de prompts
4. **PAG opcional:** Solo se usa si `pag_scale > 0`
5. **Face refine configurable:** Puede desactivarse con variable de entorno
