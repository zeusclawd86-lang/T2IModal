# Flujo de Resolución del Checkpoint

Este diagrama muestra el algoritmo de resolución del checkpoint del modelo Nova Anime IL v5.5.

## Diagrama de Secuencia

```mermaid
sequenceDiagram
    participant Caller as NovaAnimeModel.load()
    participant Function as get_checkpoint_path()
    participant FileSystem as File System
    participant Volume as Modal Volume<br/>(/cache)
    participant CivitAI as CivitAI API
    participant HF as Hugging Face Hub

    Caller->>Function: get_checkpoint_path()
    activate Function
    
    Note over Function: Estrategia de resolución<br/>con 5 niveles de fallback
    
    rect rgb(240, 248, 255)
    Note over Function,FileSystem: Nivel 1: Checkpoint embebido en imagen
    Function->>FileSystem: ¿Existe LOCAL_CHECKPOINT_IN_IMAGE?<br/>(/local/model.safetensors)
    activate FileSystem
    
    alt Archivo existe
        FileSystem-->>Function: True
        deactivate FileSystem
        Note over Function: ✓ Estrategia más rápida<br/>~0ms overhead
        Function-->>Caller: "/local/model.safetensors"
    else Archivo no existe
        FileSystem-->>Function: False
        deactivate FileSystem
    end
    end
    
    rect rgb(255, 250, 240)
    Note over Function: Nivel 2: Variable de entorno CHECKPOINT_PATH
    Function->>Function: ¿Existe env var CHECKPOINT_PATH?
    
    alt CHECKPOINT_PATH definida
        Function->>FileSystem: ¿Existe path?
        activate FileSystem
        
        alt Path existe
            FileSystem-->>Function: True
            deactivate FileSystem
            Note over Function: ✓ Ruta explícita<br/>~0ms overhead
            Function-->>Caller: os.environ["CHECKPOINT_PATH"]
        else Path no existe
            FileSystem-->>Function: False
            deactivate FileSystem
            Note over Function: Path inválido,<br/>continuar a nivel 3
        end
    end
    end
    
    rect rgb(240, 255, 240)
    Note over Function,Volume: Nivel 3: Volume cache (reutilización)
    Function->>FileSystem: mkdir -p /cache/checkpoint
    activate FileSystem
    FileSystem-->>Function: OK
    deactivate FileSystem
    
    Function->>Volume: ¿Existe /cache/checkpoint/model.safetensors?
    activate Volume
    
    alt Checkpoint cacheado
        Volume-->>Function: True
        deactivate Volume
        Note over Function: ✓ Cache hit<br/>~0ms overhead<br/>Checkpoint de cold start previo
        Function-->>Caller: "/cache/checkpoint/model.safetensors"
    else Cache miss
        Volume-->>Function: False
        deactivate Volume
        Note over Function: Primera ejecución o cache limpio<br/>Proceder a descarga
    end
    end
    
    rect rgb(255, 245, 245)
    Note over Function,CivitAI: Nivel 4: CHECKPOINT_URL (descarga)
    Function->>Function: ¿Existe env var CHECKPOINT_URL?
    
    alt CHECKPOINT_URL definida
        Note over Function: URL configurada<br/>Proceder a descarga
        
        Function->>Function: ¿URL contiene "civitai.com"?
        
        alt Es CivitAI
            Note over Function: Autenticación requerida<br/>para modelos restringidos
            
            Function->>Function: Leer CIVITAI_API_KEY/CIVITAI_TOKEN
            
            alt API Key disponible
                Function->>Function: Parsear URL
                Note over Function: Agregar ?token=XXX<br/>+ Header "Authorization: Bearer XXX"
                
                Function->>CivitAI: GET /api/download/models/1500882<br/>?token=XXX<br/>Authorization: Bearer XXX
                activate CivitAI
                
                alt Token válido
                    CivitAI-->>Function: 200 OK<br/>Stream: 6.46 GB
                    Note over Function: Descarga en progreso<br/>~2-5 minutos<br/>(depende de ancho de banda)
                    
                    Function->>Volume: Escribir bytes a<br/>/cache/checkpoint/model.safetensors
                    activate Volume
                    Volume-->>Function: Escritura completada
                    deactivate Volume
                    
                    deactivate CivitAI
                    
                    Note over Function: ✓ Descarga exitosa<br/>Cache guardado para futuros usos
                    Function-->>Caller: "/cache/checkpoint/model.safetensors"
                    
                else Token inválido (403 Forbidden)
                    CivitAI-->>Function: 403 Forbidden<br/>Error: Invalid token
                    deactivate CivitAI
                    
                    Function->>Function: Lanzar RuntimeError
                    Note over Function: Error detallado:<br/>"CivitAI devolvió 403 Forbidden.<br/>Comprueba CIVITAI_API_KEY..."
                    Function-->>Caller: RuntimeError
                    
                else Otro error HTTP
                    CivitAI-->>Function: Error HTTP
                    deactivate CivitAI
                    Function-->>Caller: Propagar excepción
                end
                
            else API Key no disponible
                Note over Function: Intentar descarga sin auth<br/>(puede fallar si modelo requiere login)
                
                Function->>CivitAI: GET /api/download/models/1500882
                activate CivitAI
                
                alt Modelo público
                    CivitAI-->>Function: 200 OK<br/>Stream: 6.46 GB
                    deactivate CivitAI
                    Function->>Volume: Escribir a cache
                    activate Volume
                    Volume-->>Function: OK
                    deactivate Volume
                    Function-->>Caller: "/cache/checkpoint/model.safetensors"
                else Modelo requiere auth
                    CivitAI-->>Function: 401/403 Error
                    deactivate CivitAI
                    Function-->>Caller: Error HTTP
                end
            end
            
        else URL custom (S3, GCS, R2, HTTP)
            Note over Function: URL genérica<br/>Sin autenticación
            
            Function->>Function: urllib.request.urlretrieve(url, cached)
            Note over Function: Descarga directa<br/>~2-5 minutos
            
            Function->>Volume: Guardar en cache
            activate Volume
            Volume-->>Function: OK
            deactivate Volume
            
            Function-->>Caller: "/cache/checkpoint/model.safetensors"
        end
        
    end
    end
    
    rect rgb(248, 248, 255)
    Note over Function,HF: Nivel 5: HuggingFace (fallback público)
    
    Note over Function: Ninguna estrategia previa funcionó<br/>Usar HuggingFace por defecto
    
    Function->>Function: Leer NOVA_ANIME_HF_ID
    Note over Function: Default: "John6666/nova-anime-xl-il-v80-sdxl"<br/>⚠️ Este es IL v8.0, no v5.5
    
    Function->>Function: Leer NOVA_ANIME_HF_FILENAME
    
    alt NOVA_ANIME_HF_FILENAME especificado
        Note over Function: Filename explícito<br/>Ej: "model.safetensors"
        
        Function->>HF: hf_hub_download(<br/>  repo_id=NOVA_ANIME_HF_ID,<br/>  filename=NOVA_ANIME_HF_FILENAME,<br/>  local_dir=/cache/checkpoint<br/>)
        activate HF
        
        alt Archivo existe en repo
            HF-->>Function: Descarga completada<br/>Path: /cache/checkpoint/model.safetensors
            deactivate HF
            Function-->>Caller: "/cache/checkpoint/model.safetensors"
        else Archivo no existe
            HF-->>Function: 404 Not Found
            deactivate HF
            Function-->>Caller: Propagar excepción
        end
        
    else NOVA_ANIME_HF_FILENAME no especificado
        Note over Function: Auto-detectar primer .safetensors
        
        Function->>HF: list_repo_files(repo_id)
        activate HF
        HF-->>Function: Lista de archivos del repo
        deactivate HF
        
        Function->>Function: Filtrar archivos *.safetensors
        
        alt Hay archivos .safetensors
            Note over Function: Usar primer .safetensors<br/>encontrado
            
            Function->>HF: hf_hub_download(<br/>  repo_id=NOVA_ANIME_HF_ID,<br/>  filename=safetensors[0],<br/>  local_dir=/cache/checkpoint<br/>)
            activate HF
            HF-->>Function: Descarga completada
            deactivate HF
            
            Function->>Volume: Guardar en cache
            activate Volume
            Volume-->>Function: OK
            deactivate Volume
            
            Note over Function: ✓ Descarga desde HF exitosa<br/>⚠️ Verificar que sea la versión correcta
            Function-->>Caller: "/cache/checkpoint/..."
            
        else No hay .safetensors
            Note over Function: Repo no contiene<br/>archivos .safetensors
            
            Function->>Function: Lanzar RuntimeError
            Note over Function: "No .safetensors in {hf_id}.<br/>Set CHECKPOINT_URL or<br/>NOVA_ANIME_HF_FILENAME."
            Function-->>Caller: RuntimeError
        end
    end
    end
    
    deactivate Function
```

## Estrategias de Resolución

### Nivel 1: Checkpoint Embebido en Imagen Docker

**Path:** `/local/model.safetensors`

**Ventajas:**
- Máxima velocidad (~0ms overhead)
- No requiere descarga
- No depende de servicios externos

**Desventajas:**
- Aumenta tamaño de imagen Docker (~6.5 GB)
- Dificulta actualización del modelo
- Aumenta tiempo de build de imagen

**Configuración:**
```python
# En infrastructure.py
image = modal.Image.debian_slim().copy_local_file(
    "/path/local/model.safetensors",
    "/local/model.safetensors"
)
```

### Nivel 2: Variable de Entorno CHECKPOINT_PATH

**Configuración:**
```bash
export CHECKPOINT_PATH="/custom/path/model.safetensors"
```

**Ventajas:**
- Ruta explícita
- Control total sobre ubicación
- Útil para montar desde otros volumes

**Desventajas:**
- Requiere que el archivo exista en el path especificado
- Menos flexible que otras opciones

**Uso típico:**
Montar checkpoint desde otro Modal Volume:
```python
@app.cls(
    volumes={
        "/cache": modal.Volume.from_name("nova-anime-cache"),
        "/models": modal.Volume.from_name("shared-models")
    }
)
```

### Nivel 3: Volume Cache (Reutilización)

**Path:** `/cache/checkpoint/model.safetensors`

**Ventajas:**
- Descarga solo una vez
- Reutilización automática en cold starts
- No requiere configuración adicional

**Funcionamiento:**
1. Primera ejecución: descarga de niveles 4-5
2. Guarda en volume persistente
3. Ejecuciones siguientes: cache hit (~0ms)

**Limpieza manual:**
```python
# Limpiar cache si necesitas re-descargar
volume = modal.Volume.from_name("nova-anime-cache")
volume.remove_file("/cache/checkpoint/model.safetensors", recursive=False)
```

### Nivel 4: CHECKPOINT_URL (Descarga desde URL)

**Configuración Modal Secret:**
```bash
# En Modal Dashboard → Secrets
CHECKPOINT_URL="https://civitai.com/api/download/models/1500882"
CIVITAI_API_KEY="tu_token_aqui"
```

**Opción A: CivitAI (Recomendado)**
- URL: `https://civitai.com/api/download/models/1500882`
- Requiere: `CIVITAI_API_KEY`
- Modelo: Nova Anime XL IL v5B (exacto de Replicate)
- Tamaño: ~6.46 GB

**Obtener API Key:**
1. Ir a https://civitai.com/user/account
2. Click "Create API Key"
3. Copiar token
4. Añadir a Modal Secret como `CIVITAI_API_KEY`

**Opción B: URL Custom (S3, GCS, R2)**
```bash
CHECKPOINT_URL="https://s3.amazonaws.com/my-bucket/nova-anime-v5.5.safetensors"
```

**Ventajas:**
- Control total sobre hosting
- Mejor latencia (elegir región)
- Sin límites de rate de API externas

**Opción C: Cloudflare R2 (Recomendado para producción)**
```bash
CHECKPOINT_URL="https://pub-xxxxx.r2.dev/model.safetensors"
```

**Ventajas:**
- Sin costos de egress
- CDN global
- Alta disponibilidad

### Nivel 5: HuggingFace (Fallback Público)

**Configuración:**
```bash
# Opcional
NOVA_ANIME_HF_ID="usuario/repo-nombre"
NOVA_ANIME_HF_FILENAME="model.safetensors"  # Opcional
HF_TOKEN="hf_xxxx"  # Solo para repos privados
```

**Default:**
- Repo: `John6666/nova-anime-xl-il-v80-sdxl`
- ⚠️ Versión: IL v8.0 (no v5.5)

**Para usar IL v5.5 desde HF:**
1. Subir checkpoint a tu repo privado HF
2. Configurar `NOVA_ANIME_HF_ID="tu-usuario/nova-anime-v5.5"`
3. Configurar `HF_TOKEN` en Modal Secret

**Ventajas:**
- Hosting gratuito
- Alta disponibilidad
- Cache automático de HF

**Desventajas:**
- Repo default es v8.0, no v5.5
- Rate limits en API
- Requiere upload manual del checkpoint correcto

## Tabla Comparativa

| Estrategia | Cold Start | Setup | Mantenimiento | Recomendado para |
|------------|------------|-------|---------------|------------------|
| **Embebido** | ~30s | Alto | Alto | Producción estable |
| **CHECKPOINT_PATH** | ~30s | Medio | Bajo | Volumes compartidos |
| **Volume Cache** | ~30s (2da+) | Bajo | Bajo | Desarrollo |
| **CivitAI URL** | ~3-6 min (1ra) | Bajo | Bajo | **Desarrollo (recomendado)** |
| **Custom URL** | ~2-5 min (1ra) | Medio | Bajo | **Producción (recomendado)** |
| **HuggingFace** | ~3-6 min (1ra) | Medio | Medio | Fallback |

## Flujo de Decisión

```
¿Checkpoint embebido en imagen?
├─ Sí → Usar /local/model.safetensors [NIVEL 1] ✓
└─ No
    ├─ ¿Variable CHECKPOINT_PATH definida?
    │   ├─ Sí → Usar ruta de env var [NIVEL 2] ✓
    │   └─ No
    │       ├─ ¿Existe /cache/checkpoint/model.safetensors?
    │       │   ├─ Sí → Usar cache [NIVEL 3] ✓
    │       │   └─ No
    │       │       ├─ ¿Variable CHECKPOINT_URL definida?
    │       │       │   ├─ Sí
    │       │       │   │   ├─ ¿URL de CivitAI?
    │       │       │   │   │   ├─ Sí
    │       │       │   │   │   │   ├─ ¿CIVITAI_API_KEY disponible?
    │       │       │   │   │   │   │   ├─ Sí → Descargar con auth [NIVEL 4A] ✓
    │       │       │   │   │   │   │   └─ No → Descargar sin auth (puede fallar)
    │       │       │   │   │   └─ No → Descargar desde URL custom [NIVEL 4B] ✓
    │       │       │   └─ No
    │       │       │       └─ Usar HuggingFace [NIVEL 5]
    │       │       │           ├─ ¿NOVA_ANIME_HF_FILENAME especificado?
    │       │       │           │   ├─ Sí → Descargar archivo específico ✓
    │       │       │           │   └─ No → Auto-detectar .safetensors
    │       │       │           │       ├─ ¿Hay .safetensors en repo?
    │       │       │           │       │   ├─ Sí → Descargar primer .safetensors ✓
    │       │       │           │       │   └─ No → RuntimeError ✗
```

## Recomendaciones por Entorno

### Desarrollo Local
```bash
# Modal Secret: nova-anime-checkpoint
CHECKPOINT_URL="https://civitai.com/api/download/models/1500882"
CIVITAI_API_KEY="tu_token_civitai"
```

**Ventajas:**
- Setup rápido (1 minuto)
- Primera descarga lenta (~5 min)
- Cache automático para siguientes ejecuciones

### Producción
**Opción 1: Cloudflare R2**
```bash
# 1. Subir checkpoint a R2
# 2. Generar URL pública
CHECKPOINT_URL="https://pub-xxxxx.r2.dev/nova-anime-v5.5.safetensors"
```

**Opción 2: AWS S3 (misma región que Modal)**
```bash
CHECKPOINT_URL="https://s3.us-east-1.amazonaws.com/bucket/model.safetensors"
```

**Opción 3: Embebido en imagen**
```python
# Solo si el modelo NO cambia frecuentemente
image = image.copy_local_file(
    "model.safetensors",
    "/local/model.safetensors"
)
```

### Testing/CI
```bash
# Usar cache de volume (más rápido después del primer test)
# No configurar CHECKPOINT_URL → usa cache automáticamente
```

## Troubleshooting

### Error: "CivitAI returned 403 Forbidden"

**Causa:** Token inválido o modelo requiere aceptar términos

**Solución:**
1. Verificar `CIVITAI_API_KEY` en Modal Secrets
2. Ir a https://civitai.com/models/376130
3. Aceptar términos del modelo (si es primera vez)
4. Regenerar API key en https://civitai.com/user/account

### Error: "No .safetensors in HuggingFace repo"

**Causa:** Repo no contiene archivos `.safetensors`

**Solución:**
1. Especificar `NOVA_ANIME_HF_FILENAME` explícitamente
2. O cambiar a `CHECKPOINT_URL`

### Download muy lento (>10 min)

**Causa:** Ancho de banda limitado o servidor lento

**Solución:**
1. Usar URL custom en región cercana a Modal
2. Cloudflare R2 (CDN global)
3. Embedder checkpoint en imagen

### Cache no se reutiliza

**Causa:** Volume no se persiste correctamente

**Verificar:**
```python
# En NovaAnimeModel.load()
modal.Volume.from_name("nova-anime-cache").commit()
```

**Solución:**
1. Verificar que volume existe en Modal Dashboard
2. Verificar permisos de escritura
3. Check logs de commit

### Checkpoint corrupto

**Causa:** Descarga interrumpida o archivo dañado

**Solución:**
```python
# Limpiar cache y re-descargar
volume = modal.Volume.from_name("nova-anime-cache")
volume.remove_file("/cache/checkpoint/model.safetensors")
```
