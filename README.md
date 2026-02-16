# TelegramChatbotModal

Sistema de generación de imágenes con **Nova Anime IL v5.5** desplegado en [Modal](https://modal.com) con infraestructura serverless y GPU en la nube.

## Descripción

Este proyecto permite generar imágenes de anime de alta calidad utilizando el modelo Nova Anime IL v5.5. Las imágenes se generan en la infraestructura de Modal (con GPU) y se descargan automáticamente a la carpeta `outputs/` del proyecto local.

### Características

- Generación de imágenes con IA usando Stable Diffusion
- Infraestructura serverless con Modal (sin necesidad de GPU local)
- API REST opcional para integración con otros servicios
- Soporte para prompts personalizados y control de seeds
- Descarga automática de imágenes generadas

---

## Requisitos Previos

- **Python 3.10 o superior**
- **Git**
- Cuenta activa en [Modal](https://modal.com) (plan gratuito disponible)
- Token de API de CivitAI o Hugging Face (para descargar el modelo)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd TelegramChatbotModal
```

### 2. Configurar entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# En Windows (CMD):
.\.venv\Scripts\activate.bat

# En Linux/macOS:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Modal

```bash
# Configurar credenciales de Modal (abre el navegador para autenticación)
modal setup
```

Este comando abrirá tu navegador para iniciar sesión en Modal y guardará el token de autenticación localmente.

---

## Configuración del Modelo

El proyecto requiere el checkpoint de **Nova Anime IL v5.5**. Configura el acceso al modelo mediante Secrets en Modal.

### Opción 1: CivitAI (Recomendado)

1. Ve a [Modal Secrets Dashboard](https://modal.com/secrets)
2. Crea un nuevo Secret con el nombre que prefieras
3. Añade las siguientes variables:
   - `CHECKPOINT_URL`: `https://civitai.com/api/download/models/1500882`
   - `CIVITAI_API_KEY`: Tu [token de CivitAI](https://civitai.com/user/account)

### Opción 2: Hugging Face

1. Ve a [Modal Secrets Dashboard](https://modal.com/secrets)
2. Crea un nuevo Secret con:
   - `HF_TOKEN`: Tu token de Hugging Face
   - `NOVA_ANIME_HF_ID`: `usuario/repositorio-del-checkpoint`

Para más detalles sobre la configuración del checkpoint, paridad con Replicate y ajustes, ver `docs/nova-anime.md`.

---

## Uso

### Generación Simple

Genera una imagen con un prompt personalizado:

```bash
modal run app.py --prompt "score_9, masterpiece, 1girl, cherry blossoms"
```

La imagen se guardará en `outputs/output_YYYYMMDD_HHMMSS.png`

### Generación con Seed

Para resultados reproducibles, especifica un seed:

```bash
modal run app.py --prompt "score_9, 1boy, samurai armor" --seed 42
```

### Carpeta de Salida Personalizada

Guarda las imágenes en una ubicación diferente:

```bash
modal run app.py --prompt "tu prompt" --save-dir ./mis_imagenes
```

### Nota Importante

Siempre ejecuta los comandos desde la **raíz del proyecto** (`TelegramChatbotModal/`).

**Importante:** Usa `modal run`, **NO** `python`. El modelo se ejecuta en la nube de Modal con GPU. Si ejecutas `python app.py` directamente, el script intentará ejecutarse localmente (sin GPU) y fallará o será muy lento.

---

## API REST (Opcional)

Puedes exponer un endpoint HTTP para integrar la generación de imágenes con otros servicios.

### Iniciar el servidor

```bash
modal serve app.py
```

El comando mostrará una URL pública (ej: `https://tu-usuario--nova-anime-predict.modal.run`)

### Ejemplo de uso

```bash
curl -X POST "https://TU-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "score_9, masterpiece, 1girl, cherry blossoms", "seed": 42}' \
  --output imagen.png
```

### Parámetros del API

- `prompt` (string, requerido): Descripción de la imagen a generar
- `seed` (integer, opcional): Semilla para reproducibilidad
- `num_inference_steps` (integer, opcional): Número de pasos de inferencia
- `guidance_scale` (float, opcional): Escala de guiado del modelo

---

## Estructura del Proyecto

```
TelegramChatbotModal/
├── app.py                    # Entry point Modal (modal run / deploy)
├── requirements.txt          # Dependencias mínimas para ejecutar localmente (modal)
├── README.md                 # Documentación principal
├── api/                      # FastAPI endpoints y schemas
├── config/                   # Infraestructura Modal + constantes
├── core/                     # Checkpoint, prompt encoder, face refinement
├── model/                    # NovaAnimeModel (GPU class)
├── docs/                     # Diagramas Mermaid + documentación técnica
├── outputs/                  # Imágenes generadas (output_YYYYMMDD_HHMMSS.png)
└── test/                     # Scripts de prueba / simulación
```

---

## Troubleshooting

### Error: "No module named 'modal'"

Asegúrate de haber instalado las dependencias:

```bash
pip install -r requirements.txt
```

### Error: "CUDA not available"

Este error aparece si ejecutas el script con `python` en lugar de `modal run`. Recuerda que el modelo debe ejecutarse en Modal con GPU:

```bash
modal run app.py --prompt "tu prompt"
```

### Error: "Secret not found"

Verifica que hayas configurado correctamente el Secret en [Modal Dashboard](https://modal.com/secrets) con las variables `CHECKPOINT_URL` y `CIVITAI_API_KEY`.

---

## Documentación Adicional

Para información detallada sobre:
- Parámetros avanzados del modelo
- Especificaciones técnicas
- Licencia y términos de uso
- Configuración avanzada de la API

Consulta **`docs/nova-anime.md`** y la carpeta **`docs/`**.

---

## Licencia

Consulta el archivo de licencia correspondiente al modelo Nova Anime IL v5.5 en CivitAI.

---

## Soporte

Para problemas o preguntas:
1. Revisa la sección de Troubleshooting
2. Consulta la documentación de [Modal](https://modal.com/docs)
3. Revisa los issues del repositorio
