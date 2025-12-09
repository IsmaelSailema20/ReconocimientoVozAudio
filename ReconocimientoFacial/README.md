# 🔐 Sistema de Login con Reconocimiento Facial

Sistema de autenticación biométrica usando Deep Learning (FaceNet + CNN) desarrollado con Django.

## 🎯 Características

- ✅ **Reconocimiento facial en tiempo real** mediante cámara web
- 🔒 **Autenticación biométrica segura** con umbral de confianza del 75%
- 🎨 **Interfaz elegante y profesional** con diseño responsive
- ⚡ **Alta precisión** usando FaceNet (512D embeddings) + CNN personalizada
- 🛡️ **Detección de personas desconocidas** con análisis de confianza avanzado
- 📊 **Dashboard informativo** con detalles de sesión

## 🏗️ Arquitectura del Sistema

### Backend

- **Framework:** Django 5.2.7
- **ML Model:** FaceNet (keras-facenet) + CNN Classifier
- **Computer Vision:** OpenCV (cv2)
- **Face Detection:** Haar Cascade Classifier

### Frontend

- **HTML5/CSS3** con diseño moderno y gradientes
- **JavaScript Vanilla** para captura de cámara
- **Canvas API** para procesamiento de imágenes
- **Fetch API** para comunicación con backend

### Modelo de IA

- **FaceNet preentrenado:** Genera embeddings de 512 dimensiones
- **CNN Clasificadora:** Red neuronal personalizada entrenada con tu dataset
- **Threshold:** 0.75 (75% confianza mínima)
- **Análisis de gap:** Verifica diferencia entre top-1 y top-2 predicciones

## 📁 Estructura del Proyecto

```
ReconocimientoFacial/
├── facial_login_system/          # Configuración Django
│   ├── settings.py
│   └── urls.py
├── face_recognition_app/         # Aplicación principal
│   ├── views.py                 # Lógica de autenticación
│   ├── urls.py                  # Rutas de la app
│   ├── face_recognition_model.py # Sistema de reconocimiento
│   └── templates/
│       └── face_recognition_app/
│           ├── login.html       # Página de login facial
│           └── dashboard.html   # Dashboard de usuario
├── EntrenamientoModelo/
│   └── models/
│       ├── best_classifier_model.h5  # Modelo CNN entrenado
│       └── label_encoder.pkl         # Codificador de nombres
└── manage.py
```

## 🚀 Instalación y Uso

### 1. Activar entorno virtual

```powershell
cd c:\Users\User\Desktop\Projects\ProyectoFinalIA
.\envPROJ\Scripts\Activate.ps1
```

### 2. Instalar dependencias (si no están instaladas)

```powershell
pip install django opencv-python numpy tensorflow keras-facenet pillow
```

### 3. Iniciar servidor Django

```powershell
cd ReconocimientoFacial
python manage.py runserver
```

### 4. Acceder al sistema

Abrir navegador en: **http://127.0.0.1:8000/**

## 🎮 Uso del Sistema

### Login Facial

1. **Permitir acceso a cámara** cuando el navegador lo solicite
2. **Posicionarse frente a la cámara** con buena iluminación
3. **Esperar reconocimiento automático** (cada 2 segundos)
4. Si eres reconocido → **Acceso concedido** (redirect a dashboard)
5. Si no eres reconocido → **Acceso denegado** (persona desconocida)

### Personas Autorizadas

El sistema reconocerá a las personas que están en tu dataset:

- IsmaelSailema
- AlisonSalas
- WilliamTacuri
- (Y cualquier otra persona en `TrainingData/faces/`)

### Cerrar Sesión

Click en el botón **"Cerrar Sesión"** en el dashboard.

## 🔧 API Endpoints

### `POST /api/recognize/`

Reconoce rostro y crea sesión si es autorizado.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response (Autorizado):**

```json
{
  "success": true,
  "authorized": true,
  "name": "IsmaelSailema",
  "confidence": 0.9135,
  "message": "¡Bienvenido/a IsmaelSailema!"
}
```

**Response (No autorizado):**

```json
{
  "success": true,
  "authorized": false,
  "name": "Desconocido",
  "confidence": 0.6349,
  "message": "Acceso denegado. Persona no reconocida."
}
```

### `GET /api/check-session/`

Verifica estado de autenticación.

**Response:**

```json
{
  "authenticated": true,
  "user_name": "IsmaelSailema"
}
```

## 🎨 Diseño UI/UX

### Principios de Diseño Aplicados

1. **Jerarquía Visual:** Títulos grandes, información clara y estructurada
2. **Color Psychology:**

   - Púrpura/Azul → Confianza y tecnología
   - Verde → Éxito y autorización
   - Naranja → Advertencia (desconocido)
   - Rojo → Error o acceso denegado

3. **Feedback Visual:**

   - Animaciones suaves (slideIn, pulse)
   - Barra de confianza con gradiente
   - Estados claros (scanning, success, error)

4. **Responsive Design:** Adaptable a móviles y tablets

5. **Accesibilidad:**
   - Contraste adecuado
   - Iconos descriptivos
   - Mensajes claros

## 🔐 Seguridad

### Medidas Implementadas

1. **Threshold alto (0.75):** Minimiza falsos positivos
2. **Análisis de confidence gap:** Detecta incertidumbre del modelo
3. **Sesiones con timeout:** 1 hora de duración
4. **No almacenamiento de imágenes:** Solo se procesan embeddings
5. **CSRF protection:** Django CSRF habilitado (excepto API endpoints)

### Casos de Uso

✅ **Acceso Concedido:**

- Confianza ≥ 75%
- Gap entre top-1 y top-2 ≥ 15%
- Persona en dataset

❌ **Acceso Denegado:**

- Confianza < 75%
- Gap < 15% (modelo confundido)
- Persona no en dataset

## 🧪 Testing

### Prueba con Personas del Dataset

1. Posiciona a IsmaelSailema, AlisonSalas o WilliamTacuri frente a la cámara
2. Debe reconocer y dar acceso automáticamente

### Prueba con Persona Desconocida

1. Posiciona a alguien que NO está en el dataset (ej: Rafa)
2. Debe mostrar "Desconocido" y denegar acceso

### Prueba sin Rostro

1. Tapa la cámara o apunta a un objeto
2. Debe mostrar "No se detectó rostro"

## 📊 Métricas del Modelo

- **Arquitectura:** FaceNet (InceptionResNetV2) + CNN (256→128→64→output)
- **Embeddings:** 512 dimensiones
- **Dataset:** Imágenes RGB 160x160
- **Accuracy esperada:** >95% en personas del dataset
- **Falsos positivos:** <5% con threshold 0.75

## 🛠️ Troubleshooting

### La cámara no se inicia

- Verificar permisos del navegador
- Probar en Chrome/Edge (mejor compatibilidad)
- Usar HTTPS en producción

### Reconocimiento muy lento

- Reducir frecuencia en `setInterval` (actualmente 2000ms)
- Reducir resolución de cámara
- Usar GPU si está disponible

### Muchos "Desconocido"

- Bajar threshold a 0.70 en `views.py` y `login.html`
- Reentrenar modelo con más imágenes
- Verificar iluminación

### Error al cargar modelo

- Verificar rutas en `views.py`:
  - `MODEL_PATH`
  - `LABEL_ENCODER_PATH`
- Asegurar que existen los archivos `.h5` y `.pkl`

## 📝 Notas Técnicas

### Por qué FaceNet + CNN

1. **FaceNet (Pretrained):**

   - Entrenado en millones de rostros
   - Genera embeddings robustos de 512D
   - No requiere reentrenamiento

2. **CNN Personalizada:**
   - Aprende patrones específicos de tu dataset
   - Rápida de entrenar (pocos parámetros)
   - Fácil de actualizar con nuevas personas

### Flujo de Reconocimiento

```
Imagen Cámara → Detección Haar Cascade → Crop + Resize 160x160
→ FaceNet Embedding (512D) → CNN Classifier → Predicción + Confianza
→ Análisis Threshold → Autorización (Sí/No)
```

## 🎓 Créditos

- **Desarrollo:** Sistema de login biométrico con Django
- **Modelo:** FaceNet + CNN Classifier entrenada en dataset personalizado
- **UI/UX:** Diseño moderno siguiendo principios de Material Design
- **Framework:** Django 5.2.7, TensorFlow, OpenCV

---

## 📞 Soporte

Para problemas o preguntas, revisar:

1. Este README
2. Logs de Django en consola
3. Consola del navegador (F12)

**¡Disfruta de tu sistema de login facial! 🎉**
