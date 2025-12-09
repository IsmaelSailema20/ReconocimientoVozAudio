from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import cv2
import base64
import numpy as np
import json
from pathlib import Path
from .face_recognition_model import FaceRecognitionSystem

# Ruta al modelo entrenado
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'EntrenamientoModelo' / 'models' / 'best_classifier_model.h5'
LABEL_ENCODER_PATH = BASE_DIR / 'EntrenamientoModelo' / 'models' / 'label_encoder.pkl'

# Inicializar sistema de reconocimiento facial (solo una vez)
face_system = None

def get_face_system():
    """Inicializa el sistema de reconocimiento facial si no existe"""
    global face_system
    if face_system is None:
        try:
            face_system = FaceRecognitionSystem(
                model_path=MODEL_PATH,
                label_encoder_path=LABEL_ENCODER_PATH
            )
        except Exception as e:
            print(f"Error al inicializar sistema de reconocimiento: {e}")
            raise
    return face_system


def index(request):
    """Vista principal - página de login con reconocimiento facial"""
    # Si ya está autenticado, redirigir al dashboard
    if request.session.get('authenticated', False):
        return redirect('dashboard')
    
    return render(request, 'face_recognition_app/login.html')


def dashboard(request):
    """Dashboard para usuarios autenticados"""
    if not request.session.get('authenticated', False):
        return redirect('index')
    
    user_name = request.session.get('user_name', 'Usuario')
    context = {
        'user_name': user_name,
        'login_time': request.session.get('login_time', '')
    }
    return render(request, 'face_recognition_app/dashboard.html', context)


def logout_view(request):
    """Cerrar sesión"""
    request.session.flush()
    return redirect('index')


@csrf_exempt
@require_http_methods(["POST"])
def recognize_face(request):
    """
    API endpoint para reconocer rostro desde imagen base64
    
    Recibe: JSON con imagen en base64
    Retorna: JSON con resultado del reconocimiento
    """
    try:
        # Obtener sistema de reconocimiento
        system = get_face_system()
        
        # Parsear JSON
        data = json.loads(request.body)
        image_data = data.get('image', '')
        
        if not image_data:
            return JsonResponse({
                'success': False,
                'error': 'No se recibió imagen'
            }, status=400)
        
        # Decodificar imagen base64
        # Formato: "data:image/jpeg;base64,/9j/4AAQ..."
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JsonResponse({
                'success': False,
                'error': 'No se pudo decodificar la imagen'
            }, status=400)
        
        # Realizar predicción con parámetros ajustados
        result = system.predict(image, threshold=0.80, min_confidence_gap=0.20)
        
        # Si es una persona autorizada, crear sesión
        if result['authorized'] and result['name'] != 'Desconocido':
            request.session['authenticated'] = True
            request.session['user_name'] = result['name']
            from datetime import datetime
            request.session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return JsonResponse({
                'success': True,
                'authorized': True,
                'name': result['name'],
                'confidence': result['confidence'],
                'confidence_gap': result['confidence_gap'],
                'top_predictions': result['top_predictions'],
                'rejection_reasons': result['rejection_reasons'],
                'message': f'¡Bienvenido/a {result["name"]}!'
            })
        else:
            return JsonResponse({
                'success': True,
                'authorized': False,
                'name': result['name'],
                'confidence': result['confidence'],
                'confidence_gap': result.get('confidence_gap', 0),
                'top_predictions': result.get('top_predictions', []),
                'rejection_reasons': result.get('rejection_reasons', []),
                'message': 'Acceso denegado. Persona no reconocida.'
            })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def verify_face_stream(request):
    """
    API endpoint para verificación continua de rostro
    Retorna resultado de reconocimiento sin crear sesión
    """
    try:
        system = get_face_system()
        
        data = json.loads(request.body)
        image_data = data.get('image', '')
        
        if not image_data:
            return JsonResponse({
                'success': False,
                'error': 'No se recibió imagen'
            }, status=400)
        
        # Decodificar imagen
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JsonResponse({
                'success': False,
                'error': 'No se pudo decodificar la imagen'
            }, status=400)
        
        # Realizar predicción
        result = system.predict(image, threshold=0.75)
        
        return JsonResponse({
            'success': True,
            'name': result['name'],
            'confidence': result['confidence'],
            'authorized': result['authorized'],
            'face_detected': result['face_detected']
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def check_session(request):
    """Verifica si el usuario está autenticado"""
    return JsonResponse({
        'authenticated': request.session.get('authenticated', False),
        'user_name': request.session.get('user_name', '')
    })
