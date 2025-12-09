"""
Módulo de reconocimiento facial usando FaceNet + CNN
"""
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from keras_facenet import FaceNet


class FaceRecognitionSystem:
    """Sistema de reconocimiento facial en tiempo real"""
    
    def __init__(self, model_path, label_encoder_path):
        """
        Inicializa el sistema de reconocimiento facial
        
        Args:
            model_path: Ruta al modelo CNN entrenado (.h5)
            label_encoder_path: Ruta al label encoder (.pkl)
        """
        self.model_path = Path(model_path)
        self.label_encoder_path = Path(label_encoder_path)
        
        # Cargar FaceNet
        print("Cargando FaceNet...")
        self.facenet_model = FaceNet()
        
        # Cargar modelo clasificador
        print(f"Cargando modelo desde {self.model_path}...")
        self.classifier_model = load_model(str(self.model_path))
        
        # Cargar label encoder
        print(f"Cargando label encoder desde {self.label_encoder_path}...")
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Cargar detector de rostros Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("✓ Sistema de reconocimiento facial inicializado correctamente")
        print(f"✓ Clases reconocidas: {list(self.label_encoder.classes_)}")
    
    def detect_face(self, image):
        """
        Detecta el rostro más grande en una imagen
        
        Args:
            image: Imagen BGR de OpenCV
            
        Returns:
            face_crop: Rostro recortado y redimensionado a 160x160 (RGB)
            face_box: Coordenadas (x, y, w, h) del rostro detectado
            None si no se detecta rostro
        """
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Tomar el rostro más grande
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        
        # Convertir a RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Recortar rostro con padding
        padding = 20
        top = max(0, y - padding)
        left = max(0, x - padding)
        bottom = min(img_rgb.shape[0], y + h + padding)
        right = min(img_rgb.shape[1], x + w + padding)
        
        face_crop = img_rgb[top:bottom, left:right]
        
        # Redimensionar a 160x160 (tamaño requerido por FaceNet)
        face_resized = cv2.resize(face_crop, (160, 160))
        
        return face_resized, (x, y, w, h)
    
    def predict(self, image, threshold=0.80, min_confidence_gap=0.20):
        """
        Predice la identidad de una persona en la imagen
        
        Args:
            image: Imagen BGR de OpenCV
            threshold: Umbral de confianza mínimo (0-1). Por defecto 0.80 (80%)
            min_confidence_gap: Gap mínimo entre top-1 y top-2 (0-1). Por defecto 0.20 (20%)
            
        Returns:
            dict con:
                - name: Nombre predicho o "Desconocido"
                - confidence: Confianza de la predicción (0-1)
                - confidence_gap: Diferencia entre top-1 y top-2
                - face_detected: True si se detectó un rostro
                - face_box: Coordenadas del rostro (x, y, w, h)
                - authorized: True si es una persona conocida con confianza suficiente
                - top_predictions: Lista con top 3 predicciones
                - rejection_reasons: Lista de razones si fue rechazado
        """
        # Detectar rostro
        face_resized, face_box = self.detect_face(image)
        
        if face_resized is None:
            return {
                'name': 'No se detectó rostro',
                'confidence': 0.0,
                'confidence_gap': 0.0,
                'face_detected': False,
                'face_box': None,
                'authorized': False,
                'top_predictions': [],
                'rejection_reasons': ['no_face_detected']
            }
        
        # Extraer embedding con FaceNet
        face_batch = np.expand_dims(face_resized, axis=0)
        embedding = self.facenet_model.embeddings(face_batch)
        
        # Predecir con clasificador CNN
        prediction = self.classifier_model.predict(embedding, verbose=0)[0]
        
        # Top 3 predicciones
        top_3_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = [
            {
                'class': self.label_encoder.classes_[idx],
                'confidence': float(prediction[idx])
            }
            for idx in top_3_indices
        ]
        
        # Mejor predicción
        predicted_class = top_3_indices[0]
        confidence = float(prediction[predicted_class])
        
        # Calcular gap de confianza
        if len(top_3_indices) > 1:
            confidence_gap = float(prediction[top_3_indices[0]] - prediction[top_3_indices[1]])
        else:
            confidence_gap = confidence
        
        # Verificar criterios de autorización
        rejection_reasons = []
        
        if confidence < threshold:
            rejection_reasons.append(f'confidence_too_low_{confidence:.2%}<{threshold:.0%}')
        
        if confidence_gap < min_confidence_gap:
            rejection_reasons.append(f'gap_too_small_{confidence_gap:.2%}<{min_confidence_gap:.0%}')
        
        # Decidir si autorizar
        authorized = len(rejection_reasons) == 0
        
        if not authorized:
            name = "Desconocido"
        else:
            name = self.label_encoder.classes_[predicted_class]
            rejection_reasons = ['authorized']
        
        return {
            'name': name,
            'confidence': confidence,
            'confidence_gap': confidence_gap,
            'face_detected': True,
            'face_box': face_box,
            'authorized': authorized,
            'top_predictions': top_predictions,
            'rejection_reasons': rejection_reasons
        }
    
    def draw_result(self, image, result):
        """
        Dibuja el resultado de la predicción en la imagen
        
        Args:
            image: Imagen BGR de OpenCV
            result: Diccionario con resultado de predict()
            
        Returns:
            image: Imagen con resultado dibujado
        """
        img_copy = image.copy()
        
        if not result['face_detected']:
            # Mensaje de no detección
            cv2.putText(
                img_copy,
                result['name'],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return img_copy
        
        # Dibujar rectángulo alrededor del rostro
        x, y, w, h = result['face_box']
        color = (0, 255, 0) if result['authorized'] else (0, 165, 255)  # Verde o Naranja
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Preparar texto
        name_text = result['name']
        conf_text = f"Confianza: {result['confidence']:.1%}"
        
        # Fondo para el texto
        text_y = y - 10 if y > 40 else y + h + 25
        cv2.rectangle(img_copy, (x, text_y - 25), (x + w, text_y + 5), color, -1)
        
        # Texto con nombre y confianza
        cv2.putText(
            img_copy,
            name_text,
            (x + 5, text_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img_copy,
            conf_text,
            (x + 5, text_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        
        return img_copy
