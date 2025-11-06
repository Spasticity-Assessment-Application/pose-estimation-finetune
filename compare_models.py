"""
Compare la précision entre le modèle Keras et TFLite
Mesure l'impact de la quantization
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import config
import os
from pathlib import Path


def load_keras_model(model_path):
    """Charge le modèle Keras"""
    return keras.models.load_model(model_path)


def load_tflite_model(model_path):
    """Charge le modèle TFLite"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_keras(model, image):
    """Prédiction avec Keras"""
    image_batch = np.expand_dims(image, axis=0)
    heatmaps = model.predict(image_batch, verbose=0)[0]
    return heatmaps


def predict_tflite(interpreter, image):
    """Prédiction avec TFLite"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    
    # Quantization de l'entrée si nécessaire
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        image_batch = (image_batch / input_scale + input_zero_point).astype(np.uint8)
    
    # Inférence
    interpreter.set_tensor(input_details[0]['index'], image_batch)
    interpreter.invoke()
    
    # Sortie
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Déquantization de la sortie si nécessaire
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - output_zero_point) * output_scale
    
    return output


def extract_keypoints(heatmaps):
    """Extrait les coordonnées des keypoints"""
    keypoints = []
    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        y = max_pos[0] / heatmap.shape[0]  # Normalisé
        x = max_pos[1] / heatmap.shape[1]
        confidence = heatmap[max_pos]
        keypoints.append({'x': x, 'y': y, 'confidence': confidence})
    return keypoints


def compute_metrics(keras_kpts, tflite_kpts):
    """Calcule les métriques de comparaison"""
    distances = []
    conf_diffs = []
    
    for k_kpt, t_kpt in zip(keras_kpts, tflite_kpts):
        # Distance euclidienne (en coordonnées normalisées)
        dist = np.sqrt((k_kpt['x'] - t_kpt['x'])**2 + (k_kpt['y'] - t_kpt['y'])**2)
        distances.append(dist)
        
        # Différence de confiance
        conf_diff = abs(k_kpt['confidence'] - t_kpt['confidence'])
        conf_diffs.append(conf_diff)
    
    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'mean_conf_diff': np.mean(conf_diffs),
        'max_conf_diff': np.max(conf_diffs)
    }


def compare_models(keras_path, tflite_path, X_test, num_samples=50):
    """
    Compare les modèles Keras et TFLite
    
    Args:
        keras_path: Chemin vers le modèle Keras (.h5)
        tflite_path: Chemin vers le modèle TFLite
        X_test: Images de test
        num_samples: Nombre d'échantillons à tester
    """
    print("=" * 60)
    print("🔍 COMPARAISON KERAS vs TFLITE")
    print("=" * 60)
    
    # Charger les modèles
    print("\n📂 Chargement des modèles...")
    keras_model = load_keras_model(keras_path)
    tflite_interpreter = load_tflite_model(tflite_path)
    print("✅ Modèles chargés")
    
    # Tester sur plusieurs échantillons
    print(f"\n🧪 Test sur {num_samples} échantillons...")
    all_metrics = []
    
    for i in range(min(num_samples, len(X_test))):
        image = X_test[i]
        
        # Prédictions
        keras_heatmaps = predict_keras(keras_model, image)
        tflite_heatmaps = predict_tflite(tflite_interpreter, image)
        
        # Extraire keypoints
        keras_kpts = extract_keypoints(keras_heatmaps)
        tflite_kpts = extract_keypoints(tflite_heatmaps)
        
        # Métriques
        metrics = compute_metrics(keras_kpts, tflite_kpts)
        all_metrics.append(metrics)
    
    # Statistiques globales
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE LA COMPARAISON")
    print("=" * 60)
    
    avg_distance = np.mean([m['mean_distance'] for m in all_metrics])
    max_distance = np.max([m['max_distance'] for m in all_metrics])
    avg_conf_diff = np.mean([m['mean_conf_diff'] for m in all_metrics])
    
    print(f"\n🎯 Précision de localisation:")
    print(f"   - Distance moyenne: {avg_distance:.4f} (normalisé)")
    print(f"   - Distance max: {max_distance:.4f}")
    print(f"   - En pixels (192x192): {avg_distance * 192:.1f} px")
    
    print(f"\n📈 Différence de confiance:")
    print(f"   - Moyenne: {avg_conf_diff:.4f}")
    
    # Interprétation
    print("\n💡 Interprétation:")
    if avg_distance < 0.02:
        print("   ✅ EXCELLENT - Différence négligeable (<4px)")
    elif avg_distance < 0.05:
        print("   ✔️  BON - Différence acceptable (<10px)")
    elif avg_distance < 0.10:
        print("   ⚠️  MOYEN - Différence notable (<20px)")
    else:
        print("   ❌ IMPORTANT - Grosse différence (>20px)")
    
    print("\n🔧 Recommandations:")
    if avg_distance > 0.05:
        print("   1. Utiliser la conversion optimisée (entrées/sorties float32)")
        print("   2. Augmenter le dataset représentatif à 500+ échantillons")
        print("   3. Envisager le Quantization-Aware Training (QAT)")
        print("   4. Utiliser le modèle optimisé avec ReLU6 et Sigmoid")
    else:
        print("   ✅ La quantization est bien calibrée!")
    
    print("=" * 60)
    
    return {
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'avg_conf_diff': avg_conf_diff
    }


if __name__ == "__main__":
    # Exemple d'utilisation
    print("✅ Module compare_models.py chargé")
    print("\n💡 Utilisation:")
    print("from compare_models import compare_models")
    print("results = compare_models(")
    print("    keras_path='models/pose_model_best.h5',")
    print("    tflite_path='models/pose_model_quantized.tflite',")
    print("    X_test=X_val,")
    print("    num_samples=50")
    print(")")
