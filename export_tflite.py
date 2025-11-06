"""
Export du modèle au format TensorFlow Lite pour déploiement mobile
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config


def convert_to_tflite(model_path, output_path, quantize=True, representative_dataset=None):
    """
    Convertit un modèle Keras en TensorFlow Lite
    
    Args:
        model_path: Chemin vers le modèle SavedModel ou .h5
        output_path: Chemin de sortie pour le fichier .tflite
        quantize: Activer la quantization (int8)
        representative_dataset: Dataset représentatif pour la quantization
    
    Returns:
        tflite_model_size: Taille du modèle en Ko
    """
    print("=" * 60)
    print("📦 CONVERSION EN TENSORFLOW LITE")
    print("=" * 60)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle depuis: {model_path}")
    
    # Créer le converter
    if model_path.endswith('.h5'):
        model = keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        # SavedModel format
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Configuration du converter
    if quantize and representative_dataset is not None:
        print("\n⚙️  Configuration de la quantization INT8 optimisée...")
        
        # Activer la quantization post-training
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Fournir un dataset représentatif pour la quantization
        converter.representative_dataset = representative_dataset
        
        # AMÉLIORATION 1: Garder les entrées/sorties en float32 pour plus de précision
        # (seulement les poids internes sont quantizés en INT8)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback pour opérations non supportées
        ]
        # NE PAS quantizer les entrées/sorties pour garder la précision
        # converter.inference_input_type = tf.uint8  # DÉSACTIVÉ
        # converter.inference_output_type = tf.uint8  # DÉSACTIVÉ
        
    elif quantize:
        # Quantization simple sans dataset représentatif (float16)
        print("\n⚙️  Configuration de la quantization FLOAT16...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        print("\n⚙️  Pas de quantization (modèle float32)")
    
    # Convertir
    print("\n🔄 Conversion en cours...")
    tflite_model = converter.convert()
    
    # Sauvegarder
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Afficher la taille
    tflite_model_size = len(tflite_model) / 1024  # en Ko
    print(f"\n✅ Modèle TFLite sauvegardé: {output_path}")
    print(f"📊 Taille du modèle: {tflite_model_size:.2f} Ko")
    
    print("=" * 60)
    
    return tflite_model_size


def create_representative_dataset_generator(X_val, num_samples=100):
    """
    Crée un générateur de dataset représentatif pour la quantization
    AMÉLIORÉ: Utilise plus d'échantillons et couvre mieux la distribution
    
    Args:
        X_val: Dataset de validation
        num_samples: Nombre d'échantillons à utiliser (augmenté pour meilleure calibration)
    
    Returns:
        representative_dataset_gen: Générateur pour le converter
    """
    def representative_dataset_gen():
        # AMÉLIORATION 2: Utiliser TOUS les échantillons de validation pour meilleure calibration
        # Au lieu de prendre séquentiellement, on mélange pour couvrir toute la distribution
        indices = np.random.permutation(len(X_val))[:num_samples]
        for idx in indices:
            # Prendre un échantillon
            sample = X_val[idx:idx+1].astype(np.float32)
            yield [sample]
    
    return representative_dataset_gen


def test_tflite_model(tflite_path, X_test, y_test, num_samples=10):
    """
    Teste le modèle TFLite et compare avec les prédictions originales
    
    Args:
        tflite_path: Chemin vers le modèle .tflite
        X_test: Images de test
        y_test: Heatmaps de test
        num_samples: Nombre d'échantillons à tester
    
    Returns:
        avg_error: Erreur moyenne
    """
    print("\n🧪 Test du modèle TFLite...")
    
    # Charger l'interpréteur TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Obtenir les détails des entrées/sorties
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Détails de l'interpréteur:")
    print(f"   - Input shape: {input_details[0]['shape']}")
    print(f"   - Input type: {input_details[0]['dtype']}")
    print(f"   - Output shape: {output_details[0]['shape']}")
    print(f"   - Output type: {output_details[0]['dtype']}")
    
    # Tester sur quelques échantillons
    errors = []
    for i in range(min(num_samples, len(X_test))):
        # Préparer l'entrée
        input_data = X_test[i:i+1].astype(np.float32)
        
        # Si le modèle attend des uint8, il faut quantizer l'entrée
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        
        # Inférence
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Récupérer la sortie
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Si la sortie est quantizée, il faut la déquantizer
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Calculer l'erreur
        error = np.mean(np.abs(output_data - y_test[i:i+1]))
        errors.append(error)
    
    avg_error = np.mean(errors)
    print(f"\n📊 Résultats du test:")
    print(f"   - Nombre d'échantillons testés: {len(errors)}")
    print(f"   - Erreur moyenne (MAE): {avg_error:.6f}")
    
    return avg_error


def export_model(model=None, model_path=None, X_val=None, model_name="pose_model"):
    """
    Pipeline complet d'export du modèle en TFLite
    
    Args:
        model: Modèle Keras (optionnel si model_path est fourni)
        model_path: Chemin vers le modèle sauvegardé (optionnel si model est fourni)
        X_val: Dataset de validation pour la quantization
        model_name: Nom du modèle
    
    Returns:
        tflite_path: Chemin vers le fichier .tflite
    """
    print("=" * 60)
    print("🚀 EXPORT DU MODÈLE EN TENSORFLOW LITE")
    print("=" * 60)
    
    # Si un modèle Keras est fourni, le sauvegarder d'abord
    if model is not None:
        saved_model_dir = os.path.join(config.MODELS_DIR, f"{model_name}_for_export")
        print(f"\n💾 Sauvegarde du modèle au format SavedModel...")
        model.save(saved_model_dir, save_format='tf')
        model_path = saved_model_dir
    
    if model_path is None:
        raise ValueError("Vous devez fournir soit 'model' soit 'model_path'")
    
    # Chemin de sortie pour le .tflite
    tflite_path = os.path.join(config.MODELS_DIR, config.TFLITE_MODEL_NAME)
    
    # Créer le dataset représentatif si X_val est fourni et quantization activée
    representative_dataset = None
    if config.TFLITE_QUANTIZATION and X_val is not None:
        # AMÉLIORATION 3: Utiliser plus d'échantillons pour la calibration (500 au lieu de 100)
        num_calibration_samples = min(500, len(X_val))
        print(f"\n📊 Création du dataset représentatif ({num_calibration_samples} échantillons)...")
        representative_dataset = create_representative_dataset_generator(
            X_val, 
            num_samples=num_calibration_samples
        )
    
    # Convertir en TFLite
    tflite_size = convert_to_tflite(
        model_path=model_path,
        output_path=tflite_path,
        quantize=config.TFLITE_QUANTIZATION,
        representative_dataset=representative_dataset
    )
    
    print(f"\n✅ Export terminé!")
    print(f"📱 Modèle prêt pour le déploiement mobile: {tflite_path}")
    
    # Instructions pour l'utilisation
    print("\n" + "=" * 60)
    print("📱 UTILISATION DU MODÈLE TFLITE")
    print("=" * 60)
    print("\n🤖 Android (Java/Kotlin):")
    print("   1. Ajoutez le .tflite dans assets/")
    print("   2. Ajoutez la dépendance: implementation 'org.tensorflow:tensorflow-lite:2.x.x'")
    print("   3. Chargez avec: Interpreter.create(...)")
    print("   4. Utilisez GPU Delegate ou NNAPI pour accélérer")
    
    print("\n🍎 iOS (Swift/Objective-C):")
    print("   1. Ajoutez le .tflite au projet Xcode")
    print("   2. Ajoutez TensorFlowLiteSwift via CocoaPods/SPM")
    print("   3. Chargez avec: Interpreter(modelPath: ...)")
    print("   4. Utilisez Metal Delegate pour accélérer")
    
    print("\n🔄 Conversion CoreML (optionnel pour iOS):")
    print("   - Utilisez coremltools pour convertir .tflite en .mlmodel")
    print("=" * 60)
    
    return tflite_path


if __name__ == "__main__":
    print("✅ Module export_tflite.py chargé avec succès")
    print("📝 Utilisez main.py pour exporter le modèle après l'entraînement")
