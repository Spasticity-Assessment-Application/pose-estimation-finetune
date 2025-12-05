#!/usr/bin/env python3
"""
Annotate extracted frames with pose estimation models
"""
import cv2
import numpy as np
import json
from tensorflow import keras
import os
from pathlib import Path
import config

def load_model_config(model_path):
    """Charge la configuration du modèle"""
    model_dir = Path(model_path).parent
    config_file = model_dir / "model_config.json"

    if config_file.exists():
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        return model_config
    return None

def load_keras_model(model_path):
    """Charge le modèle Keras"""
    print(f"🔄 Chargement du modèle: {model_path}")

    # Essayer SavedModel d'abord
    saved_model_path = model_path.replace('_finetune_best.h5', '_saved_model')
    if os.path.exists(saved_model_path):
        try:
            import tensorflow as tf
            model = tf.saved_model.load(saved_model_path)
            model.is_saved_model = True
            print("✅ SavedModel chargé")
            return model
        except Exception as e:
            print(f"⚠️  Échec SavedModel: {e}")

    # Fallback vers .h5
    try:
        model = keras.models.load_model(model_path)
        print("✅ Modèle .h5 chargé")
        return model
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None

def preprocess_image(image, input_size=(256, 256)):
    """Prétraite l'image pour le modèle"""
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch

def postprocess_heatmap(heatmap, threshold=0.1):
    """Extrait les keypoints de la heatmap"""
    keypoints = []
    h, w = heatmap.shape

    for i in range(3):  # 3 keypoints: hanche, genou, cheville
        kp_map = heatmap[:, :, i]
        max_val = np.max(kp_map)

        if max_val > threshold:
            y, x = np.unravel_index(np.argmax(kp_map), kp_map.shape)
            keypoints.append((int(x * 4), int(y * 4)))  # Scale back to original size
        else:
            keypoints.append(None)

    return keypoints

def annotate_image(image, keypoints, model_name):
    """Annotate l'image avec les keypoints"""
    annotated = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Bleu, Vert, Rouge
    labels = ['Hanche', 'Genou', 'Cheville']

    for i, (kp, color, label) in enumerate(zip(keypoints, colors, labels)):
        if kp is not None:
            x, y = kp
            cv2.circle(annotated, (x, y), 8, color, -1)
            cv2.putText(annotated, label, (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Ajouter le nom du modèle
    cv2.putText(annotated, model_name, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated

def predict_pose(model, image, config):
    """Prédit la pose sur une image"""
    input_size = config.get('input_size', [256, 256])
    heatmap_size = config.get('heatmap_size', [64, 64])

    # Préprocessing
    processed = preprocess_image(image, tuple(input_size))

    # Prédiction
    if hasattr(model, 'is_saved_model') and model.is_saved_model:
        # Pour SavedModel
        import tensorflow as tf
        infer = model.signatures["serving_default"]
        result = infer(image_input=tf.convert_to_tensor(processed))
        heatmap = result['output_0'].numpy()[0]
    else:
        # Pour modèle Keras
        heatmap = model.predict(processed, verbose=0)[0]

    # Post-processing
    heatmap = heatmap[0]  # Remove batch dimension
    keypoints = postprocess_heatmap(heatmap)

    return keypoints

def main():
    # Configuration
    frames_dir = Path("frames_extraction")
    models = {
        'DLC_Mobil': 'output/DLC_Mobil_20251120_125726/models/pose_model_dlc_finetune_best.h5',
        'MNv3S': 'output/MNv3S_20251124_134351/models/pose_model_dlc_finetune_best.h5',
        'MNv3L': 'output/MNv3L_20251125_185928/models/pose_model_dlc_finetune_best.h5'
    }

    # Frames à annoter
    frame_files = ['frame_0025.jpg', 'frame_0050.jpg', 'frame_0075.jpg', 'frame_0100.jpg']

    print("🎯 ANNOTATION DE FRAMES")
    print("=" * 50)

    # Créer répertoire de sortie
    output_dir = Path("frames_annotated")
    output_dir.mkdir(exist_ok=True)

    # Charger les modèles
    loaded_models = {}
    model_configs = {}

    for model_name, model_path in models.items():
        print(f"🔄 Chargement {model_name}...")
        model = load_keras_model(model_path)
        if model is not None:
            loaded_models[model_name] = model
            config_data = load_model_config(model_path)
            model_configs[model_name] = config_data
            print(f"✅ {model_name} chargé")
        else:
            print(f"❌ Échec chargement {model_name}")

    if not loaded_models:
        print("❌ Aucun modèle chargé")
        return

    # Annoter chaque frame avec chaque modèle
    for frame_file in frame_files:
        frame_path = frames_dir / frame_file
        if not frame_path.exists():
            print(f"⚠️ Frame non trouvé: {frame_file}")
            continue

        print(f"\n📸 Traitement: {frame_file}")

        # Charger l'image
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"❌ Impossible de charger: {frame_file}")
            continue

        # Annoter avec chaque modèle
        for model_name, model in loaded_models.items():
            print(f"   🤖 {model_name}...")

            try:
                keypoints = predict_pose(model, image, model_configs[model_name])
                annotated = annotate_image(image, keypoints, model_name)

                # Sauvegarder
                output_filename = f"{frame_file.replace('.jpg', '')}_{model_name}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), annotated)

                print(f"   ✅ Sauvegardé: {output_filename}")

            except Exception as e:
                print(f"   ❌ Erreur avec {model_name}: {e}")

    print("\n✅ ANNOTATION TERMINÉE")
    print(f"📂 Résultats: {output_dir}")

    # Lister les fichiers créés
    annotated_files = list(output_dir.glob("*.jpg"))
    print(f"\n📋 {len(annotated_files)} images annotées créées:")
    for f in sorted(annotated_files):
        print(f"   - {f.name}")

if __name__ == "__main__":
    main()