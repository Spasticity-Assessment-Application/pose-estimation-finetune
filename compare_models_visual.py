"""
Visualisation comparative des prédictions de différents modèles
"""
import cv2
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import keras
import argparse
import json
from matplotlib import pyplot as plt

def load_model_with_fallback(model_path):
    """Charge un modèle avec fallback TFLite"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model, "keras"
    except Exception as e:
        print(f"⚠️ Keras failed: {e}")
        try:
            model_dir = os.path.dirname(model_path)
            tflite_names = ["pose_model_dlc_float32.tflite", "pose_model_dlc_dynamic.tflite"]
            for name in tflite_names:
                tflite_path = os.path.join(model_dir, name)
                if os.path.exists(tflite_path):
                    interpreter = tf.lite.Interpreter(model_path=tflite_path)
                    interpreter.allocate_tensors()
                    return interpreter, "tflite"
        except Exception as e2:
            print(f"❌ TFLite failed: {e2}")
            raise

def load_model_config(model_dir):
    """Charge la config du modèle"""
    config_path = os.path.join(model_dir, "models", "model_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_image(img_path, input_size):
    """Prétraite l'image"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), img.shape[:2]

def predict_keypoints(model, model_type, img, input_size, heatmap_size):
    """Prédit les keypoints"""
    if model_type == "keras":
        heatmaps = model.predict(img, verbose=0)[0]
    else:  # tflite
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        heatmaps = model.get_tensor(output_details[0]['index'])[0]

    keypoints = []
    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        x_norm = max_pos[1] / heatmap_size[1]
        y_norm = max_pos[0] / heatmap_size[0]
        confidence = float(heatmap[max_pos])
        keypoints.append((x_norm, y_norm, confidence))

    return keypoints

def draw_keypoints(img, keypoints, color, labels=None):
    """Dessine les keypoints sur l'image"""
    if labels is None:
        labels = ["Hanche", "Genoux", "Cheville"]

    img_copy = img.copy()
    for i, (x_norm, y_norm, conf) in enumerate(keypoints):
        x = int(x_norm * img.shape[1])
        y = int(y_norm * img.shape[0])

        # Cercle
        cv2.circle(img_copy, (x, y), 8, color, -1)
        cv2.circle(img_copy, (x, y), 12, (255, 255, 255), 2)

        # Label
        label = f"{labels[i]}: {conf:.2f}"
        cv2.putText(img_copy, label, (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img_copy

def create_comparison_visualization(image_path, model_dirs, model_names, output_path):
    """Crée une visualisation comparative"""
    # Charger l'image originale
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Préparer la figure
    n_models = len(model_dirs)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6*(n_models+1), 6))

    # Image originale
    axes[0].imshow(original_img)
    axes[0].set_title("Image Originale", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]  # Rouge, Vert, Bleu, Orange

    for i, (model_dir, model_name) in enumerate(zip(model_dirs, model_names), 1):
        try:
            # Charger le modèle
            model_path = os.path.join(model_dir, "models", model_name)
            model, model_type = load_model_with_fallback(model_path)

            # Charger la config
            config = load_model_config(model_dir)
            input_size = tuple(config['input_size'])
            heatmap_size = tuple(config['heatmap_size'])

            # Prétraiter l'image
            processed_img, orig_shape = preprocess_image(image_path, input_size)

            # Prédire
            keypoints = predict_keypoints(model, model_type, processed_img, input_size, heatmap_size)

            # Dessiner
            img_with_kp = draw_keypoints(original_img, keypoints, colors[i-1])

            # Afficher
            axes[i].imshow(img_with_kp)
            backbone = config['backbone']
            pck_placeholder = "PCK à calculer"
            axes[i].set_title(f"{backbone}\n{model_name}", fontsize=12, fontweight='bold')
            axes[i].axis('off')

        except Exception as e:
            axes[i].imshow(np.zeros_like(original_img))
            axes[i].set_title(f"Erreur: {model_name}\n{str(e)[:50]}...", fontsize=10)
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualisation sauvegardée: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualisation comparative des modèles")
    parser.add_argument('--image', type=str, required=True,
                       help="Chemin vers l'image de test")
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help="Dossiers des modèles à comparer")
    parser.add_argument('--model-names', type=str, nargs='+',
                       default=['pose_model_dlc_finetune_best.h5'] * 10,
                       help="Noms des fichiers modèles")
    parser.add_argument('--output', type=str, default='model_comparison.png',
                       help="Fichier de sortie")

    args = parser.parse_args()

    if len(args.models) != len(args.model_names):
        print("❌ Nombre de modèles et de noms différents")
        return

    create_comparison_visualization(args.image, args.models, args.model_names, args.output)

if __name__ == "__main__":
    main()