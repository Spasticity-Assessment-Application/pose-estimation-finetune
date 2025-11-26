"""
Test du modèle Keras (.h5) sur une vidéo
"""
import cv2
import numpy as np
import json
from tensorflow import keras
import argparse
import os
from pathlib import Path
import config


def load_model_config(model_path):
    """Charge la configuration du modèle depuis model_config.json"""
    model_dir = Path(model_path).parent
    config_file = model_dir / "model_config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        print(f"✅ Configuration chargée: {config_file}")
        print(f"   - Backbone: {model_config.get('backbone', 'unknown')}")
        print(f"   - Input size: {model_config.get('input_size', [192, 192])}")
        print(f"   - Heatmap size: {model_config.get('heatmap_size', [64, 64])}")
        return model_config
    else:
        print(f"⚠️  Configuration non trouvée: {config_file}")
        print(f"💡 Utilisation des paramètres par défaut")
        return None


def load_keras_model(model_path):
    """Charge le modèle Keras ou SavedModel avec compatibilité TensorFlow 2.15+"""
    print(f"🔄 Chargement du modèle...")
    
    # Essayer d'abord le SavedModel si disponible
    saved_model_path = model_path.replace('_finetune_best.h5', '_saved_model')
    if os.path.exists(saved_model_path):
        print(f"🔄 Chargement SavedModel: {saved_model_path}")
        try:
            import tensorflow as tf
            model = tf.saved_model.load(saved_model_path)
            model.is_saved_model = True  # Marquer comme SavedModel
            print("✅ SavedModel chargé")
            return model
        except Exception as e:
            print(f"⚠️  Échec SavedModel: {e}")
    
    # Fallback vers .h5
    print(f"🔄 Chargement modèle .h5: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.is_saved_model = False
        print("✅ Modèle .h5 chargé")
        return model
    except Exception as e:
        print(f"❌ Échec du chargement du modèle: {e}")
        raise e


def get_model_input_size(model):
    """Extrait la taille d'entrée attendue par le modèle"""
    try:
        if hasattr(model, 'is_saved_model') and model.is_saved_model:
            # Pour SavedModel
            import tensorflow as tf
            infer = model.signatures['serving_default']
            input_spec = infer.structured_input_signature[1]['image_input']
            # input_spec est un TensorSpec avec shape comme (None, H, W, C)
            height, width = input_spec.shape[1], input_spec.shape[2]
            return (width, height)
        else:
            # Pour modèle Keras
            input_shape = model.input_shape
            # Format attendu: (batch, height, width, channels)
            height = input_shape[1]
            width = input_shape[2]
            return (width, height)
    except:
        # Fallback pour MobileNetV2 et modèles anciens
        return (192, 192)


def preprocess_frame(frame, input_size=(192, 192)):
    """Prétraite une frame pour le modèle"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    return frame_batch


def predict_frame(model, frame, input_size):
    """Fait une prédiction sur une frame"""
    input_data = preprocess_frame(frame, input_size)
    
    if hasattr(model, 'is_saved_model') and model.is_saved_model:
        # Pour SavedModel
        import tensorflow as tf
        infer = model.signatures['serving_default']
        result = infer(image_input=tf.convert_to_tensor(input_data))
        heatmaps = result['output_0'].numpy()[0]
    else:
        # Pour modèle Keras
        heatmaps = model.predict(input_data, verbose=0)[0]
    
    return heatmaps


def extract_keypoints_from_heatmaps(heatmaps, frame_shape, expected_heatmap_size=None):
    """Extrait les coordonnées des keypoints depuis les heatmaps"""
    h, w = frame_shape[:2]
    keypoints = []
    
    # Utiliser la taille attendue si fournie, sinon utiliser la taille réelle
    if expected_heatmap_size:
        heatmap_h, heatmap_w = expected_heatmap_size
    else:
        heatmap_h, heatmap_w = heatmaps.shape[0], heatmaps.shape[1]

    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Convertir en coordonnées de l'image en utilisant la taille attendue
        y = int(max_pos[0] * h / heatmap_h)
        x = int(max_pos[1] * w / heatmap_w)
        confidence = heatmap[max_pos]
        keypoints.append({'x': x, 'y': y, 'confidence': confidence})

    return keypoints


def draw_keypoints(frame, keypoints, labels=None):
    """Dessine les keypoints sur la frame"""
    if labels is None:
        labels = config.BODYPARTS

    colors = [
        (255, 0, 0),    # Hanche - Rouge
        (0, 255, 0),    # Genoux - Vert
        (0, 0, 255)     # Cheville - Bleu
    ]

    for i, kp in enumerate(keypoints):
        color = colors[i % len(colors)]

        # Dessiner le point
        cv2.circle(frame, (kp['x'], kp['y']), 8, color, -1)
        cv2.circle(frame, (kp['x'], kp['y']), 10, (255, 255, 255), 2)

        # Ajouter le label et la confiance
        label = f"{labels[i]}: {kp['confidence']:.2f}"
        cv2.putText(frame, label, (kp['x'] + 15, kp['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Dessiner les connexions (squelette)
    if len(keypoints) >= 3:
        # Hanche -> Genoux
        cv2.line(frame, (keypoints[0]['x'], keypoints[0]['y']), 
                (keypoints[1]['x'], keypoints[1]['y']), (255, 255, 0), 2)
        # Genoux -> Cheville
        cv2.line(frame, (keypoints[1]['x'], keypoints[1]['y']), 
                (keypoints[2]['x'], keypoints[2]['y']), (255, 255, 0), 2)

    return frame


def process_video(video_path, model_path, output_path=None):
    """Traite une vidéo complète"""
    print(f"💡 Utilisation du modèle: {model_path}")
    print(f"📹 Vidéo: {video_path}")

    # Charger la configuration du modèle
    model_config = load_model_config(model_path)
    expected_heatmap_size = None
    if model_config and 'heatmap_size' in model_config:
        expected_heatmap_size = tuple(model_config['heatmap_size'])
        print(f"🎯 Heatmap size attendu: {expected_heatmap_size}")

    # Charger le modèle
    model = load_keras_model(model_path)
    
    # Détecter la taille d'entrée du modèle
    input_size = get_model_input_size(model)
    print(f"📊 Taille d'entrée du modèle: {input_size[0]}x{input_size[1]}")

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

    # Propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📊 Propriétés de la vidéo:")
    print(f"   - Résolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Frames: {total_frames}")

    # Préparer la sortie
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"output/{video_name}_keras_annotated.mp4"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n💾 Sortie: {output_path}")

    frame_count = 0
    print("\n🔄 Traitement des frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction
        heatmaps = predict_frame(model, frame, input_size)

        # Extraire keypoints avec la taille attendue
        keypoints = extract_keypoints_from_heatmaps(heatmaps, (height, width), expected_heatmap_size)

        # Dessiner keypoints
        annotated_frame = frame.copy()
        draw_keypoints(annotated_frame, keypoints)

        # Écrire la frame
        out.write(annotated_frame)

        frame_count += 1

        # Afficher le progrès
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progrès: {progress:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()

    print(f"\n✅ Traitement terminé: {frame_count} frames traitées")
    print(f"🎉 Vidéo annotée sauvegardée: {output_path}")

    return output_path


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test du modèle Keras sur une vidéo")
    parser.add_argument('--video', required=True, help='Chemin vers la vidéo à analyser')
    parser.add_argument('--model', default=None,
                       help='Chemin vers le modèle Keras (.h5)')
    parser.add_argument('--output', help='Chemin de sortie pour la vidéo annotée')

    args = parser.parse_args()

    # Vérifier que la vidéo existe
    if not os.path.exists(args.video):
        print(f"❌ Vidéo non trouvée: {args.video}")
        return

    # Trouver le modèle si non spécifié
    if not args.model:
        # Chercher dans tous les dossiers de modèles
        output_dir = Path(config.OUTPUT_DIR)
        keras_models = []
        
        # Parcourir tous les dossiers de modèles
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                models_subdir = model_dir / "models"
                if models_subdir.exists():
                    keras_models.extend(list(models_subdir.glob("*.h5")))
        
        if keras_models:
            # Prendre le plus récent
            args.model = str(max(keras_models, key=os.path.getctime))
            print(f"💡 Utilisation du modèle Keras le plus récent: {args.model}")
        else:
            print("❌ Aucun modèle Keras (.h5) trouvé!")
            print("💡 Entraînez d'abord le modèle avec: python main.py")
            return

    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"❌ Modèle non trouvé: {args.model}")
        return

    # Output par défaut
    if not args.output:
        video_name = Path(args.video).stem
        
        # Utiliser le dossier videos du modèle actuel
        model_path = Path(args.model)
        model_dir = model_path.parent.parent  # Remonter de models/ vers le dossier du modèle
        videos_dir = model_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        args.output = str(videos_dir / f"{video_name}_keras_annotated.mp4")

    # Traiter la vidéo
    try:
        output_path = process_video(args.video, args.model, args.output)
        print("\n💡 Touches:")
        print("   - 'q': Quitter")
        print("   - 'espace': Pause/Resume")
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")


if __name__ == "__main__":
    main()
