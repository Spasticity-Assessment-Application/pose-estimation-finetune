"""
Évaluation du modèle sur les données de test annotées dans test-labeled-data
"""
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tensorflow import keras
import argparse
import json
from tqdm import tqdm
import config


def load_model_config(model_dir):
    """Charge la configuration du modèle"""
    config_file = os.path.join(model_dir, "models", "model_config.json")

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        print(f"✅ Configuration chargée: {config_file}")
        return model_config
    else:
        print(f"⚠️  Configuration non trouvée: {config_file}")
        return None


def load_keras_model(model_path):
    """Charge le modèle Keras avec gestion d'erreurs de compatibilité"""
    print(f"🔄 Chargement du modèle Keras: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        print("✅ Modèle chargé et recompilé")
        return model, "keras"
    except Exception as e:
        print(f"⚠️  Erreur Keras: {e}")
        print("🔄 Tentative avec TFLite...")

        try:
            import tensorflow as tf

            # Essayer de trouver le fichier TFLite correspondant
            model_dir = os.path.dirname(model_path)
            model_base = os.path.basename(model_path).replace('.h5', '')

            # Essayer différents noms de fichiers TFLite
            possible_tflite_names = [
                f"{model_base}.tflite",
                f"{model_base.replace('_finetune_best', '_float32')}.tflite",
                f"{model_base.replace('_final', '_float32')}.tflite",
                "pose_model_dlc_float32.tflite"
            ]

            tflite_path = None
            for name in possible_tflite_names:
                candidate_path = os.path.join(model_dir, name)
                if os.path.exists(candidate_path):
                    tflite_path = candidate_path
                    break

            if tflite_path:
                print(f"🔄 Chargement du modèle TFLite: {tflite_path}")
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                print("✅ Modèle TFLite chargé")
                return interpreter, "tflite"
            else:
                print(f"❌ Fichier TFLite non trouvé: {tflite_path}")
                raise FileNotFoundError(f"Modèle TFLite non trouvé pour {model_path}")

        except Exception as e2:
            print(f"❌ Impossible de charger le modèle: {e2}")
            raise


def parse_test_csv(csv_path, video_folder, target_size=(256, 256)):
    """Parse le CSV de test et extrait les annotations"""
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    annotations = {}

    for idx, row in df.iterrows():
        image_name = row.iloc[2]
        image_path = os.path.join(video_folder, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️  Image non trouvée: {image_path}")
            continue

        # Obtenir la taille originale de l'image
        img = cv2.imread(image_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Extraire les coordonnées des keypoints
        keypoints = []
        try:
            for i in range(3):  # 3 keypoints
                x_idx = 3 + i * 2
                y_idx = 4 + i * 2
                x = float(row.iloc[x_idx])
                y = float(row.iloc[y_idx])

                if not (np.isnan(x) or np.isnan(y)):
                    # Normaliser les coordonnées par la taille originale
                    x_norm = x / orig_w
                    y_norm = y / orig_h
                    keypoints.append((x_norm, y_norm))
                else:
                    keypoints = None
                    break
        except (ValueError, IndexError) as e:
            print(f"⚠️  Erreur de parsing pour {image_name}: {e}")
            continue

        if keypoints is not None and len(keypoints) == 3:
            annotations[image_name] = {
                'image_path': image_path,
                'keypoints': keypoints,  # Coordonnées normalisées (0-1)
                'orig_size': (orig_w, orig_h)
            }

    return annotations


def preprocess_image(img_path, input_size):
    """Prétraite une image pour le modèle"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)

    if config.NORMALIZE:
        img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    return img


def predict_keypoints(model, model_type, img_path, input_size, heatmap_size):
    """Prédit les keypoints pour une image avec Keras ou TFLite"""
    input_data = preprocess_image(img_path, input_size)

    if model_type == "keras":
        heatmaps = model.predict(input_data, verbose=0)[0]
    elif model_type == "tflite":
        import tensorflow as tf

        # Préparer les tensors d'entrée
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Assurer le bon type de données
        input_data = input_data.astype(np.float32)

        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        heatmaps = model.get_tensor(output_details[0]['index'])[0]
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")

    keypoints = []

    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)

        # Coordonnées normalisées (0-1) sur l'image redimensionnée
        x_norm = max_pos[1] / heatmap_size[1]
        y_norm = max_pos[0] / heatmap_size[0]
        confidence = float(heatmap[max_pos])
        keypoints.append((x_norm, y_norm, confidence))

    return keypoints


def calculate_pck(pred_keypoints, gt_keypoints, threshold=0.05):
    """Calcule PCK (Percentage of Correct Keypoints)"""
    correct = 0
    total = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        if gt is not None:
            pred_x, pred_y = pred[0], pred[1]
            gt_x, gt_y = gt

            # Distance normalisée (0-1)
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

            if distance < threshold:  # Threshold en unités normalisées
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


def calculate_pck_multiple_thresholds(pred_keypoints, gt_keypoints, thresholds=[0.02, 0.05, 0.1, 0.15]):
    """Calcule PCK pour plusieurs seuils"""
    results = {}
    
    for threshold in thresholds:
        pck = calculate_pck(pred_keypoints, gt_keypoints, threshold)
        results[f"pck_{int(threshold*100)}"] = pck
    
    return results


def calculate_mse(pred_keypoints, gt_keypoints):
    """Calcule MSE entre prédictions et ground truth (en unités normalisées)"""
    mse = 0
    count = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        if gt is not None:
            pred_x, pred_y = pred[0], pred[1]
            gt_x, gt_y = gt
            mse += (pred_x - gt_x)**2 + (pred_y - gt_y)**2
            count += 1

    return mse / count if count > 0 else 0


def evaluate_model_on_test_data(model, model_type, model_config, test_data_dir, pck_threshold=0.05):
    """Évalue le modèle sur toutes les données de test"""
    input_size = tuple(model_config['input_size'])
    heatmap_size = tuple(model_config['heatmap_size'])

    results = {}

    # Lister tous les sous-dossiers dans test-labeled-data
    test_data_path = Path(test_data_dir)

    for folder in test_data_path.iterdir():
        if not folder.is_dir():
            continue

        print(f"\n📁 Évaluation du dossier: {folder.name}")

        # Chercher le fichier CSV
        csv_files = list(folder.glob("CollectedData*.csv"))
        if not csv_files:
            print(f"⚠️  Aucun fichier CSV trouvé dans {folder}")
            continue

        csv_path = csv_files[0]
        annotations = parse_test_csv(csv_path, folder)

        if not annotations:
            print(f"⚠️  Aucune annotation valide trouvée dans {csv_path}")
            continue

        print(f"📊 {len(annotations)} images annotées trouvées")

        # Évaluer chaque image
        pck_scores = []
        mse_scores = []
        confidences = []
        pck_multi_scores = {f"pck_{int(t*100)}": [] for t in [0.02, 0.05, 0.1, 0.15]}

        for image_name, ann in tqdm(annotations.items(), desc=f"Évaluation {folder.name}"):
            img_path = ann['image_path']
            gt_keypoints = ann['keypoints']  # Déjà normalisés

            try:
                pred_keypoints = predict_keypoints(model, model_type, img_path, input_size, heatmap_size)

                # Calculer PCK standard (avec seuil personnalisé)
                pck = calculate_pck(pred_keypoints, gt_keypoints, threshold=pck_threshold)
                pck_scores.append(pck)

                # Calculer PCK multiples seuils
                pck_multi = calculate_pck_multiple_thresholds(pred_keypoints, gt_keypoints)
                for key, value in pck_multi.items():
                    pck_multi_scores[key].append(value)

                # Calculer MSE
                mse = calculate_mse(pred_keypoints, gt_keypoints)
                mse_scores.append(mse)

                # Collecter les confiances
                confs = [kp[2] for kp in pred_keypoints]
                confidences.extend(confs)

            except Exception as e:
                print(f"❌ Erreur lors du traitement de {image_name}: {e}")
                continue

        # Résultats pour ce dossier
        if pck_scores:
            folder_results = {
                'num_images': len(pck_scores),
                'pck_mean': np.mean(pck_scores),
                'pck_std': np.std(pck_scores),
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'confidence_mean': np.mean(confidences),
                'confidence_std': np.std(confidences)
            }
            
            # Ajouter les PCK multiples seuils
            for key, scores in pck_multi_scores.items():
                folder_results[f"{key}_mean"] = np.mean(scores)
                folder_results[f"{key}_std"] = np.std(scores)
            
            results[folder.name] = folder_results

            print(f"   - PCK@{int(pck_threshold*100)}%: {folder_results['pck_mean']:.3f} ± {folder_results['pck_std']:.3f}")
            print(f"   - PCK@2%: {folder_results['pck_2_mean']:.3f} ± {folder_results['pck_2_std']:.3f}")
            print(f"   - PCK@10%: {folder_results['pck_10_mean']:.3f} ± {folder_results['pck_10_std']:.3f}")
            print(f"   - PCK@15%: {folder_results['pck_15_mean']:.3f} ± {folder_results['pck_15_std']:.3f}")
            print(f"   - MSE: {folder_results['mse_mean']:.4f} ± {folder_results['mse_std']:.4f} (normalisé)")
            print(f"   - Confiance moyenne: {folder_results['confidence_mean']:.2f} ± {folder_results['confidence_std']:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle sur données de test annotées")
    parser.add_argument('--model-dir', type=str, required=True,
                       help="Dossier contenant le modèle (avec models/ et model_config.json)")
    parser.add_argument('--model-name', type=str, default='pose_model_dlc_finetune_best.h5',
                       help="Nom du fichier modèle .h5")
    parser.add_argument('--test-data-dir', type=str, default='test-labeled-data',
                       help="Dossier contenant les données de test annotées")
    parser.add_argument('--pck-threshold', type=float, default=0.05,
                       help="Seuil PCK en unités normalisées (0-1), défaut 0.05 (5%)")

    args = parser.parse_args()

    # Charger la configuration du modèle
    model_config = load_model_config(args.model_dir)
    if model_config is None:
        print("❌ Impossible de charger la configuration du modèle")
        return

    # Charger le modèle
    model_path = os.path.join(args.model_dir, 'models', args.model_name)
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return

    model, model_type = load_keras_model(model_path)

    # Évaluer sur les données de test
    test_data_dir = args.test_data_dir
    if not os.path.exists(test_data_dir):
        print(f"❌ Dossier de test non trouvé: {test_data_dir}")
        return

    results = evaluate_model_on_test_data(model, model_type, model_config, test_data_dir, args.pck_threshold)

    # Afficher les résultats globaux
    print("\n" + "="*80)
    print("📊 RÉSULTATS GLOBAUX")
    print("="*80)

    if results:
        all_pck = []
        all_mse = []
        total_images = 0

        for folder, res in results.items():
            all_pck.extend([res['pck_mean']] * res['num_images'])  # Pondérer par nombre d'images
            all_mse.extend([res['mse_mean']] * res['num_images'])
            total_images += res['num_images']

        print(f"📈 Nombre total d'images évaluées: {total_images}")
        print(f"🎯 PCK moyen global: {np.mean(all_pck):.3f}")
        print(f"📏 MSE moyen global: {np.mean(all_mse):.4f} (normalisé)")
        print(f"🔍 Confiance moyenne globale: {np.mean([res['confidence_mean'] for res in results.values()]):.2f}")
    else:
        print("❌ Aucun résultat d'évaluation")


if __name__ == "__main__":
    main()