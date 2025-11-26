"""
Analyse détaillée des erreurs par articulation (keypoint)
Calcule PCK, MSE et distribution d'erreurs pour chaque articulation séparément
"""
import os
import numpy as np
import pandas as pd
import cv2
import argparse
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Charge le modèle Keras avec compatibilité TensorFlow 2.15+"""
    from tensorflow import keras
    import tensorflow as tf
    print(f"🔄 Chargement du modèle: {model_path}")
    
    # Essayer d'abord le SavedModel si disponible
    saved_model_path = model_path.replace('_finetune_best.h5', '_saved_model')
    if os.path.exists(saved_model_path):
        print(f"🔄 Chargement SavedModel: {saved_model_path}")
        try:
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

    return model


def extract_keypoints_from_heatmaps(heatmaps, heatmap_size=(96, 96)):
    """
    Extrait les coordonnées des keypoints depuis les heatmaps

    Args:
        heatmaps: Array (H, W, num_keypoints) ou (num_keypoints, H, W)
        heatmap_size: Tuple (height, width)

    Returns:
        keypoints: Liste de (x_norm, y_norm, confidence) pour chaque keypoint
    """
    keypoints = []

    # Gérer les deux formats possibles
    if heatmaps.shape[-1] == len(config.BODYPARTS):
        # Format (H, W, num_keypoints)
        num_keypoints = heatmaps.shape[-1]
        h, w = heatmaps.shape[:2]
    else:
        # Format (num_keypoints, H, W)
        num_keypoints = heatmaps.shape[0]
        h, w = heatmaps.shape[1:3]
        heatmaps = np.transpose(heatmaps, (1, 2, 0))  # Convertir en (H, W, num_keypoints)

    for k in range(num_keypoints):
        heatmap = heatmaps[:, :, k]

        # Trouver la position du maximum
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y_idx, x_idx = max_pos

        # Convertir en coordonnées normalisées [0, 1]
        x_norm = x_idx / (w - 1)
        y_norm = y_idx / (h - 1)

        # Confiance = valeur maximale de la heatmap
        confidence = float(heatmap[max_pos])

        keypoints.append((x_norm, y_norm, confidence))

    return keypoints


def calculate_keypoint_errors(pred_keypoints, gt_keypoints, keypoint_names):
    """
    Calcule les erreurs pour chaque keypoint individuellement

    Args:
        pred_keypoints: Liste de (x, y, confidence) prédits
        gt_keypoints: Liste de (x, y) ground truth
        keypoint_names: Liste des noms des keypoints

    Returns:
        errors: Dict avec erreurs par keypoint
    """
    errors = {}

    for i, (pred, gt, name) in enumerate(zip(pred_keypoints, gt_keypoints, keypoint_names)):
        if gt is not None:
            pred_x, pred_y = pred[0], pred[1]
            gt_x, gt_y = gt

            # Distance euclidienne normalisée
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

            # Erreur absolue en X et Y
            error_x = abs(pred_x - gt_x)
            error_y = abs(pred_y - gt_y)

            errors[name] = {
                'distance': distance,
                'error_x': error_x,
                'error_y': error_y,
                'confidence': pred[2],
                'pred': (pred_x, pred_y),
                'gt': (gt_x, gt_y)
            }
        else:
            errors[name] = None

    return errors


def analyze_keypoint_performance(model, model_config, test_data_dir):
    """
    Analyse détaillée des performances par keypoint

    Args:
        model: Modèle chargé
        model_config: Configuration du modèle
        test_data_dir: Dossier des données de test

    Returns:
        results: Dict avec analyse détaillée
    """
    print("🔍 Analyse des erreurs par articulation...")

    # Mapping des articulations pour gérer les différences de nommage
    bodypart_mapping = {
        'Cuisse': 'Hanche',  # Cuisse -> Hanche
        'Pied': 'Cheville',  # Pied -> Cheville
        'Genoux': 'Genoux'   # Genoux reste Genoux
    }

    # Initialiser les collecteurs de données
    keypoint_errors = defaultdict(list)

    # Seuils PCK à tester
    pck_thresholds = [0.02, 0.05, 0.1, 0.15]  # 2%, 5%, 10%, 15%

    # Initialiser keypoint_pck avec toutes les clés nécessaires
    def init_pck_dict():
        d = {}
        for threshold in pck_thresholds:
            pct = int(threshold * 100)
            d[f'correct_{pct}'] = 0
            d[f'total_{pct}'] = 0
        return d

    keypoint_pck = defaultdict(init_pck_dict)

    # Parcourir tous les dossiers de test
    test_path = Path(test_data_dir)
    for folder in test_path.iterdir():
        if not folder.is_dir():
            continue

        print(f"\n📁 Analyse du dossier: {folder.name}")

        # Trouver le fichier CSV
        csv_files = list(folder.glob("CollectedData*.csv"))
        if not csv_files:
            continue

        # Charger les annotations
        df = pd.read_csv(csv_files[0], header=[0, 1, 2])

        # Traiter chaque image
        for idx, row in tqdm(df.iterrows(), desc=f"Traitement {folder.name}"):
            image_name = row.iloc[2]
            image_path = os.path.join(test_data_dir, folder.name, image_name)

            if not os.path.exists(image_path):
                continue

            # Charger et prétraiter l'image
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Redimensionner selon la config du modèle
            target_size = (model_config['input_size'][1], model_config['input_size'][0])  # (W, H)
            image_resized = cv2.resize(image, target_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            input_batch = np.expand_dims(image_normalized, axis=0)

            # Prédiction
            if hasattr(model, 'is_saved_model') and model.is_saved_model:
                # Pour SavedModel
                import tensorflow as tf
                infer = model.signatures['serving_default']
                result = infer(image_input=tf.convert_to_tensor(input_batch))
                predictions = result['output_0'].numpy()[0]
            else:
                # Pour modèle Keras
                predictions = model.predict(input_batch, verbose=0)[0]  # Shape: (H, W, num_keypoints)

            # Extraire les keypoints prédits
            pred_keypoints = extract_keypoints_from_heatmaps(predictions, tuple(model_config['heatmap_size']))

            # Extraire les keypoints ground truth
            gt_keypoints = []
            try:
                for bodypart in model_config['bodyparts']:
                    # Utiliser le mapping inverse pour les données de test
                    data_bodypart = None
                    for data_name, model_name in bodypart_mapping.items():
                        if model_name == bodypart:
                            data_bodypart = data_name
                            break
                    if data_bodypart is None:
                        data_bodypart = bodypart  # Fallback si pas de mapping

                    x = float(row[('jules', data_bodypart, 'x')])
                    y = float(row[('jules', data_bodypart, 'y')])

                    # Normaliser par rapport à la taille originale
                    orig_h, orig_w = image.shape[:2]
                    x_norm = x / orig_w
                    y_norm = y / orig_h

                    gt_keypoints.append((x_norm, y_norm))
            except Exception as e:
                print(f"DEBUG: Exception when extracting keypoints: {e}")
                continue  # Skip si données manquantes

            if len(gt_keypoints) != len(pred_keypoints):
                continue

            # Calculer les erreurs par keypoint
            errors = calculate_keypoint_errors(pred_keypoints, gt_keypoints, model_config['bodyparts'])

            # Accumuler les erreurs
            for kp_name, error_data in errors.items():
                if error_data is not None:
                    # Utiliser le mapping pour convertir les noms d'articulations
                    mapped_name = bodypart_mapping.get(kp_name, kp_name)
                    keypoint_errors[mapped_name].append(error_data)

                    # Calculer PCK pour différents seuils
                    for threshold in pck_thresholds:
                        if error_data['distance'] < threshold:
                            keypoint_pck[kp_name][f'correct_{int(threshold*100)}'] += 1
                        keypoint_pck[kp_name][f'total_{int(threshold*100)}'] += 1

    # Calculer les statistiques finales
    results = {}

    for kp_name in model_config['bodyparts']:
        if kp_name in keypoint_errors and keypoint_errors[kp_name]:
            errors_list = keypoint_errors[kp_name]

            # Statistiques de distance
            distances = [e['distance'] for e in errors_list]
            errors_x = [e['error_x'] for e in errors_list]
            errors_y = [e['error_y'] for e in errors_list]
            confidences = [e['confidence'] for e in errors_list]

            # Calculer PCK
            pck_scores = {}
            for threshold in pck_thresholds:
                threshold_key = int(threshold * 100)
                correct = keypoint_pck[kp_name][f'correct_{threshold_key}']
                total = keypoint_pck[kp_name][f'total_{threshold_key}']
                pck_scores[f'pck_{threshold_key}'] = correct / total if total > 0 else 0

            results[kp_name] = {
                'count': len(errors_list),
                'distance_stats': {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'median': np.median(distances),
                    'min': np.min(distances),
                    'max': np.max(distances),
                    'percentiles': {
                        '25': np.percentile(distances, 25),
                        '75': np.percentile(distances, 75),
                        '90': np.percentile(distances, 90),
                        '95': np.percentile(distances, 95)
                    }
                },
                'error_x_stats': {
                    'mean': np.mean(errors_x),
                    'std': np.std(errors_x)
                },
                'error_y_stats': {
                    'mean': np.mean(errors_y),
                    'std': np.std(errors_y)
                },
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences)
                },
                'pck_scores': pck_scores,
                'raw_errors': errors_list  # Pour analyses supplémentaires
            }

    return results


def plot_error_distributions(results, save_path=None):
    """Crée des graphiques de distribution d'erreurs"""
    if not results:
        return

    # Préparer les données pour le graphique
    keypoint_names = []
    mean_errors = []
    std_errors = []

    for kp_name, data in results.items():
        keypoint_names.append(kp_name)
        mean_errors.append(data['distance_stats']['mean'])
        std_errors.append(data['distance_stats']['std'])

    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1: Erreurs moyennes par articulation
    bars = ax1.bar(keypoint_names, mean_errors, yerr=std_errors, capsize=5,
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_ylabel('Erreur moyenne (unités normalisées)')
    ax1.set_title('Erreur moyenne par articulation')
    ax1.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar, mean, std in zip(bars, mean_errors, std_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                '.3f', ha='center', va='bottom', fontsize=10)

    # Graphique 2: Distribution des erreurs (boxplot)
    all_errors = []
    labels = []
    for kp_name, data in results.items():
        distances = [e['distance'] for e in data['raw_errors']]
        all_errors.append(distances)
        labels.append(f'{kp_name}\n(n={len(distances)})')

    bp = ax2.boxplot(all_errors, tick_labels=labels, patch_artist=True)
    ax2.set_ylabel('Erreur (unités normalisées)')
    ax2.set_title('Distribution des erreurs par articulation')
    ax2.grid(True, alpha=0.3)

    # Colorer les boxplots
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {save_path}")

    plt.show()


def print_detailed_results(results):
    """Affiche les résultats détaillés"""
    print("\n" + "=" * 80)
    print("📊 ANALYSE DÉTAILLÉE DES ERREURS PAR ARTICULATION")
    print("=" * 80)

    if not results:
        print("❌ AUCUN RÉSULTAT À AFFICHER")
        print("   Vérifiez que les données de test contiennent les articulations attendues")
        return

    print(f"📊 Nombre d'articulations analysées: {len(results)}")

    for kp_name, data in results.items():
        print(f"\n🔹 {kp_name.upper()}")
        print("-" * 40)
        print(f"   📏 Nombre d'échantillons: {data['count']}")

        print(f"\n   📊 Erreur de distance:")
        stats = data['distance_stats']
        print(f"      Moyenne: {stats['mean']:.4f}")
        print(f"      Écart-type: {stats['std']:.4f}")
        print(f"      Médiane: {stats['median']:.4f}")
        print(f"      Min: {stats['min']:.4f}")
        print(f"      Max: {stats['max']:.4f}")
        print("   📊 Percentiles:")
        pct = stats['percentiles']
        print(f"      25%: {pct['25']:.4f}")
        print(f"      75%: {pct['75']:.4f}")
        print(f"      90%: {pct['90']:.4f}")
        print(f"      95%: {pct['95']:.4f}")
        print(f"\n   📊 Erreur en X: {data['error_x_stats']['mean']:.4f} ± {data['error_x_stats']['std']:.4f}")
        print(f"   📊 Erreur en Y: {data['error_y_stats']['mean']:.4f} ± {data['error_y_stats']['std']:.4f}")
        print(f"   📊 Confiance moyenne: {data['confidence_stats']['mean']:.3f} ± {data['confidence_stats']['std']:.3f}")

        print(f"\n   🎯 Scores PCK:")
        for threshold, score in data['pck_scores'].items():
            pct = threshold.replace('pck_', '') + '%'
            print(f"      PCK@{pct}: {score:.1f}%")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyse détaillée des erreurs par articulation")
    parser.add_argument('--model-dir', type=str, required=True,
                       help="Dossier contenant le modèle (avec models/ et model_config.json)")
    parser.add_argument('--model-name', type=str, default='pose_model_dlc_finetune_best.h5',
                       help="Nom du fichier modèle .h5")
    parser.add_argument('--test-data-dir', type=str, default='test-labeled-data',
                       help="Dossier contenant les données de test annotées")
    parser.add_argument('--plot', action='store_true',
                       help="Générer des graphiques de distribution d'erreurs")
    parser.add_argument('--save-plot', type=str, default=None,
                       help="Chemin pour sauvegarder le graphique")

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

    model = load_keras_model(model_path)

    if model is None:
        print("\n🛑 Analyse interrompue à cause de l'erreur de chargement du modèle")
        print("Veuillez résoudre le problème de compatibilité avant de continuer.")
        return

    # Analyser les performances par keypoint
    results = analyze_keypoint_performance(model, model_config, args.test_data_dir)

    # Afficher les résultats
    print_detailed_results(results)

    # Générer les graphiques si demandé
    if args.plot or args.save_plot:
        plot_error_distributions(results, args.save_plot)


if __name__ == "__main__":
    main()