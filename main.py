"""
Script principal pour le pipeline DeepLabCut-Style
Usage: python main.py --backbone MobileNetV3Small --save-data
"""
import os
import argparse
import numpy as np
from datetime import datetime
import config
from data_preprocessing import prepare_data
from model_deeplabcut import create_deeplabcut_model
from train_deeplabcut import train_deeplabcut_progressive
from train_deeplabcut import save_final_model, plot_training_history
from export_tflite import export_model, test_tflite_model


def main(args):
    """Pipeline complet DeepLabCut-Style"""
    print("\n" + "=" * 80)
    print("🔬 PIPELINE DEEPLABCUT-STYLE - POSE ESTIMATION")
    print("=" * 80)

    # Configurer le backbone (utilise la valeur par défaut de config.BACKBONE)
    config.BACKBONE = args.backbone
    
    # Utiliser toujours les tailles DeepLabCut
    if args.backbone in config.DEEPLABCUT_INPUT_SIZES:
        recommended_size = config.DEEPLABCUT_INPUT_SIZES[args.backbone]
        config.IMAGE_SIZE = recommended_size
        config.INPUT_SHAPE = (*recommended_size, 3)
        
        # Calculer heatmap size avec le stride
        heatmap_h = recommended_size[0] // config.DEEPLABCUT_HEATMAP_STRIDE
        heatmap_w = recommended_size[1] // config.DEEPLABCUT_HEATMAP_STRIDE
        config.HEATMAP_SIZE = (heatmap_h, heatmap_w)
        
        print(f"\n📦 Backbone: {args.backbone}")
        print(f"📊 Input size: {recommended_size[0]}×{recommended_size[1]}")
        print(f"📊 Heatmap size: {heatmap_h}×{heatmap_w} (stride {config.DEEPLABCUT_HEATMAP_STRIDE})")
    else:
        # Fallback pour backbones non DeepLabCut
        print(f"\n⚠️  Backbone {args.backbone} non supporté en mode DeepLabCut, utilisation des paramètres par défaut")
        print(f"📦 Backbone: {args.backbone}")
        print(f"📊 Input size: {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]}")
        print(f"📊 Heatmap size: {config.HEATMAP_SIZE[0]}×{config.HEATMAP_SIZE[1]}")

    # ÉTAPE 0: Configuration des dossiers
    print("\n📁 CONFIGURATION DES DOSSIERS")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_folder_name = config.get_model_folder_name(config.BACKBONE, timestamp)
    model_dir, models_dir, logs_dir, videos_dir = config.setup_model_directories(model_folder_name)

    print(f"📂 Dossier modèle: {model_folder_name}")
    print(f"   - Modèles: {models_dir}")
    print(f"   - Logs: {logs_dir}")
    print(f"   - Vidéos: {videos_dir}")

    tflite_path = None  # Initialiser

    # ÉTAPE 1: Préparation des données
    if not args.skip_data_prep:
        print("\nÉTAPE 1/4 - PRÉPARATION DES DONNÉES")
        X_train, X_val, y_train, y_val = prepare_data()

        if args.save_data:
            data_path = os.path.join(model_dir, "preprocessed_data.npz")
            np.savez_compressed(data_path, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
            print(f"💾 Données sauvegardées: {data_path}")
    else:
        print("\n⏩ Chargement des données prétraitées...")
        data_path = os.path.join(model_dir, "preprocessed_data.npz")
        data = np.load(data_path)
        X_train = data['X_train']
        X_val = data['X_val']
        y_train = data['y_train']
        y_val = data['y_val']
        print(f"✅ Données chargées depuis: {data_path}")
    
    # ÉTAPE 2: Construction du modèle
    if not args.skip_training:
        print("\nÉTAPE 2/4 - CONSTRUCTION DU MODÈLE")
        model = create_deeplabcut_model()

        # ÉTAPE 3: Entraînement
        print("\nÉTAPE 3/4 - ENTRAÎNEMENT DEEPLABCUT-STYLE")
        model_name = "pose_model_dlc"  # Nom pour DeepLabCut

        history, metrics = train_deeplabcut_progressive(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            model_dir=model_dir
        )
        
        final_model_path, saved_model_dir = save_final_model(model, model_name, model_dir)

        if args.plot_history:
            plot_path = os.path.join(logs_dir, f"{model_name}_history.png")
            plot_training_history(history, save_path=plot_path)
    else:
        print("\n⏩ Chargement du modèle entraîné...")
        model_path = args.model_path
        if not model_path:
            raise ValueError("Vous devez fournir --model_path si --skip_training est activé")
        saved_model_dir = model_path
        model_name = "pose_model"
        print(f"✅ Modèle chargé depuis: {saved_model_dir}")

    # ÉTAPE 4: Export TFLite
    tflite_paths = None
    if not args.skip_export:
        print("\nÉTAPE 4/4 - EXPORT TENSORFLOW LITE")
        tflite_paths = export_model(model_path=saved_model_dir, X_val=X_val, model_name=model_name, model_dir=model_dir)

        if args.test_tflite:
            # Tester le modèle recommandé (dynamic)
            test_tflite_model(tflite_paths['dynamic'], X_val, y_val, num_samples=10)
    
    # Résumé final
    print("\n" + "=" * 80)
    print("🎉 PIPELINE DEEPLABCUT TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\n📂 Résultats sauvegardés dans: {model_dir}")
    print(f"   - Modèles: {models_dir}")
    print(f"   - Logs: {logs_dir}")
    print(f"   - Vidéos: {videos_dir}")

    print("\n" + "=" * 80)


def parse_arguments():
    """
    Parse les arguments de la ligne de commande
    """
    parser = argparse.ArgumentParser(
        description="Pipeline DeepLabCut-Style pour la pose estimation"
    )
    
    # Options de workflow
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help="Sauter la préparation des données (charge depuis le cache)"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Sauter l'entraînement (utilise un modèle existant)"
    )
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help="Sauter l'export TFLite"
    )
    
    # Configuration du modèle
    parser.add_argument(
        '--backbone',
        type=str,
        default=config.BACKBONE,
        choices=[
            'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large',
            'EfficientNetLite0', 'EfficientNetLite1', 'EfficientNetLite2', 
            'EfficientNetLite3', 'EfficientNetLite4'
        ],
        help="Backbone à utiliser (défaut: MobileNetV3Small pour DeepLabCut)"
    )
    
    # Options de sauvegarde
    parser.add_argument(
        '--save-data',
        action='store_true',
        help="Sauvegarder les données prétraitées"
    )
    parser.add_argument(
        '--plot-history',
        action='store_true',
        default=True,
        help="Tracer les courbes d'apprentissage"
    )
    parser.add_argument(
        '--test-tflite',
        action='store_true',
        default=True,
        help="Tester le modèle TFLite après conversion"
    )
    
    # Chemins
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Chemin vers un modèle existant (si --skip-training)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parser les arguments
    args = parse_arguments()
    
    # Lancer le pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur lors de l'exécution du pipeline:")
        print(f"   {type(e).__name__}: {e}")
        raise
