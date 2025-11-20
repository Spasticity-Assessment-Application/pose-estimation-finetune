"""
Script principal pour l'entraînement DeepLabCut-Style
Usage: python main_deeplabcut.py --backbone MobileNetV3Small --save-data
"""
import os
import argparse
import numpy as np
from datetime import datetime
import config
from data_preprocessing import prepare_data
from model_deeplabcut import create_deeplabcut_model
from train_deeplabcut import train_deeplabcut_progressive
from train import save_final_model, plot_training_history
from export_tflite import export_model


def main(args):
    """Pipeline complet DeepLabCut-Style"""
    print("\n" + "=" * 80)
    print("🔬 PIPELINE DEEPLABCUT-STYLE - POSE ESTIMATION")
    print("=" * 80)
    
    # Configurer le backbone
    if args.backbone:
        config.BACKBONE = args.backbone
        
        # Adapter les tailles pour DeepLabCut (images plus grandes)
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
    
    # Utiliser les augmentations DeepLabCut
    config.ADVANCED_AUGMENTATION = config.ADVANCED_AUGMENTATION_DEEPLABCUT
    config.BATCH_SIZE = config.DEEPLABCUT_BATCH_SIZE
    
    print(f"\n🎯 Configuration DeepLabCut:")
    print(f"   - Backbone: {config.BACKBONE}")
    print(f"   - Input: {config.INPUT_SHAPE[0]}×{config.INPUT_SHAPE[1]}")
    print(f"   - Heatmap: {config.HEATMAP_SIZE[0]}×{config.HEATMAP_SIZE[1]}")
    print(f"   - Batch size: {config.BATCH_SIZE}")
    print(f"   - Total epochs: {sum(config.DEEPLABCUT_EPOCHS.values())}")
    
    # Configuration des dossiers
    print("\n📁 CONFIGURATION DES DOSSIERS")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_folder_name = f"DLC_{config.BACKBONE[:5]}_{timestamp}"
    model_dir, models_dir, logs_dir, videos_dir = config.setup_model_directories(model_folder_name)
    
    print(f"📂 Dossier modèle: {model_folder_name}")
    print(f"   - Modèles: {models_dir}")
    print(f"   - Logs: {logs_dir}")
    
    # ÉTAPE 1: Préparation des données
    if not args.skip_data_prep:
        print("\n" + "=" * 80)
        print("ÉTAPE 1/4 - PRÉPARATION DES DONNÉES")
        print("=" * 80)
        X_train, X_val, y_train, y_val = prepare_data()
        
        if args.save_data:
            data_path = os.path.join(model_dir, "preprocessed_data.npz")
            np.savez_compressed(data_path, X_train=X_train, X_val=X_val, 
                              y_train=y_train, y_val=y_val)
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
    
    # ÉTAPE 2: Construction du modèle DeepLabCut
    if not args.skip_training:
        print("\n" + "=" * 80)
        print("ÉTAPE 2/4 - CONSTRUCTION DU MODÈLE DEEPLABCUT")
        print("=" * 80)
        model = create_deeplabcut_model()
        
        # Afficher le résumé
        if args.model_summary:
            model.summary()
        
        # ÉTAPE 3: Entraînement progressif DeepLabCut
        print("\n" + "=" * 80)
        print("ÉTAPE 3/4 - ENTRAÎNEMENT PROGRESSIF DEEPLABCUT")
        print("=" * 80)
        
        model_name = "pose_model_dlc"
        history, metrics = train_deeplabcut_progressive(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            model_dir=model_dir
        )
        
        # Sauvegarder le modèle final
        final_model_path, saved_model_dir = save_final_model(model, model_name, model_dir)
        
        # Tracer l'historique
        if args.plot_history:
            plot_path = os.path.join(logs_dir, f"{model_name}_history.png")
            
            # Créer un objet history-like pour plot_training_history
            class HistoryWrapper:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            plot_training_history(HistoryWrapper(history), save_path=plot_path)
            print(f"📊 Graphique d'entraînement sauvegardé: {plot_path}")
    
    else:
        print("\n⏩ Mode skip-training activé")
        if not args.model_path:
            raise ValueError("--model-path requis avec --skip-training")
        saved_model_dir = args.model_path
        model_name = "pose_model_dlc"
    
    # ÉTAPE 4: Export TFLite
    if not args.skip_export:
        print("\n" + "=" * 80)
        print("ÉTAPE 4/4 - EXPORT TENSORFLOW LITE")
        print("=" * 80)
        
        tflite_paths = export_model(
            model_path=saved_model_dir,
            X_val=X_val,
            model_name=model_name,
            model_dir=model_dir
        )
        
        print(f"\n✅ Modèles TFLite exportés:")
        for quant_type, path in tflite_paths.items():
            print(f"   - {quant_type}: {path}")
    
    # Résumé final
    print("\n" + "=" * 80)
    print("🎉 PIPELINE DEEPLABCUT TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\n📂 Résultats dans: {model_dir}")
    print(f"\n🧪 Pour tester le modèle:")
    print(f"   python test_video_keras.py --video test-videos/votre_video.mp4 \\")
    print(f"          --model {models_dir}/pose_model_dlc_finetune_best.h5")
    print("\n" + "=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pipeline DeepLabCut-Style pour pose estimation"
    )
    
    # Options de workflow
    parser.add_argument('--skip-data-prep', action='store_true',
                       help="Sauter la préparation des données")
    parser.add_argument('--skip-training', action='store_true',
                       help="Sauter l'entraînement")
    parser.add_argument('--skip-export', action='store_true',
                       help="Sauter l'export TFLite")
    
    # Configuration
    parser.add_argument('--backbone', type=str, default='MobileNetV3Small',
                       choices=['MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large', "EfficientNetLite0"],
                       help="Backbone à utiliser")
    
    # Options
    parser.add_argument('--save-data', action='store_true',
                       help="Sauvegarder les données prétraitées")
    parser.add_argument('--plot-history', action='store_true', default=True,
                       help="Tracer les courbes d'apprentissage")
    parser.add_argument('--model-summary', action='store_true',
                       help="Afficher le résumé du modèle")
    parser.add_argument('--model-path', type=str, default=None,
                       help="Chemin vers un modèle existant")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrompu")
    except Exception as e:
        print(f"\n\n❌ Erreur: {type(e).__name__}: {e}")
        raise
