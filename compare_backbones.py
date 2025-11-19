#!/usr/bin/env python3
"""
Script de comparaison automatique entre MobileNetV2 et EfficientNetLite0
Entraîne les deux modèles et compare les résultats
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def train_model(backbone, advanced_training=True):
    """
    Entraîne un modèle avec le backbone spécifié
    
    Args:
        backbone: Nom du backbone (MobileNetV2 ou EfficientNetLite0)
        advanced_training: Utiliser l'entraînement avancé
    
    Returns:
        model_dir: Dossier du modèle entraîné
    """
    print("\n" + "=" * 80)
    print(f"🏋️  ENTRAÎNEMENT {backbone.upper()}")
    print("=" * 80)
    
    if backbone == "EfficientNetLite0":
        # Utiliser le script dédié qui ajuste les tailles
        cmd = f"python train_efficientnet.py --backbone {backbone}"
        if advanced_training:
            cmd += " --advanced-training"
    else:
        # MobileNetV2 - utilisation directe
        cmd = f"python main.py --backbone {backbone} --save-data"
        if advanced_training:
            cmd += " --advanced-training"
    
    print(f"\n📝 Commande: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"\n❌ Erreur lors de l'entraînement de {backbone}")
        return None
    
    # Trouver le dernier dossier créé
    output_dir = Path("output")
    
    # Chercher selon le préfixe
    prefix = "MNv2" if backbone == "MobileNetV2" else "ENL0"
    model_dirs = sorted([d for d in output_dir.glob(f"{prefix}_*") if d.is_dir()], 
                       key=lambda x: x.stat().st_mtime, 
                       reverse=True)
    
    if model_dirs:
        print(f"\n✅ Modèle entraîné: {model_dirs[0].name}")
        return model_dirs[0]
    else:
        print(f"\n⚠️  Dossier du modèle non trouvé")
        return None


def extract_metrics(model_dir):
    """
    Extrait les métriques depuis les logs du modèle
    
    Args:
        model_dir: Dossier du modèle
    
    Returns:
        metrics: Dictionnaire des métriques
    """
    metrics = {
        "model_dir": str(model_dir),
        "backbone": "MobileNetV2" if "MNv2" in str(model_dir) else "EfficientNetLite0"
    }
    
    # Chercher le fichier CSV de logs
    logs_dir = model_dir / "logs"
    csv_files = list(logs_dir.glob("*_training_log.csv"))
    
    if csv_files:
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        
        # Extraire les meilleures métriques
        metrics["best_val_loss"] = df["val_loss"].min()
        metrics["best_val_mae"] = df["val_mae"].min()
        metrics["final_val_loss"] = df["val_loss"].iloc[-1]
        metrics["final_val_mae"] = df["val_mae"].iloc[-1]
        metrics["epochs_trained"] = len(df)
        
        # Trouver l'epoch du meilleur modèle
        best_epoch = df["val_loss"].idxmin()
        metrics["best_epoch"] = int(best_epoch) + 1
        metrics["best_train_loss"] = df.loc[best_epoch, "loss"]
        metrics["best_train_mae"] = df.loc[best_epoch, "mae"]
    
    # Taille des modèles
    models_dir = model_dir / "models"
    if (models_dir / "pose_model_best.h5").exists():
        h5_size = (models_dir / "pose_model_best.h5").stat().st_size / (1024 * 1024)
        metrics["keras_model_size_mb"] = round(h5_size, 2)
    
    if (models_dir / "pose_model_dynamic.tflite").exists():
        tflite_size = (models_dir / "pose_model_dynamic.tflite").stat().st_size / 1024
        metrics["tflite_model_size_kb"] = round(tflite_size, 2)
    
    return metrics


def compare_metrics(metrics_mobilenet, metrics_efficientnet):
    """
    Compare les métriques des deux modèles
    
    Args:
        metrics_mobilenet: Métriques MobileNetV2
        metrics_efficientnet: Métriques EfficientNetLite0
    """
    print("\n" + "=" * 80)
    print("📊 COMPARAISON DES RÉSULTATS")
    print("=" * 80)
    
    print(f"\n{'Métrique':<30} {'MobileNetV2':>20} {'EfficientNetLite0':>20} {'Amélioration':>15}")
    print("-" * 90)
    
    # Val Loss
    mb_loss = metrics_mobilenet.get("best_val_loss", 0)
    ef_loss = metrics_efficientnet.get("best_val_loss", 0)
    improvement = ((mb_loss - ef_loss) / mb_loss * 100) if mb_loss > 0 else 0
    symbol = "✅" if improvement > 0 else "⚠️"
    print(f"{'Val Loss (meilleur)':<30} {mb_loss:>20.6f} {ef_loss:>20.6f} {improvement:>13.1f}% {symbol}")
    
    # Val MAE
    mb_mae = metrics_mobilenet.get("best_val_mae", 0)
    ef_mae = metrics_efficientnet.get("best_val_mae", 0)
    improvement = ((mb_mae - ef_mae) / mb_mae * 100) if mb_mae > 0 else 0
    symbol = "✅" if improvement > 0 else "⚠️"
    print(f"{'Val MAE (meilleur)':<30} {mb_mae:>20.6f} {ef_mae:>20.6f} {improvement:>13.1f}% {symbol}")
    
    # Taille Keras
    mb_size = metrics_mobilenet.get("keras_model_size_mb", 0)
    ef_size = metrics_efficientnet.get("keras_model_size_mb", 0)
    diff = ef_size - mb_size
    symbol = "⚠️" if diff > 0 else "✅"
    print(f"{'Taille Keras (MB)':<30} {mb_size:>20.2f} {ef_size:>20.2f} {diff:>13.2f} MB {symbol}")
    
    # Taille TFLite
    mb_tflite = metrics_mobilenet.get("tflite_model_size_kb", 0)
    ef_tflite = metrics_efficientnet.get("tflite_model_size_kb", 0)
    diff = ef_tflite - mb_tflite
    symbol = "⚠️" if diff > 0 else "✅"
    print(f"{'Taille TFLite (KB)':<30} {mb_tflite:>20.2f} {ef_tflite:>20.2f} {diff:>13.2f} KB {symbol}")
    
    # Epochs
    mb_epochs = metrics_mobilenet.get("best_epoch", 0)
    ef_epochs = metrics_efficientnet.get("best_epoch", 0)
    print(f"{'Meilleur epoch':<30} {mb_epochs:>20} {ef_epochs:>20}")
    
    print("\n" + "=" * 80)
    
    # Résumé
    print("\n💡 RÉSUMÉ:")
    if ef_loss < mb_loss:
        print(f"   ✅ EfficientNetLite0 obtient {((mb_loss - ef_loss) / mb_loss * 100):.1f}% de réduction de val_loss")
    else:
        print(f"   ⚠️  MobileNetV2 obtient de meilleurs résultats en val_loss")
    
    if ef_mae < mb_mae:
        print(f"   ✅ EfficientNetLite0 obtient {((mb_mae - ef_mae) / mb_mae * 100):.1f}% de réduction de val_mae")
    else:
        print(f"   ⚠️  MobileNetV2 obtient de meilleurs résultats en val_mae")
    
    print(f"   📦 EfficientNetLite0 est {(ef_size / mb_size):.1f}x plus lourd en Keras")
    print(f"   📦 EfficientNetLite0 est {(ef_tflite / mb_tflite):.1f}x plus lourd en TFLite")
    
    print("\n" + "=" * 80)


def save_comparison_report(metrics_mobilenet, metrics_efficientnet, output_file="comparison_report.json"):
    """Sauvegarde le rapport de comparaison"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "mobilenet_v2": metrics_mobilenet,
        "efficientnet_lite0": metrics_efficientnet
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Rapport sauvegardé: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MobileNetV2 vs EfficientNetLite0"
    )
    
    parser.add_argument(
        '--skip-mobilenet',
        action='store_true',
        help="Sauter l'entraînement de MobileNetV2 (utilise le dernier modèle)"
    )
    
    parser.add_argument(
        '--skip-efficientnet',
        action='store_true',
        help="Sauter l'entraînement de EfficientNetLite0 (utilise le dernier modèle)"
    )
    
    parser.add_argument(
        '--advanced-training',
        action='store_true',
        default=True,
        help="Utiliser l'entraînement avancé (activé par défaut)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 COMPARAISON AUTOMATIQUE: MOBILENETV2 vs EFFICIENTNETLITE0")
    print("=" * 80)
    
    # Entraîner ou charger MobileNetV2
    if not args.skip_mobilenet:
        mobilenet_dir = train_model("MobileNetV2", args.advanced_training)
    else:
        output_dir = Path("output")
        mobilenet_dirs = sorted([d for d in output_dir.glob("MNv2_*") if d.is_dir()], 
                               key=lambda x: x.stat().st_mtime, 
                               reverse=True)
        mobilenet_dir = mobilenet_dirs[0] if mobilenet_dirs else None
        print(f"\n📂 Utilisation de: {mobilenet_dir}")
    
    if not mobilenet_dir:
        print("❌ Impossible de trouver un modèle MobileNetV2")
        sys.exit(1)
    
    # Entraîner ou charger EfficientNetLite0
    if not args.skip_efficientnet:
        efficientnet_dir = train_model("EfficientNetLite0", args.advanced_training)
    else:
        output_dir = Path("output")
        efficientnet_dirs = sorted([d for d in output_dir.glob("ENL0_*") if d.is_dir()], 
                                   key=lambda x: x.stat().st_mtime, 
                                   reverse=True)
        efficientnet_dir = efficientnet_dirs[0] if efficientnet_dirs else None
        print(f"\n📂 Utilisation de: {efficientnet_dir}")
    
    if not efficientnet_dir:
        print("❌ Impossible de trouver un modèle EfficientNetLite0")
        sys.exit(1)
    
    # Extraire les métriques
    print("\n📈 Extraction des métriques...")
    metrics_mobilenet = extract_metrics(mobilenet_dir)
    metrics_efficientnet = extract_metrics(efficientnet_dir)
    
    # Comparer
    compare_metrics(metrics_mobilenet, metrics_efficientnet)
    
    # Sauvegarder le rapport
    save_comparison_report(metrics_mobilenet, metrics_efficientnet)
    
    print("\n✅ Comparaison terminée!")
    print("\n💡 Prochaines étapes:")
    print("   1. Tester sur vidéo: python quick_compare.py")
    print("   2. Export CSV: python export_video_analysis.py --video test.mp4")
    print("   3. Voir le rapport: cat comparison_report.json")


if __name__ == "__main__":
    main()
