#!/usr/bin/env python3
"""
Script d'entraînement pour EfficientNetLite avec ajustements automatiques
Permet une comparaison équitable entre backbones en ajustant minimalement les paramètres
"""
import os
import sys
import argparse
import shutil
from pathlib import Path


def calculate_heatmap_size(image_size, original_image_size=192, original_heatmap_size=64):
    """
    Calcule la taille proportionnelle des heatmaps
    
    Args:
        image_size: Nouvelle taille d'image
        original_image_size: Taille d'image de référence (192 pour MobileNetV2)
        original_heatmap_size: Taille heatmap de référence (64)
    
    Returns:
        heatmap_size: Taille heatmap proportionnelle arrondie
    """
    ratio = image_size / original_image_size
    heatmap_size = int(original_heatmap_size * ratio)
    # Arrondir au multiple de 8 le plus proche pour optimisation GPU
    heatmap_size = ((heatmap_size + 7) // 8) * 8
    return heatmap_size


def modify_config(backbone, image_size, heatmap_size):
    """
    Modifie temporairement config.py pour adapter les tailles
    
    Args:
        backbone: Nom du backbone
        image_size: Taille des images d'entrée
        heatmap_size: Taille des heatmaps
    
    Returns:
        backup_path: Chemin du fichier de backup
    """
    config_path = Path("config.py")
    backup_path = Path("config.py.backup")
    
    # Sauvegarder l'original
    shutil.copy(config_path, backup_path)
    
    # Lire le contenu
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Modifier HEATMAP_SIZE
    content = content.replace(
        'HEATMAP_SIZE = (64, 64)',
        f'HEATMAP_SIZE = ({heatmap_size}, {heatmap_size})'
    )
    
    # Écrire les modifications
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Configuration temporairement modifiée:")
    print(f"   - HEATMAP_SIZE: ({heatmap_size}, {heatmap_size})")
    
    return backup_path


def restore_config(backup_path):
    """Restaure la configuration originale"""
    config_path = Path("config.py")
    if backup_path.exists():
        shutil.move(backup_path, config_path)
        print("✅ Configuration originale restaurée")


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement EfficientNetLite avec ajustements automatiques"
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='EfficientNetLite0',
        choices=[
            'EfficientNetLite0', 'EfficientNetLite1', 'EfficientNetLite2',
            'EfficientNetLite3', 'EfficientNetLite4'
        ],
        help="Backbone EfficientNetLite à utiliser"
    )
    
    parser.add_argument(
        '--advanced-training',
        action='store_true',
        help="Utiliser l'entraînement avancé (Mixup, CutMix, SWA, etc.)"
    )
    
    parser.add_argument(
        '--skip-save-data',
        action='store_true',
        help="Ne pas sauvegarder les données prétraitées"
    )
    
    args = parser.parse_args()
    
    # Tailles recommandées par backbone
    backbone_sizes = {
        'EfficientNetLite0': 224,
        'EfficientNetLite1': 240,
        'EfficientNetLite2': 260,
        'EfficientNetLite3': 280,
        'EfficientNetLite4': 300,
    }
    
    image_size = backbone_sizes[args.backbone]
    heatmap_size = calculate_heatmap_size(image_size)
    
    print("=" * 60)
    print(f"🚀 ENTRAÎNEMENT {args.backbone.upper()}")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   - Backbone: {args.backbone}")
    print(f"   - Image size: {image_size}x{image_size}")
    print(f"   - Heatmap size: {heatmap_size}x{heatmap_size}")
    print(f"   - Entraînement avancé: {'Oui' if args.advanced_training else 'Non'}")
    print()
    
    # Modifier config.py temporairement
    print("🔧 Modification de la configuration...")
    backup_path = modify_config(args.backbone, image_size, heatmap_size)
    
    try:
        # Construire la commande d'entraînement
        cmd_parts = [
            "python", "main.py",
            "--backbone", args.backbone
        ]
        
        if not args.skip_save_data:
            cmd_parts.append("--save-data")
        
        if args.advanced_training:
            cmd_parts.append("--advanced-training")
        
        cmd = " ".join(cmd_parts)
        
        print("\n" + "=" * 60)
        print("🏋️  LANCEMENT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        print(f"\n📝 Commande: {cmd}\n")
        
        # Lancer l'entraînement
        exit_code = os.system(cmd)
        
        if exit_code == 0:
            print("\n" + "=" * 60)
            print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
            print("=" * 60)
            print(f"\n💡 Résultats sauvegardés dans: output/ENL*_YYYYMMDD_HHMMSS/")
            print("\n📊 Pour comparer les performances:")
            print("   python quick_compare.py")
            print()
        else:
            print("\n❌ Erreur lors de l'entraînement")
            sys.exit(exit_code)
    
    finally:
        # Toujours restaurer la configuration
        print("\n🔄 Restauration de la configuration...")
        restore_config(backup_path)


if __name__ == "__main__":
    main()
