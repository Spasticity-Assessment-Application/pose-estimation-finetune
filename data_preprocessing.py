"""
Prétraitement des données pour l'entraînement
"""
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config


def create_gaussian_heatmap(center_x, center_y, height, width, sigma=2.0):
    """Crée une heatmap gaussienne centrée sur un point"""
    center_x_px = int(center_x * width)
    center_y_px = int(center_y * height)

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    y = y[:, np.newaxis]

    heatmap = np.exp(-((x - center_x_px) ** 2 + (y - center_y_px) ** 2) / (2 * sigma ** 2))
    return heatmap


def load_image(image_path, target_size):
    """Charge et redimensionne une image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    if config.NORMALIZE:
        img = img.astype(np.float32) / 255.0

    return img


def parse_csv_file(csv_path, video_folder):
    """Parse un fichier CSV DeepLabCut et extrait les annotations"""
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    annotations = []

    for idx, row in df.iterrows():
        image_name = row.iloc[2]
        image_path = os.path.join(config.LABELED_DATA_DIR, video_folder, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️  Image non trouvée: {image_path}")
            continue
        
        # Extraire les coordonnées des keypoints
        keypoints = {}
        try:
            for bodypart, (x_idx, y_idx) in config.KEYPOINT_INDICES.items():
                x = float(row.iloc[x_idx])
                y = float(row.iloc[y_idx])
                
                # Vérifier si les coordonnées sont valides (pas NaN)
                if not (np.isnan(x) or np.isnan(y)):
                    keypoints[bodypart] = (x, y)
                else:
                    # Si un point est manquant, on skip cette annotation
                    keypoints = None
                    break
        except (ValueError, IndexError) as e:
            print(f"⚠️  Erreur de parsing pour {image_name}: {e}")
            continue
        
        if keypoints is not None and len(keypoints) == config.NUM_KEYPOINTS:
            annotations.append({
                'image_path': image_path,
                'keypoints': keypoints,
                'video_folder': video_folder
            })
    
    return annotations


def load_all_annotations():
    """
    Charge toutes les annotations de tous les dossiers labeled-data
    
    Returns:
        all_annotations: Liste de toutes les annotations
    """
    all_annotations = []
    
    # Lister tous les sous-dossiers dans labeled-data
    labeled_data_path = Path(config.LABELED_DATA_DIR)
    
    for folder in labeled_data_path.iterdir():
        if folder.is_dir() and not folder.name.endswith('_labeled'):
            # Chercher tous les fichiers CSV commençant par "CollectedData"
            csv_files = list(folder.glob("CollectedData*.csv"))
            
            if csv_files:
                # Prendre le premier fichier trouvé
                csv_file = csv_files[0]
                print(f"📂 Chargement des annotations de: {folder.name}")
                print(f"   📄 Fichier: {csv_file.name}")
                annotations = parse_csv_file(str(csv_file), folder.name)
                all_annotations.extend(annotations)
                print(f"   ✅ {len(annotations)} images annotées")
            else:
                print(f"⚠️  Aucun fichier CollectedData*.csv trouvé dans: {folder.name}")
    
    print(f"\n📊 Total: {len(all_annotations)} images annotées chargées")
    return all_annotations


def create_dataset(annotations, image_size, heatmap_size):
    """
    Crée les datasets d'images et de heatmaps
    
    Args:
        annotations: Liste des annotations
        image_size: Taille des images (height, width)
        heatmap_size: Taille des heatmaps (height, width)
    
    Returns:
        images: Array numpy (N, H, W, 3)
        heatmaps: Array numpy (N, H_hm, W_hm, num_keypoints)
    """
    images = []
    heatmaps = []
    
    print(f"\n🔄 Création du dataset...")
    
    for annotation in tqdm(annotations, desc="Traitement des images"):
        try:
            # Charger l'image
            img = load_image(annotation['image_path'], image_size)
            
            # Récupérer les dimensions originales pour normaliser les coordonnées
            original_img = cv2.imread(annotation['image_path'])
            orig_h, orig_w = original_img.shape[:2]
            
            # Créer les heatmaps pour chaque keypoint
            heatmap_stack = []
            for bodypart in config.BODYPARTS:
                x, y = annotation['keypoints'][bodypart]
                
                # Normaliser les coordonnées (0-1)
                x_norm = x / orig_w
                y_norm = y / orig_h
                
                # Créer la heatmap
                heatmap = create_gaussian_heatmap(
                    x_norm, y_norm,
                    heatmap_size[0], heatmap_size[1],
                    sigma=config.HEATMAP_SIGMA
                )
                heatmap_stack.append(heatmap)
            
            # Empiler les heatmaps (num_keypoints, H, W) -> (H, W, num_keypoints)
            heatmap_combined = np.stack(heatmap_stack, axis=-1)
            
            images.append(img)
            heatmaps.append(heatmap_combined)
            
        except Exception as e:
            print(f"\n⚠️  Erreur lors du traitement de {annotation['image_path']}: {e}")
            continue
    
    images = np.array(images, dtype=np.float32)
    heatmaps = np.array(heatmaps, dtype=np.float32)
    
    print(f"✅ Dataset créé:")
    print(f"   - Images: {images.shape}")
    print(f"   - Heatmaps: {heatmaps.shape}")
    
    return images, heatmaps


def split_dataset(images, heatmaps, train_split=0.8, random_seed=42):
    """
    Divise le dataset en train et validation
    
    Args:
        images: Array des images
        heatmaps: Array des heatmaps
        train_split: Proportion pour l'entraînement (0.8 = 80%)
        random_seed: Seed pour la reproductibilité
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        images, heatmaps,
        train_size=train_split,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\n📊 Split du dataset:")
    print(f"   - Train: {X_train.shape[0]} images")
    print(f"   - Validation: {X_val.shape[0]} images")
    
    return X_train, X_val, y_train, y_val


def prepare_data():
    """
    Pipeline complet de préparation des données
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    print("=" * 60)
    print("🚀 PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    # 1. Charger toutes les annotations
    annotations = load_all_annotations()
    
    if len(annotations) == 0:
        raise ValueError("Aucune annotation trouvée!")
    
    # 2. Créer le dataset (images + heatmaps)
    images, heatmaps = create_dataset(
        annotations,
        image_size=config.IMAGE_SIZE,
        heatmap_size=config.HEATMAP_SIZE
    )
    
    # 3. Diviser en train/val
    X_train, X_val, y_train, y_val = split_dataset(
        images, heatmaps,
        train_split=config.TRAIN_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    print("\n✅ Préparation des données terminée!")
    print("=" * 60)
    
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Test du prétraitement
    X_train, X_val, y_train, y_val = prepare_data()
    
    print("\n📊 Résumé final:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - y_train: {y_train.shape}")
    print(f"   - X_val: {X_val.shape}")
    print(f"   - y_val: {y_val.shape}")
