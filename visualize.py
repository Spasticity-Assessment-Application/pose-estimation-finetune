"""
Script pour visualiser les prédictions du modèle et les heatmaps
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import config


def visualize_heatmaps(image, heatmaps, keypoint_names=None, save_path=None):
    """
    Visualise une image avec ses heatmaps
    
    Args:
        image: Image RGB normalisée (H, W, 3)
        heatmaps: Heatmaps (H_hm, W_hm, num_keypoints)
        keypoint_names: Liste des noms des points clés
        save_path: Chemin pour sauvegarder la figure
    """
    if keypoint_names is None:
        keypoint_names = config.BODYPARTS
    
    num_keypoints = heatmaps.shape[-1]
    
    # Créer la figure
    fig, axes = plt.subplots(1, num_keypoints + 1, figsize=(4 * (num_keypoints + 1), 4))
    
    # Afficher l'image originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Afficher chaque heatmap
    for i in range(num_keypoints):
        heatmap = heatmaps[:, :, i]
        
        # Superposer la heatmap sur l'image
        # Redimensionner la heatmap à la taille de l'image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Créer une visualisation avec overlay
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(heatmap_resized, alpha=0.6, cmap='jet')
        axes[i + 1].set_title(f"{keypoint_names[i]}")
        axes[i + 1].axis('off')
        
        # Trouver le point maximum (prédiction du keypoint)
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        # Convertir en coordonnées de l'image
        y_img = int(max_pos[0] * image.shape[0] / heatmap.shape[0])
        x_img = int(max_pos[1] * image.shape[1] / heatmap.shape[1])
        axes[i + 1].plot(x_img, y_img, 'r*', markersize=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualisation sauvegardée: {save_path}")
    
    plt.show()


def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extrait les coordonnées des keypoints depuis les heatmaps
    
    Args:
        heatmaps: Heatmaps (H, W, num_keypoints)
    
    Returns:
        keypoints: Array (num_keypoints, 2) avec les coordonnées (x, y) normalisées
    """
    num_keypoints = heatmaps.shape[-1]
    keypoints = []
    
    for i in range(num_keypoints):
        heatmap = heatmaps[:, :, i]
        
        # Trouver le maximum
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Normaliser les coordonnées (0-1)
        y_norm = max_pos[0] / heatmap.shape[0]
        x_norm = max_pos[1] / heatmap.shape[1]
        
        keypoints.append([x_norm, y_norm])
    
    return np.array(keypoints)


def predict_and_visualize(model, image_path, save_path=None):
    """
    Fait une prédiction sur une image et visualise les résultats
    
    Args:
        model: Modèle Keras ou chemin vers le modèle
        image_path: Chemin vers l'image
        save_path: Chemin pour sauvegarder la visualisation
    """
    # Charger le modèle si c'est un chemin
    if isinstance(model, str):
        model = keras.models.load_model(model)
    
    # Charger et prétraiter l'image
    img_original = cv2.imread(image_path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_original, config.IMAGE_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Prédire
    input_batch = np.expand_dims(img_normalized, axis=0)
    heatmaps_pred = model.predict(input_batch, verbose=0)[0]
    
    # Visualiser
    visualize_heatmaps(img_normalized, heatmaps_pred, save_path=save_path)
    
    # Extraire les keypoints
    keypoints = extract_keypoints_from_heatmaps(heatmaps_pred)
    
    print("\n📍 Keypoints détectés (coordonnées normalisées):")
    for i, kp_name in enumerate(config.BODYPARTS):
        print(f"   {kp_name}: x={keypoints[i][0]:.3f}, y={keypoints[i][1]:.3f}")
    
    return heatmaps_pred, keypoints


def compare_predictions(model, X_val, y_val, num_samples=5, save_dir=None):
    """
    Compare les prédictions avec les ground truth sur plusieurs échantillons
    
    Args:
        model: Modèle Keras
        X_val: Images de validation
        y_val: Heatmaps de validation (ground truth)
        num_samples: Nombre d'échantillons à visualiser
        save_dir: Dossier pour sauvegarder les visualisations
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Prédire sur un batch
    predictions = model.predict(X_val[:num_samples], verbose=0)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(2, config.NUM_KEYPOINTS + 1, 
                                figsize=(4 * (config.NUM_KEYPOINTS + 1), 8))
        
        # Ligne 1: Ground Truth
        axes[0, 0].imshow(X_val[i])
        axes[0, 0].set_title("Image (GT)")
        axes[0, 0].axis('off')
        
        for j in range(config.NUM_KEYPOINTS):
            heatmap_gt = y_val[i, :, :, j]
            heatmap_resized = cv2.resize(heatmap_gt, config.IMAGE_SIZE)
            
            axes[0, j + 1].imshow(X_val[i])
            axes[0, j + 1].imshow(heatmap_resized, alpha=0.6, cmap='jet')
            axes[0, j + 1].set_title(f"{config.BODYPARTS[j]} (GT)")
            axes[0, j + 1].axis('off')
        
        # Ligne 2: Prédictions
        axes[1, 0].imshow(X_val[i])
        axes[1, 0].set_title("Image (Pred)")
        axes[1, 0].axis('off')
        
        for j in range(config.NUM_KEYPOINTS):
            heatmap_pred = predictions[i, :, :, j]
            heatmap_resized = cv2.resize(heatmap_pred, config.IMAGE_SIZE)
            
            axes[1, j + 1].imshow(X_val[i])
            axes[1, j + 1].imshow(heatmap_resized, alpha=0.6, cmap='jet')
            axes[1, j + 1].set_title(f"{config.BODYPARTS[j]} (Pred)")
            axes[1, j + 1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"comparison_{i}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Comparaison {i} sauvegardée: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("✅ Module visualize.py chargé")
    print("📝 Utilisez les fonctions pour visualiser les prédictions:")
    print("   - predict_and_visualize(model, image_path)")
    print("   - compare_predictions(model, X_val, y_val)")
