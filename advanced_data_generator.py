"""
Générateur de données personnalisé avec augmentation avancée
Applique les augmentations avancées aux images et transforme correctement les heatmaps
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config


class AdvancedDataGenerator(keras.utils.Sequence):
    """
    Générateur de données personnalisé qui applique les augmentations avancées
    tout en maintenant la cohérence entre images et heatmaps
    """
    
    def __init__(self, images, heatmaps, batch_size=32, shuffle=True, 
                 use_augmentation=True, augmentation_probability=0.7):
        """
        Args:
            images: Array numpy (N, H, W, 3) - images normalisées
            heatmaps: Array numpy (N, H_hm, W_hm, num_keypoints)
            batch_size: Taille des batches
            shuffle: Mélanger les données à chaque epoch
            use_augmentation: Activer les augmentations avancées
            augmentation_probability: Probabilité d'appliquer les augmentations
        """
        self.images = images
        self.heatmaps = heatmaps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        self.augmentation_probability = augmentation_probability
        
        # Créer l'augmenteur si activé
        self.augmentor = None
        if self.use_augmentation:
            from advanced_augmentation import create_advanced_augmentor
            self.augmentor = create_advanced_augmentor()
            
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()
    
    def __len__(self):
        """Nombre de batches par epoch"""
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        """Génère un batch de données"""
        # Obtenir les indices pour ce batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.images))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Générer les données pour ce batch
        X, y = self._generate_batch(batch_indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """Mélange les données à la fin de chaque epoch si nécessaire"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _generate_batch(self, batch_indexes):
        """
        Génère un batch de données avec augmentation
        
        Args:
            batch_indexes: Indices des images pour ce batch
            
        Returns:
            X: Batch d'images augmentées (batch_size, H, W, 3)
            y: Batch de heatmaps augmentées (batch_size, H_hm, W_hm, num_keypoints)
        """
        batch_images = []
        batch_heatmaps = []
        
        for idx in batch_indexes:
            image = self.images[idx].copy()
            heatmap = self.heatmaps[idx].copy()
            
            # Appliquer les augmentations si activées
            if self.use_augmentation and self.augmentor is not None:
                # Décider si on applique l'augmentation pour cette image
                if np.random.random() < self.augmentation_probability:
                    # Extraire les keypoints des heatmaps (positions des maxima)
                    keypoints = self._extract_keypoints_from_heatmap(heatmap)
                    
                    # Appliquer les augmentations (image + keypoints)
                    image_aug, keypoints_aug = self.augmentor.apply(image, keypoints)
                    
                    # Recréer les heatmaps à partir des keypoints augmentés
                    heatmap_aug = self._create_heatmaps_from_keypoints(
                        keypoints_aug, 
                        heatmap.shape[0], 
                        heatmap.shape[1]
                    )
                    
                    image = image_aug
                    heatmap = heatmap_aug
            
            batch_images.append(image)
            batch_heatmaps.append(heatmap)
        
        X = np.array(batch_images, dtype=np.float32)
        y = np.array(batch_heatmaps, dtype=np.float32)
        
        return X, y
    
    def _extract_keypoints_from_heatmap(self, heatmap):
        """
        Extrait les positions des keypoints à partir d'une heatmap
        
        Args:
            heatmap: Heatmap (H, W, num_keypoints)
            
        Returns:
            keypoints: Liste de (x, y) normalisés [0, 1]
        """
        keypoints = []
        h, w, num_kpts = heatmap.shape
        
        for k in range(num_kpts):
            hm = heatmap[:, :, k]
            
            # Trouver le maximum (position du keypoint)
            max_idx = np.argmax(hm)
            y, x = np.unravel_index(max_idx, hm.shape)
            
            # Normaliser les coordonnées [0, 1]
            x_norm = x / w
            y_norm = y / h
            
            keypoints.append([x_norm, y_norm])
        
        return np.array(keypoints, dtype=np.float32)
    
    def _create_heatmaps_from_keypoints(self, keypoints, height, width):
        """
        Crée des heatmaps à partir de positions de keypoints
        
        Args:
            keypoints: Array (num_keypoints, 2) avec coordonnées normalisées [0, 1]
            height: Hauteur de la heatmap
            width: Largeur de la heatmap
            
        Returns:
            heatmap: Heatmap (H, W, num_keypoints)
        """
        num_keypoints = len(keypoints)
        heatmap = np.zeros((height, width, num_keypoints), dtype=np.float32)
        
        sigma = config.HEATMAP_SIGMA
        
        for k, (x_norm, y_norm) in enumerate(keypoints):
            # Convertir en pixels
            center_x = int(x_norm * width)
            center_y = int(y_norm * height)
            
            # Créer une grille de coordonnées
            x = np.arange(0, width, 1, dtype=np.float32)
            y = np.arange(0, height, 1, dtype=np.float32)
            y = y[:, np.newaxis]
            
            # Heatmap gaussienne
            hm = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
            heatmap[:, :, k] = hm
        
        return heatmap


def create_advanced_generators(X_train, y_train, X_val, y_val, 
                               batch_size=None, 
                               use_augmentation=True):
    """
    Crée les générateurs de données avancés pour l'entraînement et la validation
    
    Args:
        X_train: Images d'entraînement (N, H, W, 3)
        y_train: Heatmaps d'entraînement (N, H_hm, W_hm, num_keypoints)
        X_val: Images de validation
        y_val: Heatmaps de validation
        batch_size: Taille des batches (utilise config.BATCH_SIZE si None)
        use_augmentation: Activer les augmentations avancées
        
    Returns:
        train_generator: Générateur pour l'entraînement
        val_generator: Générateur pour la validation
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Générateur d'entraînement (avec augmentation)
    train_generator = AdvancedDataGenerator(
        images=X_train,
        heatmaps=y_train,
        batch_size=batch_size,
        shuffle=True,
        use_augmentation=use_augmentation,
        augmentation_probability=config.ADVANCED_AUGMENTATION_PROBABILITY
    )
    
    # Générateur de validation (sans augmentation)
    val_generator = AdvancedDataGenerator(
        images=X_val,
        heatmaps=y_val,
        batch_size=batch_size,
        shuffle=False,
        use_augmentation=False
    )
    
    print(f"\n🔄 Générateurs de données créés:")
    print(f"   - Train: {len(train_generator)} batches, augmentation: {use_augmentation}")
    print(f"   - Validation: {len(val_generator)} batches, augmentation: False")
    if use_augmentation:
        print(f"   - Probabilité d'augmentation: {config.ADVANCED_AUGMENTATION_PROBABILITY:.0%}")
    
    return train_generator, val_generator


if __name__ == "__main__":
    """Test du générateur"""
    import sys
    from data_preprocessing import prepare_data
    
    print("🧪 Test du générateur de données avancé\n")
    
    # Préparer les données
    X_train, X_val, y_train, y_val = prepare_data()
    
    # Créer les générateurs
    train_gen, val_gen = create_advanced_generators(
        X_train, y_train, X_val, y_val,
        batch_size=4,
        use_augmentation=True
    )
    
    # Tester un batch
    print("\n🔍 Test d'un batch d'entraînement:")
    X_batch, y_batch = train_gen[0]
    print(f"   - X_batch shape: {X_batch.shape}")
    print(f"   - y_batch shape: {y_batch.shape}")
    print(f"   - X_batch range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print(f"   - y_batch range: [{y_batch.min():.3f}, {y_batch.max():.3f}]")
    
    print("\n✅ Test réussi!")
