"""
Augmentations avancées pour améliorer la généralisation
Inspiré de DeepLabCut, Albumentations et état de l'art
"""
import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import gaussian_filter, map_coordinates
import config


class AdvancedAugmentation:
    """Classe pour gérer les augmentations avancées"""
    
    def __init__(self, config_dict):
        """
        Args:
            config_dict: Dictionnaire de configuration ADVANCED_AUGMENTATION
        """
        self.config = config_dict
        
    def apply(self, image, keypoints):
        """
        Applique les augmentations avancées de manière aléatoire
        
        Args:
            image: Image numpy (H, W, 3) normalisée [0, 1]
            keypoints: Array (num_keypoints, 2) coordonnées normalisées [0, 1]
            
        Returns:
            image_aug, keypoints_aug: Image et keypoints augmentés
        """
        # Copier pour ne pas modifier les originaux
        image = image.copy()
        keypoints = keypoints.copy()
        
        # Décider si on applique les augmentations avancées
        if np.random.random() > config.ADVANCED_AUGMENTATION_PROBABILITY:
            return image, keypoints
        
        # Gaussian Blur
        if self._should_apply("gaussian_blur"):
            image = self._apply_gaussian_blur(image)
        
        # Color Jitter
        if self._should_apply("color_jitter"):
            image = self._apply_color_jitter(image)
        
        # Random Crop (affecte aussi les keypoints)
        if self._should_apply("random_crop"):
            image, keypoints = self._apply_random_crop(image, keypoints)
        
        # Random Occlusion (n'affecte pas les keypoints)
        if self._should_apply("random_occlusion"):
            image = self._apply_random_occlusion(image)
        
        # Elastic Transform (affecte aussi les keypoints)
        if self._should_apply("elastic_transform"):
            image, keypoints = self._apply_elastic_transform(image, keypoints)
        
        # Perspective Transform (affecte aussi les keypoints)
        if self._should_apply("perspective_transform"):
            image, keypoints = self._apply_perspective_transform(image, keypoints)
        
        # Gaussian Noise
        if self._should_apply("gaussian_noise"):
            image = self._apply_gaussian_noise(image)
        
        # Motion Blur
        if self._should_apply("motion_blur"):
            image = self._apply_motion_blur(image)
        
        return image, keypoints
    
    def _should_apply(self, aug_name):
        """Vérifie si une augmentation doit être appliquée"""
        aug_config = self.config.get(aug_name, {})
        return (aug_config.get("enabled", False) and 
                np.random.random() < aug_config.get("probability", 0))
    
    def _apply_gaussian_blur(self, image):
        """Applique un flou gaussien"""
        cfg = self.config["gaussian_blur"]
        sigma = np.random.uniform(*cfg["sigma_range"])
        
        # Convertir en uint8 pour OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Taille du kernel (doit être impair)
        ksize = int(sigma * 6) | 1  # Assure que c'est impair
        ksize = max(3, ksize)
        
        blurred = cv2.GaussianBlur(image_uint8, (ksize, ksize), sigma)
        return blurred.astype(np.float32) / 255.0
    
    def _apply_color_jitter(self, image):
        """Applique des variations de contrast, saturation et teinte"""
        cfg = self.config["color_jitter"]
        
        # Convertir en uint8 pour OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Convertir en HSV pour manipuler facilement
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Saturation
        sat_factor = np.random.uniform(*cfg["saturation_range"])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Hue
        hue_delta = np.random.uniform(-cfg["hue_delta"], cfg["hue_delta"]) * 179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        
        # Reconvertir en RGB
        image_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Contrast (sur RGB)
        contrast_factor = np.random.uniform(*cfg["contrast_range"])
        image_jittered = np.clip(
            (image_jittered.astype(np.float32) - 128) * contrast_factor + 128,
            0, 255
        ).astype(np.uint8)
        
        return image_jittered.astype(np.float32) / 255.0
    
    def _apply_random_crop(self, image, keypoints):
        """Applique un crop aléatoire et ajuste les keypoints"""
        cfg = self.config["random_crop"]
        h, w = image.shape[:2]
        
        # Taille du crop
        scale = np.random.uniform(*cfg["scale_range"])
        aspect_ratio = np.random.uniform(*cfg["aspect_ratio"])
        
        crop_h = int(h * scale)
        crop_w = int(crop_h * aspect_ratio)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # Position du crop
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # Crop l'image
        image_cropped = image[top:top+crop_h, left:left+crop_w]
        
        # Resize à la taille originale
        image_resized = cv2.resize(image_cropped, (w, h))
        
        # Ajuster les keypoints (coordonnées normalisées)
        keypoints_adjusted = keypoints.copy()
        
        # Convertir en pixels
        kp_pixels = keypoints * np.array([w, h])
        
        # Ajuster par rapport au crop
        kp_pixels[:, 0] = (kp_pixels[:, 0] - left) * (w / crop_w)
        kp_pixels[:, 1] = (kp_pixels[:, 1] - top) * (h / crop_h)
        
        # Reconvertir en normalisé
        keypoints_adjusted = kp_pixels / np.array([w, h])
        
        # Clipper dans [0, 1]
        keypoints_adjusted = np.clip(keypoints_adjusted, 0, 1)
        
        return image_resized, keypoints_adjusted
    
    def _apply_random_occlusion(self, image):
        """Ajoute des patches noirs aléatoires"""
        cfg = self.config["random_occlusion"]
        h, w = image.shape[:2]
        
        num_patches = np.random.randint(*cfg["num_patches"])
        image_occluded = image.copy()
        
        for _ in range(num_patches):
            # Taille du patch
            patch_size = np.random.uniform(*cfg["patch_size_range"])
            patch_h = int(h * patch_size)
            patch_w = int(w * patch_size)
            
            # Position
            top = np.random.randint(0, h - patch_h + 1)
            left = np.random.randint(0, w - patch_w + 1)
            
            # Appliquer
            image_occluded[top:top+patch_h, left:left+patch_w] = cfg["fill_value"]
        
        return image_occluded
    
    def _apply_elastic_transform(self, image, keypoints):
        """
        Applique une déformation élastique
        Inspiré de: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        cfg = self.config["elastic_transform"]
        h, w = image.shape[:2]
        
        alpha = np.random.uniform(*cfg["alpha_range"])
        sigma = cfg["sigma"]
        
        # Générer des champs de déplacement aléatoires
        random_state = np.random.RandomState(None)
        dx = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        dy = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        
        # Créer les grilles de coordonnées
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Appliquer la déformation à l'image
        image_transformed = np.zeros_like(image)
        for i in range(image.shape[2]):
            image_transformed[:, :, i] = map_coordinates(
                image[:, :, i], indices, order=1, mode='reflect'
            ).reshape((h, w))
        
        # Appliquer la déformation aux keypoints
        keypoints_transformed = keypoints.copy()
        kp_pixels = keypoints * np.array([w, h])
        
        for i in range(len(keypoints)):
            x_kp, y_kp = int(kp_pixels[i, 0]), int(kp_pixels[i, 1])
            if 0 <= y_kp < h and 0 <= x_kp < w:
                new_x = x_kp + dx[y_kp, x_kp]
                new_y = y_kp + dy[y_kp, x_kp]
                keypoints_transformed[i] = [new_x / w, new_y / h]
        
        keypoints_transformed = np.clip(keypoints_transformed, 0, 1)
        
        return image_transformed, keypoints_transformed
    
    def _apply_perspective_transform(self, image, keypoints):
        """Applique une transformation de perspective"""
        cfg = self.config["perspective_transform"]
        h, w = image.shape[:2]
        
        # Points source (coins de l'image)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Points destination (légèrement déformés)
        scale = cfg["distortion_scale"]
        pts2 = pts1 + np.random.uniform(-scale * w, scale * w, pts1.shape).astype(np.float32)
        
        # Matrice de transformation
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Appliquer à l'image
        image_transformed = cv2.warpPerspective(image, M, (w, h))
        
        # Appliquer aux keypoints
        keypoints_pixels = keypoints * np.array([w, h])
        keypoints_homogeneous = np.hstack([
            keypoints_pixels,
            np.ones((len(keypoints), 1))
        ])
        
        keypoints_transformed = (M @ keypoints_homogeneous.T).T
        keypoints_transformed = keypoints_transformed[:, :2] / keypoints_transformed[:, 2:]
        keypoints_transformed = keypoints_transformed / np.array([w, h])
        keypoints_transformed = np.clip(keypoints_transformed, 0, 1)
        
        return image_transformed, keypoints_transformed
    
    def _apply_gaussian_noise(self, image):
        """Ajoute du bruit gaussien"""
        cfg = self.config["gaussian_noise"]
        
        std = np.random.uniform(*cfg["std_range"])
        noise = np.random.normal(cfg["mean"], std, image.shape)
        
        image_noisy = image + noise
        return np.clip(image_noisy, 0, 1)
    
    def _apply_motion_blur(self, image):
        """Applique un flou de mouvement directionnel"""
        cfg = self.config["motion_blur"]
        
        # Taille du kernel
        ksize = np.random.randint(*cfg["kernel_size_range"])
        if ksize % 2 == 0:
            ksize += 1
        
        # Angle aléatoire
        angle = np.random.uniform(*cfg["angle_range"])
        
        # Créer le kernel de motion blur
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = np.ones(ksize)
        kernel = kernel / ksize
        
        # Rotation du kernel
        M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        
        # Normaliser
        kernel = kernel / kernel.sum()
        
        # Appliquer
        image_uint8 = (image * 255).astype(np.uint8)
        blurred = cv2.filter2D(image_uint8, -1, kernel)
        
        return blurred.astype(np.float32) / 255.0


def create_advanced_augmentor():
    """Factory pour créer l'augmenteur avec la config globale"""
    return AdvancedAugmentation(config.ADVANCED_AUGMENTATION)
