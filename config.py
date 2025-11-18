"""
Configuration pour le fine-tuning de pose estimation
"""
import os
from datetime import datetime

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_DATA_DIR = os.path.join(BASE_DIR, "labeled-data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Fonction pour créer le nom du dossier modèle
def get_model_folder_name(backbone=None, timestamp=None):
    """Génère le nom du dossier pour un modèle spécifique"""
    if backbone is None:
        backbone = BACKBONE
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Nettoyer le nom du backbone (enlever caractères spéciaux)
    backbone_mapping = {
        "MobileNetV2": "MNv2",
        "MobileNetV3Small": "MNv3S",
        "MobileNetV3Large": "MNv3L",
        "EfficientNetLite0": "ENL0",
        "EfficientNetLite1": "ENL1",
        "EfficientNetLite2": "ENL2",
        "EfficientNetLite3": "ENL3",
        "EfficientNetLite4": "ENL4",
        "EfficientNetB0": "ENB0",
        "EfficientNetB1": "ENB1",
        "EfficientNetB2": "ENB2",
        "EfficientNetB3": "ENB3",
        "EfficientNetV2B0": "ENV2B0",
        "EfficientNetV2B1": "ENV2B1",
        "EfficientNetV2B2": "ENV2B2",
        "EfficientNetV2B3": "ENV2B3",
    }
    clean_backbone = backbone_mapping.get(backbone, backbone)

    return f"{clean_backbone}_{timestamp}"

# Dossier modèle actuel (sera défini dynamiquement)
MODEL_DIR = None
MODELS_DIR = None
LOGS_DIR = None
VIDEOS_DIR = None

def setup_model_directories(model_folder_name=None):
    """Configure les dossiers pour un modèle spécifique"""
    global MODEL_DIR, MODELS_DIR, LOGS_DIR, VIDEOS_DIR

    if model_folder_name is None:
        model_folder_name = get_model_folder_name()

    MODEL_DIR = os.path.join(OUTPUT_DIR, model_folder_name)
    MODELS_DIR = os.path.join(MODEL_DIR, "models")
    LOGS_DIR = os.path.join(MODEL_DIR, "logs")
    VIDEOS_DIR = os.path.join(MODEL_DIR, "videos")

    # Créer tous les dossiers
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    return MODEL_DIR, MODELS_DIR, LOGS_DIR, VIDEOS_DIR

# Initialisation par défaut (sera remplacée lors de l'entraînement)
# setup_model_directories()  # Commenté pour éviter l'appel automatique

# Points clés
BODYPARTS = ["Hanche", "Genoux", "Cheville"]
NUM_KEYPOINTS = len(BODYPARTS)
KEYPOINT_INDICES = {
    "Hanche": (3, 4),
    "Genoux": (5, 6),
    "Cheville": (7, 8)
}

# Images
IMAGE_SIZE = (192, 192)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 2.5
NORMALIZE = True

# Entraînement
TRAIN_SPLIT = 0.8
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 5e-5
OPTIMIZER = "adam"
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
RANDOM_SEED = 42

# Learning rates spécialisés par phase pour petit dataset
PHASE_LEARNING_RATES = {
    "phase1": 1e-4,
    "phase2": 5e-5,
    "phase3": 1e-5
}

PHASE_EPOCHS = {
    "phase1": 50,
    "phase2": 40,
    "phase3": 40
}

# Modèle
BACKBONE = "MobileNetV2"  # Par défaut: MobileNetV2 (rapide, léger, performant)
PRETRAINED_WEIGHTS = "imagenet"
ALPHA = 1.0

# Tailles d'images recommandées par backbone (pour performances optimales)
BACKBONE_INPUT_SIZES = {
    "MobileNetV2": (192, 192),
    "MobileNetV3Small": (192, 192),
    "MobileNetV3Large": (224, 224),
    "EfficientNetLite0": (224, 224),
    "EfficientNetLite1": (240, 240),
    "EfficientNetLite2": (260, 260),
    "EfficientNetLite3": (280, 280),
    "EfficientNetLite4": (300, 300),
    "EfficientNetB0": (224, 224),
    "EfficientNetB1": (240, 240),
    "EfficientNetB2": (260, 260),
    "EfficientNetB3": (300, 300),
    "EfficientNetV2B0": (224, 224),
    "EfficientNetV2B1": (240, 240),
    "EfficientNetV2B2": (260, 260),
    "EfficientNetV2B3": (300, 300),
}

# Ratios de réduction du backbone (pour adapter la tête de déconvolution)
BACKBONE_REDUCTION_RATIOS = {
    "MobileNetV2": 32,          # 192/32 = 6x6
    "MobileNetV3Small": 32,
    "MobileNetV3Large": 32,
    "EfficientNetLite0": 32,    # 224/32 = 7x7
    "EfficientNetLite1": 32,
    "EfficientNetLite2": 32,
    "EfficientNetLite3": 32,
    "EfficientNetLite4": 32,
    "EfficientNetB0": 32,
    "EfficientNetB1": 32,
    "EfficientNetB2": 32,
    "EfficientNetB3": 32,
    "EfficientNetV2B0": 32,
    "EfficientNetV2B1": 32,
    "EfficientNetV2B2": 32,
    "EfficientNetV2B3": 32,
}

# Export TFLite
TFLITE_QUANTIZATION = True
TFLITE_MODEL_NAME = "pose_model_quantized.tflite"

# Augmentation - Configuration avancée (inspirée DeepLabCut + état de l'art)
USE_AUGMENTATION = True

# Augmentations géométriques (DeepLabCut baseline)
AUGMENTATION_CONFIG = {
    "rotation_range": 25,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": [0.8, 1.2],
    "horizontal_flip": True,
    "brightness_range": [0.6, 1.4],
    "shear_range": 0,
    "fill_mode": "reflect"
}

# Augmentations avancées pour améliorer la généralisation
ADVANCED_AUGMENTATION = {
    # Blur pour simuler flou caméra/mouvement
    "gaussian_blur": {
        "enabled": True,
        "probability": 0.3,
        "sigma_range": [0.5, 2.0]  # DeepLabCut: σ=1-3
    },
    
    # Color jitter pour robustesse aux conditions d'éclairage
    "color_jitter": {
        "enabled": True,
        "probability": 0.4,
        "contrast_range": [0.7, 1.3],      # ±30% contrast
        "saturation_range": [0.7, 1.3],    # ±30% saturation
        "hue_delta": 0.1                    # Légère variation teinte
    },
    
    # Random crop pour forcer robustesse spatiale
    "random_crop": {
        "enabled": True,
        "probability": 0.3,
        "scale_range": [0.8, 1.0],  # 80-100% de l'image originale
        "aspect_ratio": [0.9, 1.1]  # Maintenir aspect ratio proche
    },
    
    # Occlusion aléatoire (objets devant, mains, etc.)
    "random_occlusion": {
        "enabled": True,
        "probability": 0.2,
        "num_patches": [1, 3],       # 1-3 patches par image
        "patch_size_range": [0.05, 0.15],  # 5-15% de l'image
        "fill_value": 0              # Noir (peut être random)
    },
    
    # Elastic deformation pour poses plus naturelles
    "elastic_transform": {
        "enabled": True,
        "probability": 0.2,
        "alpha_range": [5, 15],      # Intensité déformation
        "sigma": 3                    # Lissage
    },
    
    # Perspective transform pour angles caméra
    "perspective_transform": {
        "enabled": True,
        "probability": 0.15,
        "distortion_scale": 0.2       # Légère distortion
    },
    
    # Gaussian noise pour robustesse au bruit capteur
    "gaussian_noise": {
        "enabled": True,
        "probability": 0.2,
        "mean": 0,
        "std_range": [0.01, 0.05]     # 1-5% noise
    },
    
    # Motion blur (flou directionnel)
    "motion_blur": {
        "enabled": True,
        "probability": 0.15,
        "kernel_size_range": [3, 7],
        "angle_range": [0, 180]       # Toutes directions
    }
}

# Probabilité d'appliquer les augmentations avancées
# (les augmentations de base sont toujours appliquées)
ADVANCED_AUGMENTATION_PROBABILITY = 0.7  # 70% des images

VERBOSE = 1
