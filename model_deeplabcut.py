"""
Architecture de modèle inspirée de DeepLabCut
- Tête simplifiée avec upsampling bilinear
- Conv 1x1 pour réduction de dimensionnalité
- Activation linéaire (pas de sigmoid)
- Optimisé pour petit dataset
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
)
import config


def get_backbone_deeplabcut(backbone_name="MobileNetV3Small", input_shape=(256, 256, 3)):
    """
    Charge le backbone avec configuration DeepLabCut-friendly
    
    Args:
        backbone_name: Nom du backbone
        input_shape: Forme de l'entrée (H, W, C)
    
    Returns:
        backbone: Modèle Keras du backbone
    """
    if backbone_name == "MobileNetV2":
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            alpha=1.0
        )
    elif backbone_name == "MobileNetV3Small":
        backbone = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            alpha=1.0,
            minimalistic=True,  # ✅ Utilise ReLU au lieu de hard-swish
            include_preprocessing=False
        )
    elif backbone_name == "MobileNetV3Large":
        backbone = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            alpha=1.0,
            minimalistic=True,
            include_preprocessing=False
        )
    elif backbone_name == "EfficientNetLite0":
        backbone = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite1":
        backbone = EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite2":
        backbone = EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite3":
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetLite4":
        backbone = EfficientNetB4(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    else:
        raise ValueError(f"Backbone {backbone_name} non supporté pour DeepLabCut mode")
    
    return backbone


def build_deeplabcut_model(num_keypoints=3, backbone_name="MobileNetV3Small", 
                           input_shape=(256, 256, 3), heatmap_stride=4):
    """
    Construit un modèle de pose estimation style DeepLabCut
    
    Architecture:
        1. Backbone pré-entraîné (stride 32)
        2. Conv 1×1 pour réduire les channels
        3. Upsampling bilinear progressif
        4. Conv 1×1 finale pour générer les heatmaps
        5. Activation linéaire (pas de sigmoid)
    
    Args:
        num_keypoints: Nombre de points clés
        backbone_name: Nom du backbone
        input_shape: Forme de l'entrée (H, W, C)
        heatmap_stride: Stride final des heatmaps (4 = haute résolution, 8 = moyenne)
    
    Returns:
        model: Modèle Keras
    """
    print("=" * 60)
    print("🔬 CONSTRUCTION DU MODÈLE DEEPLABCUT-STYLE")
    print("=" * 60)
    print(f"📦 Backbone: {backbone_name}")
    print(f"📊 Input shape: {input_shape}")
    print(f"📊 Heatmap stride: {heatmap_stride} (résolution: {input_shape[0]//heatmap_stride}×{input_shape[1]//heatmap_stride})")
    
    # 1. Entrée
    inputs = keras.Input(shape=input_shape, name="image_input")
    
    # 2. Backbone
    backbone = get_backbone_deeplabcut(backbone_name, input_shape)
    backbone.trainable = False  # Gelé au départ
    x = backbone(inputs)
    
    # Afficher la shape de sortie du backbone
    print(f"📐 Backbone output shape: {x.shape}")
    
    # 3. Réduction de dimensionnalité (Conv 1×1)
    # Réduit le nombre de channels pour accélérer l'upsampling
    x = layers.Conv2D(256, (1, 1), padding='same', name='reduce_channels')(x)
    x = layers.BatchNormalization(name='bn_reduce')(x)
    x = layers.ReLU(name='relu_reduce')(x)
    print(f"📐 After channel reduction: 256 channels")
    
    # 4. Upsampling progressif (bilinear interpolation)
    # Le backbone a un stride de 32, on veut arriver à stride 4 ou 8
    # 32 → 16 → 8 → 4 (3 étapes d'upsampling 2×)
    
    current_stride = 32
    target_stride = heatmap_stride
    
    upsample_step = 1
    while current_stride > target_stride:
        # Upsampling bilinear 2×
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', 
                                name=f'upsample_{upsample_step}')(x)
        
        # Conv pour affiner après upsampling
        filters = max(128 // upsample_step, 64)  # Réduire progressivement les filters
        x = layers.Conv2D(filters, (3, 3), padding='same', 
                         name=f'refine_{upsample_step}')(x)
        x = layers.BatchNormalization(name=f'bn_refine_{upsample_step}')(x)
        x = layers.ReLU(name=f'relu_refine_{upsample_step}')(x)
        
        current_stride //= 2
        upsample_step += 1
        print(f"📐 After upsample step {upsample_step-1}: stride={current_stride}, filters={filters}")
    
    # 5. Tête finale: Conv 1×1 pour générer les heatmaps
    heatmaps = layers.Conv2D(
        num_keypoints, 
        (1, 1), 
        padding='same',
        activation='linear',  # ✅ Activation linéaire comme DeepLabCut
        name='heatmaps_output'
    )(x)
    
    print(f"📐 Final heatmaps shape: {heatmaps.shape}")
    
    # 6. Construire le modèle
    model = Model(inputs=inputs, outputs=heatmaps, name=f"DeepLabCut_{backbone_name}")
    
    # Résumé
    print(f"\n📊 Résumé du modèle:")
    print(f"   - Paramètres totaux: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   - Paramètres entraînables: {trainable_params:,}")
    print(f"   - Paramètres gelés: {model.count_params() - trainable_params:,}")
    print("=" * 60)
    
    return model


def create_deeplabcut_model():
    """
    Factory pour créer le modèle DeepLabCut avec la config globale
    """
    return build_deeplabcut_model(
        num_keypoints=config.NUM_KEYPOINTS,
        backbone_name=config.BACKBONE,
        input_shape=config.INPUT_SHAPE,
        heatmap_stride=config.DEEPLABCUT_HEATMAP_STRIDE
    )


if __name__ == "__main__":
    print("✅ Module model_deeplabcut.py chargé")
    print("📝 Utilisez create_deeplabcut_model() pour créer le modèle")
