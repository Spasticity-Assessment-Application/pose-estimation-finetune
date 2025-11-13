"""
Construction du modèle de pose estimation avec support multi-backbones
Supporte: MobileNetV2/V3, EfficientNetLite, EfficientNetB, EfficientNetV2
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
)
import config


def get_backbone(backbone_name="MobileNetV2", input_shape=(192, 192, 3), alpha=1.0):
    """
    Charge le backbone pré-entraîné
    
    Args:
        backbone_name: Nom du backbone (MobileNetV2, MobileNetV3Small/Large, 
                       EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3)
        input_shape: Forme de l'entrée (H, W, C)
        alpha: Width multiplier (seulement pour MobileNet)
    
    Returns:
        backbone: Modèle Keras du backbone
    """
    # MobileNet backbones
    if backbone_name == "MobileNetV2":
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            alpha=alpha
        )
    elif backbone_name == "MobileNetV3Small":
        backbone = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            alpha=alpha,
            minimalistic=False
        )
    elif backbone_name == "MobileNetV3Large":
        backbone = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS,
            alpha=alpha,
            minimalistic=False
        )
    
    # EfficientNetLite backbones (légers, optimisés edge/mobile)
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
        # Lite4 utilise B3 comme base avec optimisations
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    # EfficientNetB backbones (haute précision)
    elif backbone_name == "EfficientNetB0":
        backbone = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB1":
        backbone = EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB2":
        backbone = EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetB3":
        backbone = EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    # EfficientNetV2 backbones (plus rapides, meilleure précision)
    elif backbone_name == "EfficientNetV2B0":
        backbone = EfficientNetV2B0(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B1":
        backbone = EfficientNetV2B1(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B2":
        backbone = EfficientNetV2B2(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    elif backbone_name == "EfficientNetV2B3":
        backbone = EfficientNetV2B3(
            input_shape=input_shape,
            include_top=False,
            weights=config.PRETRAINED_WEIGHTS
        )
    
    else:
        raise ValueError(f"Backbone non supporté: {backbone_name}. "
                        f"Backbones disponibles: MobileNetV2, MobileNetV3Small/Large, "
                        f"EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3")
    
    return backbone


def build_pose_model(num_keypoints=3, backbone_name="MobileNetV2", input_shape=(192, 192, 3)):
    """
    Construit le modèle complet de pose estimation
    
    Architecture:
        - Backbone (MobileNet/EfficientNet pré-entraîné sur ImageNet)
        - Upsampling progressif adaptatif
        - Tête convolutionnelle pour prédire les heatmaps
    
    Args:
        num_keypoints: Nombre de points clés à prédire
        backbone_name: Nom du backbone
        input_shape: Forme de l'entrée (H, W, C)
    
    Returns:
        model: Modèle Keras compilé
    """
    # 1. Créer l'entrée
    inputs = keras.Input(shape=input_shape, name="image_input")
    
    # 2. Charger le backbone
    backbone = get_backbone(backbone_name, input_shape, config.ALPHA)
    
    # GELER LE BACKBONE (fine-tuning uniquement de la tête)
    backbone.trainable = False
    print(f"🔒 Backbone gelé - {sum([1 for l in backbone.layers if not l.trainable])} couches non-entraînables")
    
    # 3. Extraire les features du backbone
    x = backbone(inputs)
    
    # 4. Déterminer la forme de sortie du backbone pour adapter la tête
    # La plupart des backbones réduisent par un facteur de 32
    # Ex: 192/32=6x6, 224/32=7x7, 240/32=7.5≈8x8
    reduction_ratio = config.BACKBONE_REDUCTION_RATIOS.get(backbone_name, 32)
    backbone_output_size = input_shape[0] // reduction_ratio
    
    print(f"📐 Sortie backbone: ~{backbone_output_size}x{backbone_output_size}")
    print(f"🎯 Cible heatmaps: {config.HEATMAP_SIZE[0]}x{config.HEATMAP_SIZE[1]}")
    
    # 5. Calculer le nombre d'upsampling nécessaires
    # Pour passer de backbone_output_size à HEATMAP_SIZE (48x48)
    # On fait 3 upsampling x2 : 6→12→24→48 ou 7→14→28→56 (puis on ajuste)
    
    # Première upsampling: x2
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='upsample_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    
    # Deuxième upsampling: x2
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name='upsample_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    
    # Troisième upsampling: x2
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='upsample_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
    
    # 6. Ajuster à la taille exacte des heatmaps si nécessaire
    # Utiliser Resizing pour garantir la taille exacte
    current_size = backbone_output_size * 8  # Après 3 upsampling x2
    if current_size != config.HEATMAP_SIZE[0]:
        x = layers.Resizing(
            config.HEATMAP_SIZE[0], 
            config.HEATMAP_SIZE[1], 
            interpolation='bilinear',
            name='resize_to_heatmap_size'
        )(x)
        print(f"🔧 Redimensionnement: {current_size}x{current_size} → {config.HEATMAP_SIZE[0]}x{config.HEATMAP_SIZE[1]}")
    
    # 7. Couche finale pour prédire les heatmaps
    # Conv2D avec activation sigmoid pour avoir des valeurs entre 0 et 1
    outputs = layers.Conv2D(num_keypoints, (1, 1), padding='same', activation='sigmoid', name='heatmaps')(x)
    
    # 8. Créer le modèle
    model = Model(inputs=inputs, outputs=outputs, name=f'pose_estimation_{backbone_name}')
    
    return model


def compile_model(model, learning_rate=1e-4, optimizer_name='adam'):
    """
    Compile le modèle avec la loss et l'optimiseur
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
        optimizer_name: Nom de l'optimiseur
    
    Returns:
        model: Modèle compilé
    """
    # Choisir l'optimiseur
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimiseur non supporté: {optimizer_name}")
    
    # Compiler avec MSE loss
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error entre heatmaps prédites et vraies
        metrics=['mae']  # Mean Absolute Error comme métrique additionnelle
    )
    
    return model


def create_model():
    """
    Pipeline complet de création et compilation du modèle
    
    Returns:
        model: Modèle Keras compilé et prêt à l'entraînement
    """
    print("=" * 60)
    print("🏗️  CONSTRUCTION DU MODÈLE")
    print("=" * 60)
    
    # 1. Construire le modèle
    print(f"\n📐 Construction du modèle avec backbone: {config.BACKBONE}")
    model = build_pose_model(
        num_keypoints=config.NUM_KEYPOINTS,
        backbone_name=config.BACKBONE,
        input_shape=config.INPUT_SHAPE
    )
    
    # 2. Compiler le modèle
    print(f"⚙️  Compilation avec {config.OPTIMIZER}, lr={config.LEARNING_RATE}")
    model = compile_model(
        model,
        learning_rate=config.LEARNING_RATE,
        optimizer_name=config.OPTIMIZER
    )
    
    # 3. Afficher le résumé
    print(f"\n📊 Résumé du modèle:")
    model.summary()
    
    print("\n✅ Modèle créé et compilé avec succès!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Test de la construction du modèle
    model = create_model()
    
    print("\n📊 Informations du modèle:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Nombre de paramètres: {model.count_params():,}")
