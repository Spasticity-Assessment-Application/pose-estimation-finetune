"""
Construction du modèle de pose estimation basé sur MobileNetV2
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
import config


def get_backbone(backbone_name="MobileNetV2", input_shape=(192, 192, 3), alpha=1.0):
    """
    Charge le backbone pré-entraîné
    
    Args:
        backbone_name: Nom du backbone ("MobileNetV2", "MobileNetV3Small", "MobileNetV3Large")
        input_shape: Forme de l'entrée (H, W, C)
        alpha: Width multiplier
    
    Returns:
        backbone: Modèle Keras du backbone
    """
    if backbone_name == "MobileNetV2":
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,  # On retire la tête de classification
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
    else:
        raise ValueError(f"Backbone non supporté: {backbone_name}")
    
    return backbone


def build_pose_model(num_keypoints=3, backbone_name="MobileNetV2", input_shape=(192, 192, 3)):
    """
    Construit le modèle complet de pose estimation
    
    Architecture:
        - Backbone MobileNet (pré-entraîné sur ImageNet)
        - Upsampling progressif
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
    
    # À ce stade, x a une forme approximative de (batch, 6, 6, 1280) pour MobileNetV2
    # On veut arriver à (batch, 48, 48, num_keypoints)
    
    # 4. Tête de déconvolution pour upsampler vers la taille des heatmaps
    
    # Première upsampling: 6x6 -> 12x12
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='upsample_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    
    # Deuxième upsampling: 12x12 -> 24x24
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name='upsample_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    
    # Troisième upsampling: 24x24 -> 48x48
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='upsample_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
    
    # 5. Couche finale pour prédire les heatmaps
    # On utilise une Conv2D avec activation sigmoid pour avoir des valeurs entre 0 et 1
    outputs = layers.Conv2D(num_keypoints, (1, 1), padding='same', activation='sigmoid', name='heatmaps')(x)
    
    # 6. Créer le modèle
    model = Model(inputs=inputs, outputs=outputs, name='pose_estimation_model')
    
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
