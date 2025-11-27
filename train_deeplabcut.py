"""
Entraînement style DeepLabCut
- Warmup de la tête uniquement
- Déblocage progressif du backbone
- LR très bas et régularisation forte
- Augmentations lourdes
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    CSVLogger, LearningRateScheduler, Callback
)
import numpy as np
import config
from advanced_data_generator import create_advanced_generators


def get_backbone_layer(model):
    """Trouve la couche du backbone dans le modèle"""
    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'mobilenet' in layer_name or 'efficientnet' in layer_name:
            return layer
    return None


def count_backbone_blocks(backbone_layer):
    """Compte le nombre de blocs dans le backbone"""
    if backbone_layer is None:
        return 0
    
    blocks = []
    for layer in backbone_layer.layers:
        layer_name = layer.name.lower()
        # MobileNetV2: expanded_conv_X
        # MobileNetV3: block_X
        if 'block' in layer_name or 'expanded_conv' in layer_name:
            blocks.append(layer.name)
    
    # Compter les blocs uniques
    unique_blocks = set([b.split('_')[0] + '_' + b.split('_')[1] for b in blocks if len(b.split('_')) > 1])
    return len(unique_blocks)


def unfreeze_backbone_blocks(model, num_blocks_to_unfreeze):
    """
    Débloque les N derniers blocs du backbone
    
    Args:
        model: Modèle Keras
        num_blocks_to_unfreeze: Nombre de blocs à débloquer (depuis la fin)
    """
    backbone_layer = get_backbone_layer(model)
    if backbone_layer is None:
        print("⚠️  Backbone non trouvé dans le modèle")
        return
    
    # Identifier tous les blocs
    block_layers = {}
    for layer in backbone_layer.layers:
        layer_name = layer.name.lower()
        if 'block' in layer_name or 'expanded_conv' in layer_name:
            # Extraire le numéro de bloc
            parts = layer_name.split('_')
            if len(parts) > 1:
                block_id = parts[0] + '_' + parts[1]
                if block_id not in block_layers:
                    block_layers[block_id] = []
                block_layers[block_id].append(layer)
    
    # Trier les blocs par ordre
    sorted_blocks = sorted(block_layers.keys())
    total_blocks = len(sorted_blocks)
    
    print(f"\n🔓 Déblocage de {num_blocks_to_unfreeze}/{total_blocks} derniers blocs du backbone")
    
    # Débloquer les N derniers blocs
    blocks_to_unfreeze = sorted_blocks[-num_blocks_to_unfreeze:] if num_blocks_to_unfreeze > 0 else []
    
    unfrozen_count = 0
    for block_id in blocks_to_unfreeze:
        for layer in block_layers[block_id]:
            layer.trainable = True
            unfrozen_count += 1
    
    print(f"✅ {unfrozen_count} couches débloquées dans les blocs: {', '.join(blocks_to_unfreeze)}")
    
    # Compter les paramètres entraînables
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    print(f"📊 Paramètres entraînables: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")


def create_deeplabcut_callbacks(phase_name, model_name, model_dir):
    """
    Crée les callbacks pour l'entraînement DeepLabCut
    
    Args:
        phase_name: Nom de la phase (warmup, unfreeze, finetune)
        model_name: Nom du modèle
        model_dir: Dossier de sortie
    """
    callbacks = []
    
    # Chemins
    models_dir = os.path.join(model_dir, "models")
    logs_dir = os.path.join(model_dir, "logs")
    
    # ModelCheckpoint - sauvegarde le meilleur modèle
    checkpoint_path = os.path.join(models_dir, f"{model_name}_{phase_name}_best.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # EarlyStopping - patience élevée comme DeepLabCut
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # ReduceLROnPlateau - réduction progressive du LR
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSVLogger
    csv_path = os.path.join(logs_dir, f"{model_name}_{phase_name}_log.csv")
    csv_logger = CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)
    
    return callbacks


class GradientClipCallback(Callback):
    """Callback pour clipper les gradients"""
    def __init__(self, clip_value=1.0):
        super().__init__()
        self.clip_value = clip_value
    
    def on_train_begin(self, logs=None):
        # Activer le gradient clipping dans l'optimizer
        if hasattr(self.model.optimizer, 'clipnorm'):
            self.model.optimizer.clipnorm = self.clip_value
            print(f"✅ Gradient clipping activé: {self.clip_value}")


def train_deeplabcut_progressive(model, X_train, y_train, X_val, y_val, 
                                 model_name="pose_model_dlc", model_dir=None):
    """
    Entraînement progressif style DeepLabCut
    
    Phase 1 (WARMUP): Tête seule, LR élevé, 20 epochs
    Phase 2 (UNFREEZE): Débloquer 1-2 derniers blocs, LR moyen, 30 epochs
    Phase 3 (FINETUNE): Débloquer progressivement, LR très bas, 50-100 epochs
    
    Args:
        model: Modèle DeepLabCut
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        model_name: Nom du modèle
        model_dir: Dossier de sortie
    
    Returns:
        history_combined: Historique combiné des 3 phases
        final_metrics: Métriques finales
    """
    print("=" * 80)
    print("🔬 ENTRAÎNEMENT PROGRESSIF DEEPLABCUT-STYLE")
    print("=" * 80)
    
    # Compter les blocs du backbone
    backbone_layer = get_backbone_layer(model)
    total_blocks = count_backbone_blocks(backbone_layer)
    print(f"\n📊 Backbone contient {total_blocks} blocs")
    
    history_combined = {
        'loss': [], 'val_loss': [],
        'mae': [], 'val_mae': [],
        'lr': []
    }
    
    # ========== PHASE 1: WARMUP - Tête seule ==========
    print("\n" + "=" * 80)
    print("🔥 PHASE 1: WARMUP - Entraînement de la tête uniquement")
    print("=" * 80)
    
    # Geler 100% du backbone
    if backbone_layer:
        backbone_layer.trainable = False
    
    # Compiler avec LR élevé
    warmup_lr = config.DEEPLABCUT_LR['warmup']
    optimizer = keras.optimizers.AdamW(
        learning_rate=warmup_lr,
        weight_decay=config.DEEPLABCUT_WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"🎯 Learning Rate: {warmup_lr}")
    print(f"🎯 Weight Decay: {config.DEEPLABCUT_WEIGHT_DECAY}")
    print(f"🎯 Epochs: {config.DEEPLABCUT_EPOCHS['warmup']}")
    
    # Callbacks
    callbacks_phase1 = create_deeplabcut_callbacks('warmup', model_name, model_dir)
    callbacks_phase1.append(GradientClipCallback(clip_value=1.0))
    
    # Générateurs avec augmentation lourde
    train_gen, val_gen = create_advanced_generators(
        X_train, y_train, X_val, y_val,
        batch_size=config.DEEPLABCUT_BATCH_SIZE,
        use_augmentation=True
    )
    
    # Entraînement Phase 1
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.DEEPLABCUT_EPOCHS['warmup'],
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Sauvegarder historique
    for key in history1.history:
        if key in history_combined:
            history_combined[key].extend(history1.history[key])
    
    print("\n✅ Phase 1 terminée")
    print(f"   - Best val_loss: {min(history1.history['val_loss']):.6f}")
    print(f"   - Best val_mae: {min(history1.history['val_mae']):.6f}")
    
    # ========== PHASE 2: UNFREEZE - Débloquer derniers blocs ==========
    print("\n" + "=" * 80)
    print("🔥 PHASE 2: UNFREEZE - Déblocage progressif du backbone")
    print("=" * 80)
    
    # Débloquer les 2 derniers blocs (ou 1 si peu de blocs)
    blocks_to_unfreeze = min(2, max(1, total_blocks // 4))
    unfreeze_backbone_blocks(model, blocks_to_unfreeze)
    
    # Compiler avec LR moyen
    unfreeze_lr = config.DEEPLABCUT_LR['unfreeze']
    optimizer = keras.optimizers.AdamW(
        learning_rate=unfreeze_lr,
        weight_decay=config.DEEPLABCUT_WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"🎯 Learning Rate: {unfreeze_lr}")
    print(f"🎯 Epochs: {config.DEEPLABCUT_EPOCHS['unfreeze']}")
    
    # Callbacks
    callbacks_phase2 = create_deeplabcut_callbacks('unfreeze', model_name, model_dir)
    callbacks_phase2.append(GradientClipCallback(clip_value=1.0))
    
    # Entraînement Phase 2
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.DEEPLABCUT_EPOCHS['unfreeze'],
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Sauvegarder historique
    for key in history2.history:
        if key in history_combined:
            history_combined[key].extend(history2.history[key])
    
    print("\n✅ Phase 2 terminée")
    print(f"   - Best val_loss: {min(history2.history['val_loss']):.6f}")
    print(f"   - Best val_mae: {min(history2.history['val_mae']):.6f}")
    
    # ========== PHASE 3: FINETUNE - Débloquer plus de blocs ==========
    print("\n" + "=" * 80)
    print("🔥 PHASE 3: FINETUNE - Fine-tuning complet mais lent")
    print("=" * 80)
    
    # Débloquer encore plus de blocs (total 4-6 blocs ou 30-50% du backbone)
    total_blocks_to_unfreeze = min(max(4, total_blocks // 3), total_blocks)
    unfreeze_backbone_blocks(model, total_blocks_to_unfreeze)
    
    # Compiler avec LR très bas + cosine decay
    finetune_lr = config.DEEPLABCUT_LR['finetune']
    
    # Cosine decay scheduler
    def cosine_decay_scheduler(epoch, lr):
        max_epochs = config.DEEPLABCUT_EPOCHS['finetune']
        min_lr = config.DEEPLABCUT_LR['finetune_min']
        max_lr = finetune_lr
        
        # Cosine annealing
        progress = epoch / max_epochs
        lr_new = min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        return lr_new
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=finetune_lr,
        weight_decay=config.DEEPLABCUT_WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"🎯 Learning Rate: {finetune_lr} → {config.DEEPLABCUT_LR['finetune_min']} (cosine decay)")
    print(f"🎯 Epochs: {config.DEEPLABCUT_EPOCHS['finetune']}")
    
    # Callbacks
    callbacks_phase3 = create_deeplabcut_callbacks('finetune', model_name, model_dir)
    callbacks_phase3.append(GradientClipCallback(clip_value=1.0))
    callbacks_phase3.append(LearningRateScheduler(cosine_decay_scheduler, verbose=1))
    
    # Entraînement Phase 3
    history3 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.DEEPLABCUT_EPOCHS['finetune'],
        callbacks=callbacks_phase3,
        verbose=1
    )
    
    # Sauvegarder historique
    for key in history3.history:
        if key in history_combined:
            history_combined[key].extend(history3.history[key])
    
    print("\n✅ Phase 3 terminée")
    print(f"   - Best val_loss: {min(history3.history['val_loss']):.6f}")
    print(f"   - Best val_mae: {min(history3.history['val_mae']):.6f}")
    
    # ========== RÉSUMÉ FINAL ==========
    print("\n" + "=" * 80)
    print("🎉 ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    
    final_metrics = {
        'best_val_loss': min(history_combined['val_loss']),
        'best_val_mae': min(history_combined['val_mae']),
        'total_epochs': len(history_combined['val_loss'])
    }
    
    print(f"\n📊 Résultats finaux:")
    print(f"   - Meilleure val_loss: {final_metrics['best_val_loss']:.6f}")
    print(f"   - Meilleure val_mae: {final_metrics['best_val_mae']:.6f}")
    print(f"   - Epochs totaux: {final_metrics['total_epochs']}")
    
    return history_combined, final_metrics


def save_final_model(model, model_name="pose_model", model_dir=None):
    """
    Sauvegarde le modèle final avec sa configuration
    
    Args:
        model: Modèle Keras entraîné
        model_name: Nom du modèle
        model_dir: Dossier racine du modèle
    
    Returns:
        tuple: (final_model_path, saved_model_dir)
    """
    import json
    
    # Déterminer le dossier des modèles
    models_dir = config.MODELS_DIR if model_dir is None else os.path.join(model_dir, "models")
    
    # Sauvegarder le modèle complet (architecture + poids)
    final_model_path = os.path.join(models_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"\n💾 Modèle final sauvegardé: {final_model_path}")
    
    # Sauvegarder aussi au format SavedModel (pour TFLite)
    saved_model_dir = os.path.join(models_dir, f"{model_name}_saved_model")
    model.export(saved_model_dir)
    print(f"💾 SavedModel sauvegardé: {saved_model_dir}")
    
    # Sauvegarder la configuration du modèle (CRUCIAL pour l'inférence correcte)
    model_config = {
        'backbone': config.BACKBONE,
        'input_size': [config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]],
        'heatmap_size': [config.HEATMAP_SIZE[0], config.HEATMAP_SIZE[1]],
        'num_keypoints': config.NUM_KEYPOINTS,
        'bodyparts': config.BODYPARTS
    }
    
    config_path = os.path.join(models_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"💾 Configuration sauvegardée: {config_path}")
    print(f"   - Backbone: {model_config['backbone']}")
    print(f"   - Input size: {model_config['input_size']}")
    print(f"   - Heatmap size: {model_config['heatmap_size']}")
    
    return final_model_path, saved_model_dir


def plot_training_history(history, save_path=None):
    """
    Trace les courbes d'apprentissage
    
    Args:
        history: Historique de l'entraînement
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Courbe de Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(history['mae'], label='Train MAE')
        axes[1].plot(history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Courbe de MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Courbes d'apprentissage sauvegardées: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib non installé, impossible de tracer les courbes")


if __name__ == "__main__":
    print("✅ Module train_deeplabcut.py chargé")
    print("📝 Utilisez train_deeplabcut_progressive() pour l'entraînement")
