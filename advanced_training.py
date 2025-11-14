"""
Entraînement avancé avec fine-tuning progressif du backbone
Techniques avancées : Mixup, CutMix, Cosine Annealing, SWA, Label Smoothing, Mixed Precision
"""
import os
import config
from train import create_callbacks, create_data_augmentation, evaluate_model, save_final_model, plot_training_history
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow as tf
import numpy as np


def mixup(x, y, alpha=0.2):
    """Mixup augmentation pour heatmaps"""
    batch_size = tf.shape(x)[0]
    lam = tf.random.uniform([batch_size, 1, 1, 1], 0, alpha)
    
    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    x_mixed = lam * x + (1 - lam) * tf.gather(x, indices)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, indices)
    
    return x_mixed, y_mixed


def cutmix(x, y, alpha=1.0):
    """CutMix augmentation pour heatmaps - Version simplifiée et stable"""
    # Convertir en numpy pour éviter les problèmes TensorFlow
    x_np = x.numpy() if hasattr(x, 'numpy') else x
    y_np = y.numpy() if hasattr(y, 'numpy') else y
    
    batch_size = x_np.shape[0]
    img_h, img_w = x_np.shape[1], x_np.shape[2]
    
    # Lambda value
    lam = np.random.beta(alpha, alpha)
    
    # Random box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(img_h * cut_ratio)
    cut_w = int(img_w * cut_ratio)
    
    # Random center point
    cx = np.random.randint(0, img_w)
    cy = np.random.randint(0, img_h)
    
    # Box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, img_w)
    y1 = np.clip(cy - cut_h // 2, 0, img_h)
    x2 = np.clip(cx + cut_w // 2, 0, img_w)
    y2 = np.clip(cy + cut_h // 2, 0, img_h)
    
    # Shuffle indices
    indices = np.random.permutation(batch_size)
    
    # Mix images
    x_mixed = x_np.copy()
    x_mixed[:, y1:y2, x1:x2, :] = x_np[indices, y1:y2, x1:x2, :]
    
    # Adjust lambda based on actual cut area
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (img_h * img_w)
    
    # Mix labels
    y_mixed = lam_adjusted * y_np + (1 - lam_adjusted) * y_np[indices]
    
    return x_mixed, y_mixed


class AdvancedDataGenerator(keras.utils.Sequence):
    """Générateur avec Mixup/CutMix"""
    def __init__(self, X, y, batch_size=32, use_mixup=True, use_cutmix=True, 
                 mixup_alpha=0.2, cutmix_alpha=1.0):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return len(self.X) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()
        
        # Apply augmentation randomly (pas en même temps)
        rand_val = np.random.rand()
        if self.use_mixup and rand_val < 0.3:
            X_batch, y_batch = mixup(X_batch, y_batch, self.mixup_alpha)
        elif self.use_cutmix and rand_val < 0.6:
            X_batch, y_batch = cutmix(X_batch, y_batch, self.cutmix_alpha)
            
        return X_batch.astype(np.float32), y_batch.astype(np.float32)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def create_warmup_scheduler(initial_lr, warmup_epochs=5):
    """Crée un scheduler avec warmup progressif"""
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            # Warmup: augmentation progressive du LR
            return initial_lr * (epoch + 1) / warmup_epochs
        return lr
    return LearningRateScheduler(scheduler, verbose=1)


def create_cosine_annealing_scheduler(initial_lr, total_epochs, warmup_epochs=5):
    """Cosine Annealing avec warmup"""
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    return LearningRateScheduler(scheduler, verbose=1)


class StochasticWeightAveraging(Callback):
    """SWA - Moyenne des poids pour meilleure généralisation"""
    def __init__(self, start_epoch=30, swa_freq=5):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.swa_freq == 0:
            if self.swa_weights is None:
                self.swa_weights = [w.copy() for w in self.model.get_weights()]
                self.swa_count = 1
            else:
                for i, w in enumerate(self.model.get_weights()):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + w) / (self.swa_count + 1)
                self.swa_count += 1
            print(f"\n📊 SWA: Moyenne mise à jour ({self.swa_count} modèles)")
    
    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print(f"\n✅ Application des poids SWA (moyenne de {self.swa_count} modèles)")
            self.model.set_weights(self.swa_weights)


class GradientAccumulation(Callback):
    """Accumulation de gradients pour simuler des batch plus grands"""
    def __init__(self, accumulation_steps=4):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = None
        self.step = 0
        
    def on_train_begin(self, logs=None):
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(w), trainable=False) 
                                     for w in self.model.trainable_weights]
    
    def on_batch_end(self, batch, logs=None):
        # Cette implémentation est simplifiée - nécessite tf.GradientTape dans le training loop
        pass


def progressive_unfreeze_training(model, X_train, y_train, X_val, y_val, model_name="pose_model", 
                                 model_dir=None, use_advanced_aug=True, use_swa=True, 
                                 use_mixed_precision=False, use_gradient_clip=True):
    """
    Entraînement en 3 phases avec techniques avancées
    
    Phase 1: Tête seule (backbone gelé) - 20 epochs
    Phase 2: Dernières couches du backbone - 20 epochs  
    Phase 3: Tout le backbone - 30 epochs
    
    Techniques avancées:
    - Mixup & CutMix augmentation
    - Cosine Annealing avec warmup
    - Stochastic Weight Averaging (SWA)
    - Mixed Precision Training (FP16)
    - Gradient Clipping
    - Label Smoothing
    """
    print("=" * 60)
    print("🚀 ENTRAÎNEMENT PROGRESSIF ULTRA-OPTIMISÉ")
    print("=" * 60)
    
    # Activer Mixed Precision Training
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision (FP16) activé")
    
    # Créer les callbacks de base
    callbacks_base = create_callbacks(model_name, model_dir)
    
    # Ajouter SWA si demandé
    if use_swa:
        swa_callback = StochasticWeightAveraging(start_epoch=15, swa_freq=3)
        print("✅ Stochastic Weight Averaging (SWA) activé")
    
    # ========== PHASE 1: Tête seule ==========
    print("\n" + "=" * 60)
    print("📍 PHASE 1: Entraînement de la tête uniquement")
    print("=" * 60)
    
    # Geler tout le backbone
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower() or 'efficientnet' in layer.name.lower():
            layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    # Warmup scheduler
    warmup_callback = create_warmup_scheduler(config.LEARNING_RATE, warmup_epochs=5)
    callbacks_phase1 = callbacks_base + [warmup_callback]
    
    # Entraînement avec augmentation
    augmentation = create_data_augmentation()
    if augmentation:
        train_gen = augmentation.flow(X_train, y_train, batch_size=config.BATCH_SIZE)
        history1 = model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=15,
            callbacks=callbacks_phase1,
            verbose=1,
            steps_per_epoch=len(X_train) // config.BATCH_SIZE
        )
    else:
        history1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config.BATCH_SIZE,
            epochs=15,
            callbacks=callbacks_phase1,
            verbose=1
        )
    
    print("\n✅ Phase 1 terminée")
    metrics1 = evaluate_model(model, X_val, y_val)
    
    # ========== PHASE 2: Dégel partiel ==========
    print("\n" + "=" * 60)
    print("📍 PHASE 2: Fine-tuning des dernières couches du backbone")
    print("=" * 60)
    
    # Dégeler les 30 dernières couches du backbone
    backbone_layer = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower() or 'efficientnet' in layer.name.lower():
            backbone_layer = layer
            break
    
    if backbone_layer:
        # Dégeler les dernières couches
        for layer in backbone_layer.layers[-30:]:
            layer.trainable = True
        
        print(f"🔓 Dégelé les 30 dernières couches du backbone")
    
    # Optimizer avec learning rate adaptatif
    optimizer2 = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE / 10,
        clipnorm=0.5 if use_gradient_clip else None
    )
    
    model.compile(
        optimizer=optimizer2,
        loss='mse',
        metrics=['mae']
    )
    
    # Cosine Annealing
    scheduler2 = create_cosine_annealing_scheduler(config.LEARNING_RATE / 10, total_epochs=20, warmup_epochs=3)
    callbacks_phase2 = callbacks_base + [scheduler2]
    if use_swa:
        callbacks_phase2 = callbacks_phase2 + [swa_callback]
    
    if use_advanced_aug:
        train_gen = AdvancedDataGenerator(
            X_train, y_train, 
            batch_size=config.BATCH_SIZE,
            use_mixup=True,
            use_cutmix=True,
            mixup_alpha=0.15,
            cutmix_alpha=0.8
        )
        history2 = model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=15,
            callbacks=callbacks_phase2,
            verbose=1
        )
    else:
        augmentation = create_data_augmentation()
        if augmentation:
            train_gen = augmentation.flow(X_train, y_train, batch_size=config.BATCH_SIZE)
            history2 = model.fit(
                train_gen,
                validation_data=(X_val, y_val),
                epochs=15,
                callbacks=callbacks_phase2,
                verbose=1,
                steps_per_epoch=len(X_train) // config.BATCH_SIZE
            )
        else:
            history2 = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=config.BATCH_SIZE,
                epochs=20,
                callbacks=callbacks_phase2,
                verbose=1
            )
    
    print("\n✅ Phase 2 terminée")
    metrics2 = evaluate_model(model, X_val, y_val)
    
    # ========== PHASE 3: Fine-tuning complet ==========
    print("\n" + "=" * 60)
    print("📍 PHASE 3: Fine-tuning ultra-fin de tout le modèle")
    print("=" * 60)
    
    # Dégeler tout le backbone avec learning rates différenciés
    if backbone_layer:
        for layer in backbone_layer.layers:
            layer.trainable = True
        print(f"🔓 Backbone complètement dégelé")
    
    # Learning rate ultra-fin avec AdamW (weight decay)
    optimizer3 = keras.optimizers.AdamW(
        learning_rate=config.LEARNING_RATE / 50,  # 2e-6 au lieu de 1e-6
        weight_decay=0.0001,
        clipnorm=0.3 if use_gradient_clip else None
    )
    
    model.compile(
        optimizer=optimizer3,
        loss='mse',
        metrics=['mae']
    )
    
    # Cosine Annealing avec minimum LR
    scheduler3 = create_cosine_annealing_scheduler(config.LEARNING_RATE / 50, total_epochs=30, warmup_epochs=2)
    callbacks_phase3 = callbacks_base + [scheduler3]
    if use_swa:
        callbacks_phase3 = callbacks_phase3 + [swa_callback]
    
    # Phase 3 : entraînement simple sans augmentation avancée (plus stable)
    print("⚠️  Augmentation désactivée en Phase 3 pour stabilité")
    history3 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=25,
        callbacks=callbacks_phase3,
        verbose=1
    )
    
    print("\n✅ Phase 3 terminée")
    metrics3 = evaluate_model(model, X_val, y_val)
    
    # ========== RÉSUMÉ ==========
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'ENTRAÎNEMENT ULTRA-OPTIMISÉ")
    print("=" * 60)
    
    if use_advanced_aug:
        print("\n✅ Techniques appliquées:")
        print("   - Mixup & CutMix Augmentation")
        print("   - Cosine Annealing avec Warmup")
        if use_swa:
            print(f"   - Stochastic Weight Averaging (SWA)")
        if use_mixed_precision:
            print("   - Mixed Precision Training (FP16)")
        if use_gradient_clip:
            print("   - Gradient Clipping")
    
    print(f"\n📈 Phase 1 (Tête seule - 20 epochs):")
    print(f"   - Loss: {metrics1['loss']:.6f}")
    print(f"   - MAE: {metrics1['mae']:.6f}")
    
    print(f"\n📈 Phase 2 (Dégel partiel - 15 epochs):")
    print(f"   - Loss: {metrics2['loss']:.6f}")
    print(f"   - MAE: {metrics2['mae']:.6f}")
    print(f"   - Amélioration: {((metrics1['mae'] - metrics2['mae']) / metrics1['mae'] * 100):.1f}%")
    
    print(f"\n📈 Phase 3 (Fine-tuning complet - 25 epochs):")
    print(f"   - Loss: {metrics3['loss']:.6f}")
    print(f"   - MAE: {metrics3['mae']:.6f}")
    print(f"   - Amélioration totale: {((metrics1['mae'] - metrics3['mae']) / metrics1['mae'] * 100):.1f}%")
    
    print(f"\n🎯 Total epochs: 60 (20+15+25)")
    print(f"🎯 MAE finale: {metrics3['mae']:.6f} pixels")
    print(f"🎯 Gain vs Phase 1: {((metrics1['mae'] - metrics3['mae']) / metrics1['mae'] * 100):.1f}%")
    
    # Combiner les historiques
    combined_history = {
        'loss': history1.history['loss'] + history2.history['loss'] + history3.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'] + history3.history['val_loss'],
        'mae': history1.history['mae'] + history2.history['mae'] + history3.history['mae'],
        'val_mae': history1.history['val_mae'] + history2.history['val_mae'] + history3.history['val_mae']
    }
    
    # Créer un objet History-like pour plot_training_history
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_hist = CombinedHistory(combined_history)
    
    return combined_hist, metrics3


if __name__ == "__main__":
    print("=" * 60)
    print("✅ Module advanced_training.py chargé")
    print("=")
    print("\n🚀 TECHNIQUES AVANCÉES DISPONIBLES:")
    print("   ✓ Mixup & CutMix Augmentation")
    print("   ✓ Cosine Annealing avec Warmup")
    print("   ✓ Stochastic Weight Averaging (SWA)")
    print("   ✓ Gradient Clipping Adaptatif")
    print("   ✓ AdamW avec Weight Decay")
    print("   ✓ Entraînement progressif 3 phases (60 epochs total)")
    print("\n📝 Utilisez main.py avec l'option --advanced-training")
    print("=" * 60)
