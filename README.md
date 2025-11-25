# Fine-tuning Pose Estimation

Modèle de pose estimation fine-tuné avec support multi-backbones pour détecter 3 keypoints : hanche, genou, cheville.

**Backbones supportés** : MobileNetV2 (défaut), MobileNetV3, EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3

## Installation

### Google Colab

```bash
# Cloner la branche avec l'entraînement avancé
!git clone -b tests-upgrade-fine-tunning https://github.com/Spasticity-Assessment-Application/pose-estimation-finetune.git
%cd pose-estimation-finetune

# Installer les dépendances
!pip install -q tensorflow==2.15.0 opencv-python pandas tqdm scikit-learn

# Uploader vos données labeled-data/ puis lancer l'entraînement
!python main.py --save-data
```

### Avec Conda (recommandé)

```bash
# Cloner/installer l'environnement
./install_conda.sh

# Activer l'environnement
conda activate pose-estimation
```

### Avec pip

```bash
pip install -r requirements.txt
```

## Utilisation

### Pipeline complet (entraînement + export)

```bash
# Avec MobileNetV3Small (recommandé pour DeepLabCut-style)
python main.py --save-data --backbone MobileNetV3Small

# Avec EfficientNetLite0 (meilleure précision)
python main.py --save-data --backbone EfficientNetLite0

# Avec EfficientNetV2 (haute précision)
python main.py --save-data --backbone EfficientNetV2B0
```

### Utiliser un modèle déjà entraîné

```bash
# Charger depuis un modèle spécifique
python main.py --skip-data-prep --skip-training --model-path output/models/pose_model_YYYYMMDD_HHMMSS_saved_model
```

## Test du modèle

### Sur une vidéo (TFLite - recommandé pour production)

```bash
python test_video.py --video "votre_video.mp4"
# Sortie: votre_video_dynamic_annotated.mp4
```

### Sur une vidéo (TFLite haute précision - pour validation)

```bash
python test_video.py --video "votre_video.mp4" --model "output/models/pose_model_float32.tflite"
# Sortie: votre_video_float32_annotated.mp4
```

### Sur une vidéo (Keras - pour validation)

```bash
python test_video_keras.py --video "votre_video.mp4"
# Sortie: votre_video_keras_annotated.mp4
```

### Sur une vidéo (Keras - pour validation)

```bash
python test_video_keras.py --video "votre_video.mp4"
```

### Comparer précision Keras vs TFLite

```bash
python quick_compare.py
# Compare Keras vs TFLite Dynamic (modèle recommandé)
# Génère: *_keras_annotated.mp4 et *_dynamic_annotated.mp4
```

### Export analyse vidéo en CSV (comparaison modèles)

```bash
# Exporte les positions des keypoints pour tous les formats de modèles
python export_video_analysis.py --video "votre_video.mp4" --model-dir "output/MNv2_20251113_123456"

# Sortie:
#   - votre_video_analysis.csv (format long)
#   - votre_video_analysis_pivot.csv (format pivot pour comparaison)
# Compare: Keras, TFLite float32, dynamic, int8
```

### Évaluation sur données de test annotées

```bash
# Évaluer le modèle sur des données de test annotées
python evaluate_test_labeled_data.py \
    --model-dir output/DLC_MNv3S_20251120_195929 \
    --test-data-dir test-labeled-data
```

## Évaluation du Modèle

Ce script permet d'évaluer un modèle de pose estimation sur des données de test annotées dans le dossier `test-labeled-data`.

### Structure des Données de Test

Le dossier `test-labeled-data` doit contenir des sous-dossiers, chacun représentant une vidéo annotée :

```
test-labeled-data/
├── video1_folder/
│   ├── CollectedData_scoring.csv  # Annotations CSV
│   ├── CollectedData_scoring.h5   # Annotations HDF5 (optionnel)
│   ├── img001.png                 # Images annotées
│   ├── img002.png
│   └── ...
└── video2_folder/
    └── ...
```

### Format du CSV d'Annotations

Le fichier CSV doit suivre le format DeepLabCut :

```csv
scorer,,,bodypart1,bodypart1,bodypart2,bodypart2,...
bodyparts,,,x,y,x,y,...
labeled-data,folder_name,image_name,x1,y1,x2,y2,...
```

### Métriques Calculées

- **PCK (Percentage of Correct Keypoints)** : Pourcentage de keypoints correctement localisés (seuil 5% de la taille d'image)
- **MSE** : Erreur quadratique moyenne en unités normalisées (0-1)
- **Confiance** : Confiance moyenne des prédictions du modèle

### Exemple de Sortie

```
📁 Évaluation du dossier: video1_folder
📊 20 images annotées trouvées
Évaluation video1_folder: 100%|████████████████████████████| 20/20 [00:02<00:00,  8.49it/s]
   - PCK: 0.567 ± 0.238
   - MSE: 0.0403 ± 0.0293 (normalisé)
   - Confiance moyenne: 0.44 ± 0.18

================================================================================
📊 RÉSULTATS GLOBAUX
================================================================================
📈 Nombre total d'images évaluées: 20
🎯 PCK moyen global: 0.567
📏 MSE moyen global: 0.0403 (normalisé)
🔍 Confiance moyenne globale: 0.44
```

### Prédiction sur une image

```bash
python predict.py --image "votre_image.jpg" --model "output/models/pose_model_best.h5"
```

## Options principales

### main.py

- `--backbone` : Choix du backbone (MobileNetV2, EfficientNetLite0-4, etc. - défaut: MobileNetV2)
- `--skip-data-prep` : Utiliser les données prétraitées
- `--skip-training` : Charger un modèle existant
- `--skip-export` : Ne pas exporter en TFLite
- `--save-data` : Sauvegarder les données prétraitées
- `--model-path` : Chemin vers un modèle existant

### test_video.py / test_video_keras.py

- `--video` : Chemin vers la vidéo à analyser
- `--model` : Chemin vers le modèle (optionnel)

## Données d'entraînement

Les données doivent être organisées comme suit :

```
labeled-data/
├── 101D/
│   ├── CollectedData_*.csv    # Fichier CSV DeepLabCut (nom variable)
│   └── [images .png]
├── 101D_labeled/              # Dossier ignoré automatiquement
└── ...
```

Format CSV DeepLabCut avec colonnes :

- Colonne 2 : nom de l'image
- Colonnes 3-4 : hanche (x,y)
- Colonnes 5-6 : genou (x,y)
- Colonnes 7-8 : cheville (x,y)

## Résultats

Après exécution, les fichiers sont sauvegardés dans `output/` avec une structure organisée :

```
output/
└── Backbone_Date/                    # ex: MNv2_20251108_190128/
    ├── models/                       # Modèles entraînés
    │   ├── pose_model_best.h5        # Meilleur modèle Keras
    │   ├── pose_model_final.h5       # Modèle final Keras
    │   ├── pose_model_saved_model/   # SavedModel pour TFLite
    │   ├── pose_model_dynamic.tflite
    │   └── pose_model_float32.tflite
    ├── logs/                         # Logs et métriques
    │   ├── pose_model_YYYYMMDD-HHMMSS/  # TensorBoard
    │   ├── pose_model_history.png    # Courbes d'apprentissage
    │   └── pose_model_training_log.csv # Logs CSV
    ├── videos/                       # Vidéos annotées de test
    └── preprocessed_data.npz         # Données prétraitées
```

### Modèles exportés

- **Dynamic (.tflite)** ⭐ RECOMMANDÉ : 6MB, précision ~1px, production mobile
- **Float32 (.tflite)** 🔬 TESTS : 22MB, précision maximale, validation

## Métriques

Le modèle atteint généralement (résultats du dernier test) :

- **Précision finale** : MAE = 0.119 (pixels)
- **Taille modèle Dynamic** : ~6MB (optimisé pour mobile)
- **Taille modèle Float32** : ~22MB (haute précision)
- **Vitesse** : ~30 FPS sur CPU mobile
- **Convergence** : Loss de 0.163 → 0.015 en 5 epochs

## Architecture

- **Backbone** : Multi-backbone support (MobileNetV2 par défaut, EfficientNetLite, EfficientNetB, EfficientNetV2)
- **Tête** : Déconvolution 3 étages avec adaptation automatique à la sortie du backbone
- **Sortie** : Heatmaps 48x48x3
- **Fine-tuning** : Backbone gelé, seulement la tête entraînée
- **Augmentation** : Rotation, translation, zoom, flip horizontal

### Backbones disponibles

**Légers (mobile/edge) :**

- `MobileNetV2` (⭐ défaut) : 192x192, ~3.5M params, très rapide
- `MobileNetV3Small` : 192x192, ~2.5M params, ultra-léger
- `EfficientNetLite0-4` : 224-300px, précision progressive

**Haute précision :**

- `EfficientNetB0-3` : 224-300px, meilleure précision
- `EfficientNetV2B0-3` : 224-300px, entraînement plus rapide
