# Fine-tuning Pose Estimation

Modèle de pose estimation fine-tuné sur MobileNetV2 pour détecter 3 keypoints : hanche, genou, cheville.

## Installation

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
python main.py --save-data
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
```

### Sur une vidéo (Keras - pour validation)

```bash
python test_video_keras.py --video "votre_video.mp4"
```

### Comparer précision Keras vs TFLite

```bash
python quick_compare.py
```

### Prédiction sur une image

```bash
python predict.py --image "votre_image.jpg" --model "output/models/pose_model_best.h5"
```

## Options principales

### main.py

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

Après exécution, les fichiers sont sauvegardés dans `output/` :

- `models/` : Modèles Keras (.h5) et TFLite (.tflite)
- `logs/` : Logs d'entraînement
- Vidéos annotées : `{nom_video}_annotated.mp4`

## Métriques

Le modèle atteint généralement :

- **Précision TFLite** : ~3-4 pixels d'erreur moyenne
- **Taille modèle** : ~6MB (quantizé)
- **Vitesse** : ~30 FPS sur CPU mobile

## Architecture

- **Backbone** : MobileNetV2 (pré-entraîné sur ImageNet)
- **Tête** : Déconvolution 3 étages
- **Sortie** : Heatmaps 48x48x3
- **Fine-tuning** : Backbone gelé, seulement la tête entraînée
