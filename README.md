# Fine-tuning Pose Estimation

Modèle de pose estimation fine-tuné avec support multi-backbones pour détecter 3 keypoints : hanche, genou, cheville.

**Backbones supportés** : MobileNetV2 (défaut), MobileNetV3, EfficientNetLite0-4, EfficientNetB0-3, EfficientNetV2B0-3

## Installation

### Google Colab

```bash
!git clone -b tests-upgrade-fine-tunning https://github.com/Spasticity-Assessment-Application/pose-estimation-finetune.git
%cd pose-estimation-finetune
!pip install -q tensorflow==2.15.0 opencv-python pandas tqdm scikit-learn
```

### Avec Conda (recommandé)

```bash
./install_conda.sh
conda activate pose-estimation
```

### Avec pip

```bash
pip install -r requirements.txt
```

## Entraînement

### Pipeline complet (entraînement + export)

```bash
# MobileNetV3Small (recommandé pour DeepLabCut-style)
python main.py --save-data --backbone MobileNetV3Small

# EfficientNetLite0 (meilleure précision)
python main.py --save-data --backbone EfficientNetLite0

# EfficientNetV2B0 (haute précision)
python main.py --save-data --backbone EfficientNetV2B0
```

### Options d'entraînement

- `--backbone` : Choix du backbone (MobileNetV2, EfficientNetLite0-4, etc.)
- `--skip-data-prep` : Utiliser les données prétraitées
- `--skip-training` : Charger un modèle existant
- `--skip-export` : Ne pas exporter en TFLite
- `--save-data` : Sauvegarder les données prétraitées

## Annotation de vidéos

### Test rapide (TFLite - recommandé pour production)

```bash
python test_video.py --video "votre_video.mp4"
# Sortie: votre_video_dynamic_annotated.mp4
```

### Test haute précision (TFLite float32)

```bash
python test_video.py --video "votre_video.mp4" --model "output/models/pose_model_float32.tflite"
# Sortie: votre_video_float32_annotated.mp4
```

### Test avec Keras (pour validation)

```bash
python test_video_keras.py --video "votre_video.mp4"
# Sortie: votre_video_keras_annotated.mp4
```

### Comparaison Keras vs TFLite

```bash
python quick_compare.py
# Génère: *_keras_annotated.mp4 et *_dynamic_annotated.mp4
```

### Export analyse en CSV

```bash
python export_video_analysis.py --video "votre_video.mp4" --model-dir "output/Dossier_Modele"
# Sortie: votre_video_analysis.csv et pivot
```

## Évaluation

### Sur données de test annotées

```bash
python evaluate_test_labeled_data.py --model-dir output/Dossier_Modele --test-data-dir test-labeled-data
```

### Analyse détaillée des erreurs par articulation

```bash
python analyze_keypoint_errors.py --model-dir output/Dossier_Modele --test-data-dir test-labeled-data --plot
```

### Prédiction sur une image

```bash
python predict.py --image "votre_image.jpg" --model "output/models/pose_model_best.h5"
```

## Données d'entraînement

Organisez vos données comme suit :

```
labeled-data/
├── 101D/
│   ├── CollectedData_*.csv    # Fichier CSV DeepLabCut
│   └── [images .png]
└── ...
```

Format CSV DeepLabCut :

- Colonne 2 : nom de l'image
- Colonnes 3-4 : hanche (x,y)
- Colonnes 5-6 : genou (x,y)
- Colonnes 7-8 : cheville (x,y)

## Résultats

Les fichiers sont sauvegardés dans `output/Dossier_Modele/` :

- `models/` : Modèles Keras et TFLite
- `logs/` : Courbes d'apprentissage et logs
- `videos/` : Vidéos annotées de test
