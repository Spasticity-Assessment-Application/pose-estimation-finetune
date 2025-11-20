#!/bin/bash

###############################################################################
# Script pour annoter les vidéos de test avec les modèles float32
# ENL0 (EfficientNetLite0) et ENL4 (EfficientNetLite4)
###############################################################################

echo "=========================================="
echo "🎬 ANNOTATION DES VIDÉOS DE TEST"
echo "=========================================="

# Configuration
BASE_DIR="/Users/jules/Document local/Jules/Cours/ETS/A25/PFE/test fine-tuning"
TEST_VIDEOS_DIR="$BASE_DIR/test-videos"
ENL0_MODEL="$BASE_DIR/output/ENL0_20251119_135726/models/pose_model_float32.tflite"
ENL4_MODEL="$BASE_DIR/output/ENL4_20251119_150714/models/pose_model_float32.tflite"

# Liste des vidéos
VIDEOS=(
    "20250925_161004.mp4"
    "20250927_230610.mp4"
    "101D.mp4"
)

echo ""
echo "📋 Configuration:"
echo "   - Modèle ENL0: $ENL0_MODEL"
echo "   - Modèle ENL4: $ENL4_MODEL"
echo "   - Nombre de vidéos: ${#VIDEOS[@]}"
echo ""

# Fonction pour annoter une vidéo
annotate_video() {
    local video=$1
    local model=$2
    local model_name=$3
    
    echo "=========================================="
    echo "🎥 Vidéo: $video"
    echo "🤖 Modèle: $model_name"
    echo "=========================================="
    
    conda run -n pose-estimation python test_video.py \
        --video "$TEST_VIDEOS_DIR/$video" \
        --model "$model" \
        --no-display
    
    if [ $? -eq 0 ]; then
        echo "✅ $video avec $model_name terminé"
    else
        echo "❌ Erreur avec $video et $model_name"
    fi
    echo ""
}

# Annoter avec EfficientNetLite0
echo "=========================================="
echo "🚀 PARTIE 1/2: EfficientNetLite0"
echo "=========================================="
echo ""

for video in "${VIDEOS[@]}"; do
    annotate_video "$video" "$ENL0_MODEL" "ENL0"
done

# Annoter avec EfficientNetLite4
echo "=========================================="
echo "🚀 PARTIE 2/2: EfficientNetLite4"
echo "=========================================="
echo ""

for video in "${VIDEOS[@]}"; do
    annotate_video "$video" "$ENL4_MODEL" "ENL4"
done

echo "=========================================="
echo "✅ TOUTES LES ANNOTATIONS TERMINÉES"
echo "=========================================="
echo ""
echo "📂 Résultats:"
echo "   - ENL0: output/ENL0_20251119_135726/videos/"
echo "   - ENL4: output/ENL4_20251119_150714/videos/"
echo ""
