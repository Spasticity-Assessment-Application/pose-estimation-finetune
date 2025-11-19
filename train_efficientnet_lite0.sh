#!/bin/bash

###############################################################################
# Script d'entraînement EfficientNetLite0 avec modifications minimales
# Ce script adapte automatiquement la taille des images et heatmaps
# tout en gardant les mêmes paramètres d'entraînement pour comparaison équitable
###############################################################################

echo "=========================================="
echo "🚀 ENTRAÎNEMENT EFFICIENTNETLITE0"
echo "=========================================="

# Configuration
BACKBONE="EfficientNetLite0"
IMAGE_SIZE=224  # Taille recommandée pour EfficientNetLite0
HEATMAP_SIZE=74  # Proportionnel: 224/192 * 64 ≈ 74

echo ""
echo "📋 Configuration:"
echo "   - Backbone: $BACKBONE"
echo "   - Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "   - Heatmap size: ${HEATMAP_SIZE}x${HEATMAP_SIZE}"
echo ""

# Créer un fichier temporaire pour ajuster HEATMAP_SIZE
echo "🔧 Ajustement de la taille des heatmaps..."

# Sauvegarder la configuration originale
cp config.py config.py.backup

# Modifier HEATMAP_SIZE dans config.py
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/HEATMAP_SIZE = (64, 64)/HEATMAP_SIZE = ($HEATMAP_SIZE, $HEATMAP_SIZE)/" config.py
else
    # Linux
    sed -i "s/HEATMAP_SIZE = (64, 64)/HEATMAP_SIZE = ($HEATMAP_SIZE, $HEATMAP_SIZE)/" config.py
fi

echo "✅ Configuration ajustée temporairement"
echo ""

# Fonction de nettoyage pour restaurer config.py
cleanup() {
    echo ""
    echo "🔄 Restauration de la configuration originale..."
    mv config.py.backup config.py
    echo "✅ Configuration restaurée"
}

# S'assurer que cleanup est appelé à la fin ou en cas d'erreur
trap cleanup EXIT

# Lancer l'entraînement
echo "=========================================="
echo "🏋️  LANCEMENT DE L'ENTRAÎNEMENT"
echo "=========================================="
echo ""

# Option 1: Entraînement standard
# python main.py --backbone $BACKBONE --save-data

# Option 2: Entraînement avancé (RECOMMANDÉ pour meilleures performances)
python main.py --backbone $BACKBONE --save-data --advanced-training

# Le script cleanup sera automatiquement appelé ici grâce au trap EXIT

echo ""
echo "=========================================="
echo "✅ ENTRAÎNEMENT TERMINÉ"
echo "=========================================="
echo ""
echo "💡 Résultats sauvegardés dans: output/ENL0_YYYYMMDD_HHMMSS/"
echo ""
echo "📊 Pour comparer avec MobileNetV2:"
echo "   python quick_compare.py"
echo ""
