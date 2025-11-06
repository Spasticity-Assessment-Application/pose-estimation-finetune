#!/bin/bash

# Script d'installation avec Conda (compatible TensorFlow)

echo "=================================================="
echo "🚀 Installation avec Conda - Pose Estimation"
echo "=================================================="

# Vérifier que conda est installé
echo ""
echo "1️⃣  Vérification de Conda..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "✅ Conda trouvé: $CONDA_VERSION"
else
    echo "❌ Conda n'est pas installé!"
    echo "💡 Installez Anaconda ou Miniconda depuis https://www.anaconda.com/"
    exit 1
fi

# Nom de l'environnement
ENV_NAME="pose-estimation"

# Vérifier si l'environnement existe
echo ""
echo "2️⃣  Vérification de l'environnement conda..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  L'environnement '$ENV_NAME' existe déjà"
    read -p "Voulez-vous le supprimer et le recréer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
        echo "✅ Environnement supprimé"
    else
        echo "↪️  Utilisation de l'environnement existant"
    fi
fi

# Créer l'environnement avec Python 3.11
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo ""
    echo "3️⃣  Création de l'environnement conda avec Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
    echo "✅ Environnement créé"
fi

# Activer l'environnement
echo ""
echo "4️⃣  Activation de l'environnement..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "✅ Environnement activé: $ENV_NAME"

# Afficher la version de Python
PYTHON_VERSION=$(python --version)
echo "✅ $PYTHON_VERSION"

# Mettre à jour pip
echo ""
echo "5️⃣  Mise à jour de pip..."
pip install --upgrade pip
echo "✅ pip mis à jour"

# Installer les dépendances
echo ""
echo "6️⃣  Installation des dépendances..."
echo "⏳ Cette étape peut prendre quelques minutes..."
pip install -r requirements.txt
echo "✅ Dépendances installées"

# Tester l'installation
echo ""
echo "7️⃣  Test de l'installation..."
python test_setup.py

# Instructions finales
echo ""
echo "=================================================="
echo "🎉 Installation terminée!"
echo "=================================================="
echo ""
echo "💡 Prochaines étapes:"
echo ""
echo "1. Activez l'environnement conda:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Lancez l'entraînement:"
echo "   python main.py --save-data"
echo ""
echo "3. Pour désactiver l'environnement:"
echo "   conda deactivate"
echo ""
echo "=================================================="
