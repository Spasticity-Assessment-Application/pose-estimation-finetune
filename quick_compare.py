"""
Script rapide de comparaison Keras vs TFLite
"""
import numpy as np
from compare_models import compare_models

# Charger les données
print("📂 Chargement des données...")
data = np.load('output/preprocessed_data.npz')
X_val = data['X_val']

# Comparer les modèles
results = compare_models(
    keras_path='output/models/pose_model_20251105_115946_best.h5',
    tflite_path='output/models/pose_model_quantized.tflite',
    X_test=X_val,
    num_samples=50  # Tous les échantillons de validation
)

print("\n" + "="*60)
print("📝 RÉSUMÉ")
print("="*60)
print(f"Distance moyenne normalisée: {results['avg_distance']:.4f}")
print(f"En pixels (192x192): {results['avg_distance'] * 192:.1f} px")
print(f"Distance maximale: {results['max_distance']:.4f}")
print("="*60)
