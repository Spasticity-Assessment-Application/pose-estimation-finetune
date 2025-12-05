#!/usr/bin/env python3
"""
Quick assessment of model training quality.
Provides a simple verdict on whether training proceeded normally.
"""

import pandas as pd
from pathlib import Path

def assess_training_quality(model_path, model_name):
    """Assess if training proceeded normally."""
    logs_dir = Path(model_path) / "logs"

    # Load all phases
    phases = ['warmup', 'finetune', 'unfreeze']
    all_data = []

    for phase in phases:
        log_file = logs_dir / f"pose_model_dlc_{phase}_log.csv"
        if log_file.exists():
            df = pd.read_csv(log_file)
            all_data.append(df)

    if not all_data:
        return "❌ ÉCHEC - Logs introuvables"

    data = pd.concat(all_data, ignore_index=True)

    # Key metrics
    final_val_loss = data['val_loss'].iloc[-1]
    initial_val_loss = data['val_loss'].iloc[0]
    min_val_loss = data['val_loss'].min()

    final_val_mae = data['val_mae'].iloc[-1]
    initial_val_mae = data['val_mae'].iloc[0]
    min_val_mae = data['val_mae'].min()

    # Stability (coefficient of variation of last 10 epochs)
    last_10_loss = data['val_loss'].tail(10)
    loss_cv = last_10_loss.std() / last_10_loss.mean()

    last_10_mae = data['val_mae'].tail(10)
    mae_cv = last_10_mae.std() / last_10_mae.mean()

    # Assessment criteria
    loss_improved = final_val_loss < initial_val_loss * 0.8  # At least 20% improvement
    mae_improved = final_val_mae < initial_val_mae * 0.8
    loss_stable = loss_cv < 0.15  # CV < 15%
    mae_stable = mae_cv < 0.20   # CV < 20%
    reasonable_final_loss = final_val_loss < 0.001  # Not too high
    reasonable_final_mae = final_val_mae < 0.01

    # Verdict
    good_criteria = sum([loss_improved, mae_improved, loss_stable, mae_stable,
                        reasonable_final_loss, reasonable_final_mae])

    if good_criteria >= 5:
        verdict = "✅ NOMINAL"
    elif good_criteria >= 3:
        verdict = "⚠️ ACCEPTABLE"
    else:
        verdict = "❌ PRÉOCCUPANT"

    return f"{verdict} - Loss: {final_val_loss:.6f}, MAE: {final_val_mae:.6f}, Stabilité: {loss_cv:.3f}"

def main():
    """Quick training assessment for all models."""
    base_path = Path("output")

    models = {
        'DLC_Mobil': 'DLC_Mobil_20251120_125726',
        'MNv3S': 'MNv3S_20251124_134351',
        'MNv3L': 'MNv3L_20251125_185928'
    }

    print("ÉVALUATION RAPIDE DE L'APPRENTISSAGE")
    print("=" * 50)

    for model_name, model_dir in models.items():
        model_path = base_path / model_dir
        assessment = assess_training_quality(model_path, model_name)
        print(f"{model_name:10}: {assessment}")

    print("\nCritères évalués:")
    print("- Amélioration des pertes (>20%)")
    print("- Stabilité en fin d'entraînement")
    print("- Valeurs finales raisonnables")

if __name__ == "__main__":
    main()