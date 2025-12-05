#!/usr/bin/env python3
"""
Analyze learning curves from training logs to assess convergence quality.
Examines val_loss and val_mae across warmup, finetune, and unfreeze phases.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_logs(model_path):
    """Load and concatenate training logs from all phases."""
    logs_dir = Path(model_path) / "logs"

    phases = ['warmup', 'finetune', 'unfreeze']
    all_data = []

    epoch_offset = 0

    for phase in phases:
        log_file = logs_dir / f"pose_model_dlc_{phase}_log.csv"
        if log_file.exists():
            df = pd.read_csv(log_file)
            df['epoch'] = df['epoch'] + epoch_offset
            df['phase'] = phase
            all_data.append(df)
            epoch_offset = df['epoch'].max() + 1

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def analyze_convergence(data, model_name):
    """Analyze convergence patterns in the training data."""
    analysis = {}

    # Overall trends
    final_val_loss = data['val_loss'].iloc[-1]
    min_val_loss = data['val_loss'].min()
    final_val_mae = data['val_mae'].iloc[-1]
    min_val_mae = data['val_mae'].min()

    analysis['final_val_loss'] = final_val_loss
    analysis['min_val_loss'] = min_val_loss
    analysis['final_val_mae'] = final_val_mae
    analysis['min_val_mae'] = min_val_mae

    # Check for convergence (last 10 epochs stability)
    last_10_loss = data['val_loss'].tail(10)
    loss_std = last_10_loss.std()
    loss_mean = last_10_loss.mean()
    analysis['loss_stability'] = loss_std / loss_mean  # coefficient of variation

    last_10_mae = data['val_mae'].tail(10)
    mae_std = last_10_mae.std()
    mae_mean = last_10_mae.mean()
    analysis['mae_stability'] = mae_std / mae_mean

    # Check for overfitting (training vs validation divergence)
    if 'loss' in data.columns and 'mae' in data.columns:
        # Compare final training and validation metrics
        final_train_loss = data['loss'].iloc[-1]
        final_train_mae = data['mae'].iloc[-1]

        analysis['overfitting_loss'] = final_train_loss < final_val_loss
        analysis['overfitting_mae'] = final_train_mae < final_val_mae

    return analysis

def plot_learning_curves(models_data, output_dir):
    """Plot learning curves for all models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    colors = ['blue', 'red', 'green']
    model_names = list(models_data.keys())

    for i, (model_name, data) in enumerate(models_data.items()):
        color = colors[i % len(colors)]

        # Val Loss
        ax1.plot(data['epoch'], data['val_loss'], color=color, label=model_name, linewidth=2)
        ax1.set_title('Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Val MAE
        ax2.plot(data['epoch'], data['val_mae'], color=color, label=model_name, linewidth=2)
        ax2.set_title('Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Training Loss (if available)
        if 'loss' in data.columns:
            ax3.plot(data['epoch'], data['loss'], color=color, label=model_name, linewidth=2)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Training MAE (if available)
        if 'mae' in data.columns:
            ax4.plot(data['epoch'], data['mae'], color=color, label=model_name, linewidth=2)
            ax4.set_title('Training MAE')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('MAE')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show

def main():
    """Main analysis function."""
    print("Starting learning curves analysis...")

    # Model paths
    base_path = Path("/Users/jules/Document local/Jules/Cours/ETS/A25/PFE/test fine-tuning/output")

    models = {
        'DLC_Mobil': 'DLC_Mobil_20251120_125726',
        'MNv3S': 'MNv3S_20251124_134351',
        'MNv3L': 'MNv3L_20251125_185928'
    }

    models_data = {}
    analyses = {}

    print("Loading training logs...")

    for model_name, model_dir in models.items():
        model_path = base_path / model_dir
        data = load_training_logs(model_path)

        if data is not None:
            models_data[model_name] = data
            analyses[model_name] = analyze_convergence(data, model_name)
            print(f"✓ Loaded {model_name}: {len(data)} epochs")
        else:
            print(f"✗ Failed to load {model_name}")

    if not models_data:
        print("No training logs found!")
        return

    # Create output directory
    output_dir = Path("learning_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Plot learning curves
    print("\nPlotting learning curves...")
    plot_learning_curves(models_data, output_dir)

    # Print analysis results
    print("\n" + "="*60)
    print("LEARNING CONVERGENCE ANALYSIS")
    print("="*60)

    for model_name, analysis in analyses.items():
        print(f"\n{model_name}:")
        print(f"  Final val_loss: {analysis['final_val_loss']:.6f}")
        print(f"  Min val_loss: {analysis['min_val_loss']:.6f}")
        print(f"  Final val_mae: {analysis['final_val_mae']:.6f}")
        print(f"  Min val_mae: {analysis['min_val_mae']:.6f}")
        print(f"  Loss stability (CV): {analysis['loss_stability']:.4f}")
        print(f"  MAE stability (CV): {analysis['mae_stability']:.4f}")

        if 'overfitting_loss' in analysis:
            print(f"  Overfitting (loss): {'Yes' if analysis['overfitting_loss'] else 'No'}")
        if 'overfitting_mae' in analysis:
            print(f"  Overfitting (MAE): {'Yes' if analysis['overfitting_mae'] else 'No'}")

    # Comparative analysis
    print("\nCOMPARATIVE ANALYSIS:")
    print("-" * 30)

    best_loss_model = min(analyses.keys(), key=lambda x: analyses[x]['final_val_loss'])
    best_mae_model = min(analyses.keys(), key=lambda x: analyses[x]['final_val_mae'])

    print(f"Best final validation loss: {best_loss_model}")
    print(f"Best final validation MAE: {best_mae_model}")

    # Check stability
    stable_models = [m for m, a in analyses.items() if a['loss_stability'] < 0.1]
    if stable_models:
        print(f"Most stable training: {', '.join(stable_models)}")
    else:
        print("All models show some training instability")

    print(f"\nPlots saved to: {output_dir}/learning_curves_comparison.png")

if __name__ == "__main__":
    main()