#!/usr/bin/env python3
"""
Simple script to annotate frames using existing prediction functions
"""
import cv2
import os
from pathlib import Path
import sys

# Add current directory to path to import local modules
sys.path.append('.')

from test_video_keras import load_keras_model, load_model_config, predict_frame, extract_keypoints_from_heatmaps, draw_keypoints, get_model_input_size

def annotate_single_frame(frame_path, model_path, output_path):
    """Annotate a single frame using the existing functions"""

    # Load model and config
    model = load_keras_model(model_path)
    if model is None:
        return False

    model_config = load_model_config(model_path)
    input_size = get_model_input_size(model)

    # Load image
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"❌ Impossible de charger: {frame_path}")
        return False

    # Predict
    heatmaps = predict_frame(model, frame, input_size)

    # Extract keypoints
    expected_heatmap_size = None
    if model_config and 'heatmap_size' in model_config:
        expected_heatmap_size = tuple(model_config['heatmap_size'])

    keypoints = extract_keypoints_from_heatmaps(heatmaps, frame.shape, expected_heatmap_size)

    # Draw keypoints
    annotated = frame.copy()
    draw_keypoints(annotated, keypoints)

    # Save
    cv2.imwrite(output_path, annotated)
    return True

def main():
    frames_dir = Path("frames_extraction")
    output_dir = Path("frames_annotated")
    output_dir.mkdir(exist_ok=True)

    models = {
        'DLC_Mobil': 'output/DLC_Mobil_20251120_125726/models/pose_model_dlc_finetune_best.h5',
        'MNv3S': 'output/MNv3S_20251124_134351/models/pose_model_dlc_finetune_best.h5',
        'MNv3L': 'output/MNv3L_20251125_185928/models/pose_model_dlc_finetune_best.h5'
    }

    frame_files = ['frame_0000.jpg', 'frame_0020.jpg', 'frame_0040.jpg', 'frame_0060.jpg']

    print("🎯 ANNOTATION SIMPLE DE FRAMES")
    print("=" * 40)

    total_processed = 0

    for frame_file in frame_files:
        frame_path = frames_dir / frame_file
        if not frame_path.exists():
            continue

        print(f"\n📸 {frame_file}")

        for model_name, model_path in models.items():
            output_filename = f"{frame_file.replace('.jpg', '')}_{model_name}.jpg"
            output_path = output_dir / output_filename

            print(f"   🤖 {model_name}...")

            if annotate_single_frame(str(frame_path), model_path, str(output_path)):
                print(f"   ✅ {output_filename}")
                total_processed += 1
            else:
                print(f"   ❌ Échec {model_name}")

    print("\n✅ TERMINÉ")
    print(f"📂 {total_processed} images créées dans {output_dir}")

    # List created files
    if output_dir.exists():
        files = list(output_dir.glob("*.jpg"))
        print(f"\n📋 Fichiers créés:")
        for f in sorted(files):
            print(f"   - {f.name}")

if __name__ == "__main__":
    main()