#!/usr/bin/env python3
"""
Extract specific frames from video and annotate them with pose models
"""
import cv2
import os
from pathlib import Path

def extract_frames(video_path, frame_numbers, output_dir):
    """Extract specific frames from video"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Vidéo: {total_frames} frames total")

    extracted = 0
    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            print(f"⚠️ Frame {frame_num} dépasse le total ({total_frames}), ignoré")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            output_path = output_dir / f"frame_{frame_num:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"✅ Frame {frame_num} sauvegardé: {output_path}")
            extracted += 1
        else:
            print(f"❌ Impossible de lire le frame {frame_num}")

    cap.release()
    return extracted == len(frame_numbers)

def main():
    video_path = "test-videos/301D_1.mov"
    frame_numbers = [0, 20, 40, 60]  # Frames à extraire (espacés régulièrement)
    output_dir = Path("frames_extraction")

    print("🎬 EXTRACTION DE FRAMES")
    print("=" * 40)

    if extract_frames(video_path, frame_numbers, output_dir):
        print(f"\n✅ {len(frame_numbers)} frames extraits avec succès")
        print(f"📂 Répertoire: {output_dir}")
    else:
        print("\n❌ Erreur lors de l'extraction")
        return

    # Lister les frames extraits
    print("\n📋 Frames extraits:")
    for frame_file in sorted(output_dir.glob("*.jpg")):
        print(f"   - {frame_file.name}")

if __name__ == "__main__":
    main()