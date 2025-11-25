"""
Benchmark de vitesse d'inférence pour modèles TFLite
Mesure le temps moyen par frame sur une vidéo
"""
import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
import json
from pathlib import Path


def load_tflite_model(model_path):
    """Charge le modèle TFLite et détecte la taille d'entrée"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Détecter la taille d'entrée
    input_shape = input_details[0]['shape']
    input_size = (input_shape[1], input_shape[2])  # (H, W)
    
    print(f"📏 Taille d'entrée détectée: {input_size[0]}×{input_size[1]}")
    
    return interpreter, input_details, output_details, input_size


def preprocess_frame(frame, input_size=(256, 256)):
    """Prétraite une frame pour le modèle"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    return frame_batch


def benchmark_inference(model_path, video_path, num_frames=100):
    """
    Mesure la vitesse d'inférence moyenne

    Args:
        model_path: Chemin vers le modèle .tflite
        video_path: Chemin vers la vidéo de test
        num_frames: Nombre de frames à tester

    Returns:
        dict: Résultats du benchmark
    """
    print("=" * 60)
    print("🚀 BENCHMARK VITESSE D'INFÉRENCE")
    print("=" * 60)
    print(f"📦 Modèle: {model_path}")
    print(f"🎬 Vidéo: {video_path}")
    print(f"📊 Frames à tester: {num_frames}")

    # Charger le modèle
    print("\n⏳ Chargement du modèle...")
    interpreter, input_details, output_details, input_size = load_tflite_model(model_path)
    print("✅ Modèle chargé")
    print(f"📏 Taille d'entrée: {input_size[0]}×{input_size[1]}")

    # Ouvrir la vidéo
    print("\n⏳ Ouverture de la vidéo...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Vidéo ouverte: {total_frames} frames, {fps:.1f} FPS")

    # Collecter les frames à tester
    frames_to_test = []
    frame_count = 0

    print(f"\n⏳ Collecte de {num_frames} frames...")
    while len(frames_to_test) < num_frames and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Prendre une frame tous les 10 frames pour variété
        if frame_count % 10 == 0:
            frames_to_test.append(frame)

        frame_count += 1

    cap.release()
    print(f"✅ {len(frames_to_test)} frames collectées")

    # Benchmark
    print("\n⏳ Benchmark en cours...")
    inference_times = []

    for i, frame in enumerate(frames_to_test):
        # Pré-traitement
        input_tensor = preprocess_frame(frame, input_size)

        # Inférence
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.perf_counter()

        # Mesurer le temps
        inference_time = (end_time - start_time) * 1000  # en ms
        inference_times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"   Frame {i+1}/{len(frames_to_test)}: {inference_time:.2f} ms")

    # Statistiques
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps_inference = 1000 / mean_time

    results = {
        'model_path': str(model_path),
        'video_path': str(video_path),
        'input_size': [int(input_size[0]), int(input_size[1])],
        'num_frames_tested': len(frames_to_test),
        'mean_inference_time_ms': round(float(mean_time), 3),
        'std_inference_time_ms': round(float(std_time), 3),
        'min_inference_time_ms': round(float(min_time), 3),
        'max_inference_time_ms': round(float(max_time), 3),
        'inference_fps': round(float(fps_inference), 2),
        'video_fps': round(float(fps), 2)
    }

    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DU BENCHMARK")
    print("=" * 60)
    print(f"⏱️  Temps moyen par frame: {results['mean_inference_time_ms']:.2f} ms")
    print(f"🎯 Écart-type: {results['std_inference_time_ms']:.2f} ms")
    print(f"⚡ FPS d'inférence: {results['inference_fps']:.1f}")
    print(f"🎬 FPS vidéo: {results['video_fps']:.1f}")
    print(f"📈 Ratio: {results['inference_fps']/results['video_fps']:.2f}x")

    # Sauvegarder les résultats
    results_path = Path(model_path).parent / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résultats sauvegardés: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark vitesse d'inférence TFLite")
    parser.add_argument('--model', type=str, required=True, help="Chemin vers le modèle .tflite")
    parser.add_argument('--video', type=str, required=True, help="Chemin vers la vidéo de test")
    parser.add_argument('--frames', type=int, default=100, help="Nombre de frames à tester")

    args = parser.parse_args()

    try:
        results = benchmark_inference(
            model_path=args.model,
            video_path=args.video,
            num_frames=args.frames
        )
        print("\n✅ Benchmark terminé avec succès!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()