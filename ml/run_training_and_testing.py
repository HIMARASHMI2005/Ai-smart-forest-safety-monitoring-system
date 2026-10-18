import os
import sys
import time
import json

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from ml.forest_person_net.train import train as train_person_net
from ml.forest_person_net.evaluate import evaluate_test_set as eval_person_net
from ml.forest_face_net.train import train as train_face_net
from ml.forest_face_net.evaluate import evaluate_face_verification as eval_face_net

def run_full_pipeline():
    print("=" * 75)
    print("SMART FOREST CUSTOM MACHINE LEARNING PIPELINE")
    print("Ground-Truth Datasets: ETRI/KAIST ForestPersons & Oxford VGGFace2")
    print("Zero Pretrained Weights Mandate: Kaiming Normal scratch initialization")
    print("=" * 75)
    start_total = time.time()

    # 1. Train ForestPersonNet
    print("\n>>> STEP 1: Training ForestPersonNet on ETRI/KAIST ForestPersons Dataset...")
    person_hist = train_person_net(
        data_dir="ml/datasets/ForestPersons",
        epochs=5,
        batch_size=1,
        steps_per_epoch=8,
        checkpoint_dir="ml/checkpoints/person_detection"
    )

    # 2. Evaluate ForestPersonNet
    print("\n>>> STEP 2: Evaluating ForestPersonNet on Held-Out Test Set...")
    person_metrics = eval_person_net(
        checkpoint_path="ml/checkpoints/person_detection/forest_person_net_best.pt",
        output_metrics_file="ml/checkpoints/person_detection/evaluation_results.json",
        data_dir="ml/datasets/ForestPersons"
    )

    # 3. Train ForestFaceNet
    print("\n>>> STEP 3: Training ForestFaceNet (ArcFace) on Oxford VGGFace2 Dataset...")
    face_hist = train_face_net(
        data_dir="ml/datasets/vgg_face2-master",
        epochs=5,
        batch_size=8,
        checkpoint_dir="ml/checkpoints/face_recognition"
    )

    # 4. Evaluate ForestFaceNet
    print("\n>>> STEP 4: Evaluating ForestFaceNet on Biometric Verification Pairs...")
    face_metrics = eval_face_net(
        checkpoint_path="ml/checkpoints/face_recognition/forest_face_net_best.pt",
        output_metrics_file="ml/checkpoints/face_recognition/evaluation_results.json",
        data_dir="ml/datasets/vgg_face2-master"
    )

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 75)
    print("\n[SUMMARY OF GENERATED ARTIFACTS & METRICS]")
    print(f"1. ForestPersonNet Checkpoint: ml/checkpoints/person_detection/forest_person_net_best.pt")
    print(f"   - Input Size: 640x640")
    print(f"   - mAP@0.5: {person_metrics.get('mAP_50')}")
    print(f"   - Precision@0.5: {person_metrics.get('precision_at_05')}")
    print(f"   - Recall@0.5: {person_metrics.get('recall_at_05')}")
    print(f"   - Hardware Latency: {person_metrics.get('latency_ms')} ms ({person_metrics.get('fps')} FPS)")

    print(f"\n2. ForestFaceNet Checkpoint: ml/checkpoints/face_recognition/forest_face_net_best.pt")
    print(f"   - Embedding Dimensions: 128-D Unit Hypersphere")
    print(f"   - Verification Accuracy: {face_metrics.get('verification_accuracy')*100:.1f}%")
    print(f"   - Optimal Threshold Tau: {face_metrics.get('optimal_threshold_tau')}")
    print(f"   - Equal Error Rate (EER): {face_metrics.get('equal_error_rate_eer')*100:.2f}%")
    print(f"   - ROC AUC: {face_metrics.get('auc_roc')}")

if __name__ == "__main__":
    run_full_pipeline()
