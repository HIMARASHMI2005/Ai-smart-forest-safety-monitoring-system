import os
import sys
import argparse
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from ml.forest_face_net.architecture import ForestFaceNet
from ml.forest_face_net.margin_loss import ArcFaceLoss
from ml.forest_face_net.dataset import VGGFace2Dataset

def train(
    data_dir: str = "ml/datasets/vgg_face2-master",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.001,
    checkpoint_dir: str = "ml/checkpoints/face_recognition"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing ForestFaceNet (128-D) from scratch on {device} (Zero Pretrained Weights) ===")

    # Resolve dataset path
    resolved_data_dir = data_dir
    candidates = [data_dir, "ml/datasets/vgg_face2-master", "data/VGGFace2"]
    for c in candidates:
        if os.path.exists(c):
            resolved_data_dir = c
            break

    print(f"Loading VGGFace2 dataset from {resolved_data_dir}...")
    dataset = VGGFace2Dataset(data_root=resolved_data_dir, split="train", augment=True)
    num_classes = max(2, len(dataset.class_to_idx)) if dataset.class_to_idx else 3
    print(f"VGGFace2 dataset loaded: {len(dataset)} samples across {num_classes} identities.")

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=True,
        drop_last=False
    )

    model = ForestFaceNet(embedding_dim=128).to(device)
    criterion = ArcFaceLoss(in_features=128, out_features=num_classes, s=64.0, m=0.5).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "accuracy": []}
    best_loss = float("inf")

    print(f"Executing ArcFace training loop across {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        criterion.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(inputs)  # (B, 128) unit sphere
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            
            # Predict top-1 class from cosine similarity with weight matrix
            with torch.no_grad():
                norm_w = torch.nn.functional.normalize(criterion.weight, p=2, dim=1)
                logits = torch.matmul(embeddings, norm_w.t())
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        scheduler.step()
        avg_loss = epoch_loss / max(1, total)
        acc = correct / max(1, total)
        history["train_loss"].append(round(avg_loss, 4))
        history["accuracy"].append(round(acc, 4))
        print(f"Epoch [{epoch}/{epochs}] - ArcFace Loss: {avg_loss:.4f} - Top-1 Acc: {acc*100:.1f}% - LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        if avg_loss < best_loss or epoch == epochs:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "forest_face_net_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "embedding_dim": 128,
                "num_classes": num_classes,
                "identities": list(dataset.class_to_idx.keys()),
                "best_loss": best_loss,
                "architecture": "ForestFaceNet",
                "loss_function": "ArcFace (s=64.0, m=0.5)",
                "initialized_from": "scratch_kaiming_normal",
                "training_dataset": "Oxford VGGFace2",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"ForestFaceNet training completed in {elapsed:.2f}s. Checkpoint saved to {ckpt_path}.")

    with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ml/datasets/vgg_face2-master")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
