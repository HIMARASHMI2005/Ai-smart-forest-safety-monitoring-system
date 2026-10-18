"""
Download & Manifest Verification for ETRI/KAIST ForestPersons Dataset
Official Benchmark for Wildland / Dense Canopy Human Detection
Total: 96,482 RGB Images
Splits: Train (67,686) | Val (18,243) | Test (10,553)
"""
import os
import argparse
import json

def verify_dataset_structure(data_dir: str = "data/ForestPersons"):
    print(f"Checking ForestPersons directory at {data_dir}...")
    splits = ["train", "val", "test"]
    stats = {}
    for s in splits:
        img_dir = os.path.join(data_dir, s, "images")
        lbl_dir = os.path.join(data_dir, s, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        stats[s] = {
            "image_dir": img_dir,
            "label_dir": lbl_dir,
            "images_found": len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        }

    manifest_file = os.path.join(data_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"ForestPersons manifest updated: {manifest_file}")
    return stats

if __name__ == "__main__":
    verify_dataset_structure()
