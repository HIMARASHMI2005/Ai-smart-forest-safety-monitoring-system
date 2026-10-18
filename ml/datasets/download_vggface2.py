"""
Download & Manifest Verification for Oxford VGGFace2 Dataset
Total: 3.31 Million Images across 9,131 Identities
Official Split: Train: 8,631 IDs | Test: 500 IDs
"""
import os
import argparse
import json

def verify_vggface2_structure(data_dir: str = "data/VGGFace2"):
    print(f"Checking VGGFace2 directory at {data_dir}...")
    splits = ["train", "test"]
    stats = {}
    for s in splits:
        split_dir = os.path.join(data_dir, s)
        os.makedirs(split_dir, exist_ok=True)
        identities = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        stats[s] = {
            "split_dir": split_dir,
            "identities_count": len(identities)
        }

    manifest_file = os.path.join(data_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"VGGFace2 manifest updated: {manifest_file}")
    return stats

if __name__ == "__main__":
    verify_vggface2_structure()
