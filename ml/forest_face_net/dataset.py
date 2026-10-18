import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class VGGFace2Dataset(Dataset):
    """
    Dataset loader for Oxford VGGFace2 Dataset.
    Identities: Subdirectories named by ID (e.g. n000106, n007241, n008937).
    Images aligned and normalized to (3, 112, 112) with standard face normalization:
    (pixel - 127.5) / 128.0
    """
    def __init__(self, data_root: str = "ml/datasets/vgg_face2-master", split: str = "train", img_size: int = 112, augment: bool = True):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.samples = []
        self.class_to_idx = {}

        # Search candidates for identity folders
        candidates = [
            os.path.join(data_root, split),
            os.path.join(data_root, "samples (test set)", "tight_crop (used for training)"),
            os.path.join(data_root, "samples (test set)", "loose_crop (release version)"),
            os.path.join(data_root, "tight_crop"),
            os.path.join(data_root, "loose_crop"),
            data_root
        ]

        found_dir = None
        for cand in candidates:
            if os.path.exists(cand):
                subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d)) and not d.startswith('.')]
                if subdirs:
                    found_dir = cand
                    break

        if found_dir:
            identities = sorted([d for d in os.listdir(found_dir) if os.path.isdir(os.path.join(found_dir, d)) and not d.startswith('.')])
            self.class_to_idx = {name: idx for idx, name in enumerate(identities)}
            for name in identities:
                id_dir = os.path.join(found_dir, name)
                for f in glob.glob(os.path.join(id_dir, "*.jpg")) + glob.glob(os.path.join(id_dir, "*.png")):
                    self.samples.append((f, self.class_to_idx[name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Standard face normalization: (pixel - 127.5) / 128.0
        norm_img = (img_rgb.astype(np.float32) - 127.5) / 128.0
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()

        return tensor, torch.tensor(label, dtype=torch.long)
