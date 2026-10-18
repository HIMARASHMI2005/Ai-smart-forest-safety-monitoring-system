import os
import glob
import cv2
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class ForestPersonsDataset(Dataset):
    """
    Dataset loader for ETRI/KAIST ForestPersons Dataset.
    Supports:
    1. Official COCO benchmark annotations (annotations/annotations/{split}.json)
    2. Per-image label format ({split}/labels/{name}.json)
    Annotations: Bounding Box [x_min, y_min, w, h] for 'Person' in dense foliage / wildland.
    Official splits: 'train', 'val', 'test'
    """
    def __init__(self, data_root: str = "ml/datasets/ForestPersons", split: str = "train", img_size: int = 640, augment: bool = True):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.samples = []

        # Candidate paths for COCO JSON annotations
        coco_candidates = [
            os.path.join(data_root, "annotations", "annotations", f"{split}.json"),
            os.path.join(data_root, "annotations", f"{split}.json"),
        ]
        coco_path = next((p for p in coco_candidates if os.path.exists(p)), None)

        if coco_path:
            with open(coco_path, "r") as f:
                coco_data = json.load(f)

            img_to_boxes = {}
            for ann in coco_data.get("annotations", []):
                img_id = ann["image_id"]
                x, y, w, h = ann["bbox"]
                if img_id not in img_to_boxes:
                    img_to_boxes[img_id] = []
                img_to_boxes[img_id].append([float(x), float(y), float(x + w), float(y + h)])

            images_dir = os.path.join(data_root, "images")
            # Build set of available relative paths for fast O(1) lookup
            available_files = set()
            if os.path.exists(images_dir):
                for folder in os.listdir(images_dir):
                    subpath = os.path.join(images_dir, folder)
                    if os.path.isdir(subpath):
                        for fn in os.listdir(subpath):
                            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                                available_files.add(f"{folder}/{fn}".replace("\\", "/"))

            for img_rec in coco_data.get("images", []):
                file_rel = img_rec["file_name"].replace("\\", "/")
                if file_rel in available_files:
                    img_p = os.path.join(images_dir, file_rel)
                    boxes = img_to_boxes.get(img_rec["id"], [])
                    self.samples.append((img_p, boxes))

        # Fallback to per-folder split
        if not self.samples:
            split_dir = os.path.join(data_root, split)
            if os.path.exists(split_dir):
                img_files = glob.glob(os.path.join(split_dir, "images", "*.jpg")) + \
                            glob.glob(os.path.join(split_dir, "images", "*.png"))
                for img_p in img_files:
                    base_name = os.path.splitext(os.path.basename(img_p))[0]
                    ann_p = os.path.join(split_dir, "labels", f"{base_name}.json")
                    boxes = []
                    if os.path.exists(ann_p):
                        with open(ann_p, "r") as f:
                            data = json.load(f)
                            for item in data.get("annotations", []):
                                x, y, w, h = item["bbox"]
                                boxes.append([float(x), float(y), float(x + w), float(y + h)])
                    self.samples.append((img_p, boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, raw_boxes = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

        h_orig, w_orig = img.shape[:2]
        boxes = [list(b) for b in raw_boxes]

        # Augmentation: Horizontal Flip (p=0.5)
        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            boxes = [[w_orig - b[2], b[1], w_orig - b[0], b[3]] for b in boxes]

        # Resize to (640, 640)
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Rescale boxes to 640x640 space
        scale_x = self.img_size / max(1.0, float(w_orig))
        scale_y = self.img_size / max(1.0, float(h_orig))
        scaled_boxes = []
        for b in boxes:
            x1 = max(0.0, min(float(self.img_size), b[0] * scale_x))
            y1 = max(0.0, min(float(self.img_size), b[1] * scale_y))
            x2 = max(0.0, min(float(self.img_size), b[2] * scale_x))
            y2 = max(0.0, min(float(self.img_size), b[3] * scale_y))
            if x2 > x1 and y2 > y1:
                scaled_boxes.append([x1, y1, x2, y2])

        boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32) if scaled_boxes else torch.zeros((0, 4), dtype=torch.float32)
        return tensor, boxes_tensor
