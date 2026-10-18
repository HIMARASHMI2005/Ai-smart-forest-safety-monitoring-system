import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from ml.forest_person_net.architecture import ForestPersonNet

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Standard greedy Non-Maximum Suppression (NMS).
    boxes: (N, 4) in [x1, y1, x2, y2]
    scores: (N,)
    Returns list of kept indices.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class PersonDetector:
    """
    Inference wrapper for ForestPersonNet.
    Preprocesses (640x640), runs forward pass, decodes coordinates,
    applies confidence thresholding & NMS.
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = ForestPersonNet(num_classes=1).to(self.device)
        self.model.eval()

        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                print(f"Loaded ForestPersonNet weights from {checkpoint_path}")
            except Exception as e:
                print(f"Note: Running ForestPersonNet initialized from scratch ({e})")

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Resizes letterboxed to 640x640 and normalizes [0, 1]."""
        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, (h_orig, w_orig)

    def decode_predictions(self, raw_outputs: Dict[str, torch.Tensor], orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Decodes multi-scale head predictions [p3, p4, p5] into normalized image boxes.
        """
        h_orig, w_orig = orig_shape
        candidate_boxes = []
        candidate_scores = []

        # Anchor box priors for 640x640
        anchors = {
            "p3": [(10, 13), (16, 30), (33, 23)],      # 80x80
            "p4": [(30, 61), (62, 45), (59, 119)],     # 40x40
            "p5": [(116, 90), (156, 198), (373, 326)]  # 20x20
        }
        strides = {"p3": 8, "p4": 16, "p5": 32}

        for scale_name, head_out in raw_outputs.items():
            # head_out: (1, 18, GridH, GridW)
            out = head_out.squeeze(0).cpu().detach()
            grid_h, grid_w = out.shape[1], out.shape[2]
            stride = strides[scale_name]
            scale_anchors = anchors[scale_name]

            out_view = out.view(3, 6, grid_h, grid_w).permute(0, 2, 3, 1)  # (3, H, W, 6)

            # Sigmoid activation on objectness and class
            conf = torch.sigmoid(out_view[..., 4])
            cls_prob = torch.sigmoid(out_view[..., 5])
            score = conf * cls_prob

            # Mask out low scores
            mask = score > self.conf_thresh
            if not mask.any():
                continue

            indices = torch.nonzero(mask)
            for idx in indices:
                a_idx, gy, gx = idx[0].item(), idx[1].item(), idx[2].item()
                s = score[a_idx, gy, gx].item()
                raw_box = out_view[a_idx, gy, gx, :4]

                # Decode tx, ty, tw, th
                bx = (torch.sigmoid(raw_box[0]).item() * 2 - 0.5 + gx) * stride
                by = (torch.sigmoid(raw_box[1]).item() * 2 - 0.5 + gy) * stride
                pw, ph = scale_anchors[a_idx]
                bw = (torch.sigmoid(raw_box[2]).item() * 2) ** 2 * pw
                bh = (torch.sigmoid(raw_box[3]).item() * 2) ** 2 * ph

                # Map back to original image space
                scale_x = w_orig / 640.0
                scale_y = h_orig / 640.0

                x1 = max(0, int((bx - bw / 2) * scale_x))
                y1 = max(0, int((by - bh / 2) * scale_y))
                x2 = min(w_orig, int((bx + bw / 2) * scale_x))
                y2 = min(h_orig, int((by + bh / 2) * scale_y))

                if x2 > x1 and y2 > y1:
                    candidate_boxes.append([x1, y1, x2, y2])
                    candidate_scores.append(s)

        if not candidate_boxes:
            return []

        boxes_np = np.array(candidate_boxes, dtype=np.float32)
        scores_np = np.array(candidate_scores, dtype=np.float32)
        keep = non_max_suppression(boxes_np, scores_np, self.iou_thresh)

        detections = []
        for k in keep:
            box = boxes_np[k].astype(int)
            detections.append({
                "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])], # [x, y, w, h]
                "coords": [int(box[0]), int(box[1]), int(box[2]), int(box[3])], # [x1, y1, x2, y2]
                "confidence": round(float(scores_np[k]), 4),
                "class_id": 0,
                "label": "Person"
            })
        return detections

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Full detection pipeline on raw BGR frame."""
        tensor, orig_shape = self.preprocess(img_bgr)
        with torch.no_grad():
            outputs = self.model(tensor)
        return self.decode_predictions(outputs, orig_shape)
