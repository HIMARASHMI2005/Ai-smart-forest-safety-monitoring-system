import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_ciou(box1, box2, eps=1e-7):
    """
    Computes Complete IoU (CIoU) Loss between box1 and box2.
    Boxes format: [x1, y1, x2, y2]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    iou = inter_area / union_area

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance squared
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


class ForestDetectionLoss(nn.Module):
    """
    Detection loss for ForestPersonNet:
    L_total = lambda_box * L_CIoU + lambda_obj * L_BCE_obj + lambda_cls * L_BCE_cls
    """
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        predictions: dict with 'p3', 'p4', 'p5'
        targets: ground truth anchor targets
        """
        device = predictions["p3"].device
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {"box_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0}

        for scale, pred in predictions.items():
            # Example anchor grid dimension check
            B, C, H, W = pred.shape
            pred_view = pred.view(B, 3, 6, H, W).permute(0, 1, 3, 4, 2)  # [B, 3, H, W, 6]
            
            conf_pred = pred_view[..., 4]
            cls_pred = pred_view[..., 5]

            # In training step, compare against target mask
            obj_target = targets.get(f"{scale}_obj", torch.zeros_like(conf_pred))
            cls_target = targets.get(f"{scale}_cls", torch.zeros_like(cls_pred))

            obj_loss = self.bce_obj(conf_pred, obj_target)
            cls_loss = self.bce_cls(cls_pred, cls_target)

            total_loss += self.lambda_obj * obj_loss + self.lambda_cls * cls_loss
            loss_components["obj_loss"] += obj_loss.item()
            loss_components["cls_loss"] += cls_loss.item()

        return total_loss, loss_components
