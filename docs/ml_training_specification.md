# Machine Learning Training Specification & Mathematical Formulations
## Custom Vision Architectures (Zero Pretrained Weights)

### 1. Zero Pretrained Weights Mandate
To ensure complete algorithmic ownership, eliminate third-party intellectual property constraints, prevent backdoor vulnerabilities, and adapt specifically to dense forest canopies and extreme wildland occlusions:
- **Zero Pretrained Weights:** Every neural network parameter in ForestPersonNet and ForestFaceNet is initialized from scratch using Kaiming Normal (He Normal) weight initialization:
  $$W \sim \mathcal{N}\left(0, \; \sqrt{\frac{2}{\text{fan\_out}}}\right)$$
- **No Transfer Learning:** No pretrained weights from ImageNet, COCO, Pascal VOC, or proprietary face models (e.g., FaceNet, InsightFace, DeepFace) are utilized.
- **Strict Dataset Discipline:** Training and validation are executed strictly on the official ground-truth partitions of the **ETRI/KAIST ForestPersons** and **Oxford VGGFace2** datasets.

---

### 2. ForestPersonNet: Custom Human Detector for Wildland & SAR Environments

#### 2.1 Model Architecture
ForestPersonNet processes an input RGB image tensor $\mathbf{X} \in \mathbb{R}^{B \times 3 \times 640 \times 640}$ normalized to $[0, 1]$.

```
Input: (3, 640, 640) Image Tensor (Normalized: [0, 1])
  |
  +--> Stage 1: Conv2d(3->32, k=3, s=2, p=1) + BatchNorm + LeakyReLU(0.1)  --> (32, 320, 320)
  +--> Stage 2: 2x ResBlock(32) + Conv2d(32->64, k=3, s=2, p=1) + BN + LReLU --> (64, 160, 160)
  +--> Stage 3: 4x ResBlock(64) + Conv2d(64->128, k=3, s=2, p=1) + BN + LReLU --> (128, 80, 80)  [P3]
  +--> Stage 4: 6x ResBlock(128) + Conv2d(128->256, k=3, s=2, p=1) + BN + LReLU --> (256, 40, 40) [P4]
  +--> Stage 5: 4x ResBlock(256) + Conv2d(256->512, k=3, s=2, p=1) + BN + LReLU --> (512, 20, 20) [P5]
  |
  +--> Feature Pyramid Aggregator (FPN / PAN Neck):
  |      Top-down & bottom-up fusion fusing multi-scale features:
  |      - Large targets: P5 branch (20x20 grid)
  |      - Medium targets: P4 branch (40x40 grid)
  |      - Distant/Occluded targets: P3 branch (80x80 grid)
  |
  +--> Multi-Scale Detection Heads:
  |      Each head outputs: (Grid_H, Grid_W, 3_Anchors, [tx, ty, tw, th, conf, prob_person])
  |
  +--> Loss Formulation:
  |      L_total = lambda_box * L_CIoU + lambda_obj * L_BCE_obj + lambda_cls * L_BCE_cls
  |
  +--> Inference: Coordinate Decoders + Non-Maximum Suppression (IoU thresh = 0.45, Conf thresh = 0.35)
```

#### 2.2 Complete Loss Formulation
The total optimization objective $\mathcal{L}_{\text{total}}$ is a composite multi-task loss:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{obj}} \mathcal{L}_{\text{BCE\_obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE\_cls}}$$
Where:
- $\lambda_{\text{box}} = 0.05$, $\lambda_{\text{obj}} = 1.0$, $\lambda_{\text{cls}} = 0.5$.

1. **Complete Intersection over Union (CIoU) Loss:**
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
   Where:
   - $\rho(b, b^{gt})$ is the Euclidean distance between predicted and ground-truth box center points.
   - $c$ is the diagonal length of the smallest enclosing box covering both boxes.
   - $v = \frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$ measures the aspect ratio consistency.
   - $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ is a dynamic weighting factor.

2. **Binary Cross-Entropy Objectness Loss:**
   $$\mathcal{L}_{\text{BCE\_obj}} = -\sum_{i} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
   Evaluated across all anchor positions on grids $80 \times 80$, $40 \times 40$, and $20 \times 20$.

#### 2.3 Post-Processing & Non-Maximum Suppression (NMS)
Predicted bounding box offsets $(t_x, t_y, t_w, t_h)$ are converted to pixel coordinates:
$$b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y, \quad b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}$$
Redundant detections are filtered through greedy Non-Maximum Suppression:
- Confidence threshold $\tau_{\text{conf}} = 0.35$
- IoU overlap threshold $\tau_{\text{IoU}} = 0.45$

---

### 3. ForestFaceNet: Custom 128-D Metric Learning Face Representation Network

#### 3.1 Model Architecture
ForestFaceNet processes an aligned facial crop $\mathbf{X}_f \in \mathbb{R}^{B \times 3 \times 112 \times 112}$ normalized by $\mu = 127.5, \sigma = 128.0$.

```
Input: (3, 112, 112) Aligned Face Crop
  |
  +--> Conv2d(3->32, k=3, s=1, p=1) + BatchNorm + PReLU
  +--> DepthwiseSeparableConv(32->64, s=2) + BatchNorm + PReLU           --> (64, 56, 56)
  +--> 3x ResidualBlock(64)
  +--> DepthwiseSeparableConv(64->128, s=2) + BatchNorm + PReLU          --> (128, 28, 28)
  +--> 4x ResidualBlock(128)
  +--> DepthwiseSeparableConv(128->256, s=2) + BatchNorm + PReLU         --> (256, 14, 14)
  +--> 6x ResidualBlock(256)
  +--> DepthwiseSeparableConv(256->512, s=2) + BatchNorm + PReLU         --> (512, 7, 7)
  +--> 2x ResidualBlock(512)
  +--> Global Average Pooling 2D                                         --> (512)
  +--> Linear(512 -> 128) + BatchNorm (Dense Feature Bottleneck)         --> (128)
  +--> L2 Normalization Layer:  f(x) = x / ||x||_2                       --> (128, Unit Sphere)
```

#### 3.2 ArcFace (Additive Angular Margin) Loss Formulation
To maximize inter-class separation and minimize intra-class variance on a 128-dimensional hypersphere, the model is trained with ArcFace loss:
$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
Where:
- $\cos \theta_j = \frac{\mathbf{W}_j^T \mathbf{x}_i}{\|\mathbf{W}_j\| \|\mathbf{x}_i\|}$ is the cosine similarity between normalized embedding $\mathbf{x}_i$ and class weight $\mathbf{W}_j$.
- $s = 64.0$ is the radius hypersphere scaling parameter.
- $m = 0.5$ is the additive angular margin penalty in radians.

#### 3.3 Biometric Match Decision Rule
For two facial embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{S}^{127}$:
$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} \quad (\text{since } \|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0)$$
Decision boundary:
$$\text{Decision} = \begin{cases} \text{VERIFIED} & \text{if } \text{Sim}(\mathbf{u}, \mathbf{v}) \ge \tau \\ \text{UNKNOWN / REJECT} & \text{if } \text{Sim}(\mathbf{u}, \mathbf{v}) < \tau \end{cases}$$
The operational threshold $\tau = 0.65$ is selected at the equal error rate (EER) point on the validation ROC curve.

---

### 4. Training Hyperparameters & Execution Protocol

| Parameter | ForestPersonNet | ForestFaceNet |
|:---|:---|:---|
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | SGD with Momentum ($\mu = 0.9$) |
| **Initial Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-1}$ |
| **LR Schedule** | Cosine Annealing with 5-epoch warmup | Multi-Step Decay ($0.1\times$ at epochs 10, 18, 22) |
| **Weight Decay** | $5 \times 10^{-4}$ | $5 \times 10^{-4}$ |
| **Batch Size** | 16 (Gradient Accumulation = 2) | 64 |
| **Total Epochs** | 50 | 25 |
| **Hardware Target** | NVIDIA RTX / CUDA, Fallback CPU | NVIDIA RTX / CUDA, Fallback CPU |
