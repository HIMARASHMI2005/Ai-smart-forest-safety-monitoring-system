# Ground-Truth Datasets Attribution & Specifications

## Overview of Official Datasets
In accordance with the **Zero Pretrained Weights** mandate, all computer vision models in the Smart Forest System are trained exclusively on official ground-truth benchmark datasets. No synthetic labels, artificial mock embeddings, or cross-split contaminations are permitted.

---

### 1. ForestPersons Dataset (ETRI / KAIST)

#### 1.1 Dataset Summary
- **Origin & Authors:** Electronics and Telecommunications Research Institute (ETRI) & Korea Advanced Institute of Science and Technology (KAIST).
- **Target Domain:** Mountain search-and-rescue (SAR), dense forest hiking corridors, wildland foliage surveillance, and drone-based overhead monitoring.
- **Total Images:** 96,482 High-Resolution RGB Images.
- **Resolution:** $1920 \times 1080$ to $3840 \times 2160$.

#### 1.2 Official Partitioning
| Split | Image Count | Percentage | Purpose |
|:---|:---|:---|:---|
| **Training Set** | 67,686 | 70.15% | Optimization of ForestPersonNet weights |
| **Validation Set** | 18,243 | 18.91% | Hyperparameter tuning and checkpoint selection |
| **Test Set (Held-Out)**| 10,553 | 10.94% | Final unedited benchmark metrics generation |

#### 1.3 Annotation Specification
- **Format:** Bounding box format $[x_{\min}, y_{\min}, w, h]$ normalized to $[0, 1]$ relative to image dimensions.
- **Target Category:** Class `0` (`person`).
- **Domain Challenges:**
  - High degree of foliage and branch occlusion ($>60\%$ partial obstruction).
  - Dappled sunlight, tree canopy shadows, and high dynamic range glare.
  - Earth-toned camouflage, mountaineering backpacks, and non-upright postures (crawling, crouching, injured).

#### 1.4 Preprocessing & Augmentation Pipeline
- Resized to $640 \times 640$ via letterboxing (maintaining original aspect ratio with neutral gray padding).
- Random horizontal flip ($p = 0.5$).
- Scale jitter ($\pm 15\%$) and translation ($\pm 10\%$).
- Photometric distortion: Brightness adjustment ($\pm 12\%$) and contrast jitter ($\pm 15\%$).
- Prohibited: Color inversion, extreme solarization, and unreal rainbow distortion.

---

### 2. VGGFace2 Dataset (Visual Geometry Group, Oxford)

#### 2.1 Dataset Summary
- **Origin & Authors:** Visual Geometry Group (VGG), Department of Engineering Science, University of Oxford.
- **Reference:** Cao, Q., Shen, L., Xie, W., Parkhi, O. M., & Zisserman, A. (2018). *VGGFace2: A dataset for recognising faces across pose and illumination*. IEEE FG 2018.
- **Target Domain:** Extreme pose variation, diverse ethnic backgrounds, varying age ranges, and uncontrolled lighting.
- **Total Dataset Size:** 3.31 Million Images across 9,131 distinct identities.

#### 2.2 Official Partitioning
| Split | Identity Count | Image Count | Purpose |
|:---|:---|:---|:---|
| **Training Set** | 8,631 Identities | ~3.14 Million Images | Training ForestFaceNet via ArcFace loss |
| **Evaluation Set** | 500 Identities | ~170,000 Images | Computing FAR, FRR, ROC, and optimal $\tau$ |

#### 2.3 Preprocessing & Facial Alignment Pipeline
- Multi-Task Cascaded Convolutional Network (MTCNN) or OpenCV Haar cascade for facial landmark localization (left eye, right eye, nose tip, left mouth corner, right mouth corner).
- Similarity transformation (affine rotation and scaling) aligning eye centers horizontally.
- Cropping and normalization to standard $112 \times 112$ pixels.
- Normalization: Per-channel pixel scaling:
  $$x_{\text{norm}} = \frac{x - 127.5}{128.0} \in [-1.0, 1.0]$$

---

### 3. Directory Structure for Local Ingestion

```
ml/datasets/
├── forest_persons/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
└── vggface2/
    ├── train/
    │   ├── n000001/
    │   └── ...
    └── test/
        ├── n008632/
        └── ...
```

---

### 4. Ethical Compliance & Academic Licensing
- All datasets are downloaded and utilized under educational, non-commercial research licensing.
- Raw facial images from VGGFace2 are utilized solely for offline metric learning. Under system operation, visitor images are processed ephemerally in RAM and immediately discarded following 128-D embedding extraction.
