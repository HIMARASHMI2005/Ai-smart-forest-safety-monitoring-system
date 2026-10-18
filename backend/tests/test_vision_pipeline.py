import torch
import numpy as np
import pytest
from ml.forest_person_net.architecture import ForestPersonNet
from ml.forest_face_net.architecture import ForestFaceNet
from ml.forest_face_net.margin_loss import ArcFaceLoss
from ml.forest_face_net.verify import compute_cosine_similarity

def test_forest_person_net_forward():
    # Model forward pass on dummy image batch
    model = ForestPersonNet(num_classes=1)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        outputs = model(dummy_input)

    assert "p3" in outputs
    assert "p4" in outputs
    assert "p5" in outputs

    # Check output shape: (1, 18, GridH, GridW) where 18 = 3 anchors * (5 + 1)
    assert outputs["p3"].shape == (1, 18, 80, 80)
    assert outputs["p4"].shape == (1, 18, 40, 40)
    assert outputs["p5"].shape == (1, 18, 20, 20)

def test_forest_face_net_forward_and_norm():
    # Model forward pass on 112x112 crop
    model = ForestFaceNet(embedding_dim=128)
    model.eval()
    with torch.no_grad():
        dummy_face = torch.randn(2, 3, 112, 112)
        embeddings = model(dummy_face)

    assert embeddings.shape == (2, 128)

    # Check L2 Unit Norm: ||x||_2 == 1.0
    norms = torch.norm(embeddings, p=2, dim=1)
    for n in norms:
        assert torch.isclose(n, torch.tensor(1.0), atol=1e-5)

def test_cosine_similarity_computation():
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert pytest.approx(compute_cosine_similarity(vec1, vec2), 0.001) == 1.0

    vec3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert pytest.approx(compute_cosine_similarity(vec1, vec3), 0.001) == 0.0

def test_arcface_loss():
    loss_fn = ArcFaceLoss(in_features=128, out_features=10, s=64.0, m=0.5)
    dummy_emb = torch.randn(4, 128)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(dummy_emb, dummy_labels)

    assert loss.item() > 0.0
    loss.backward()
    assert loss_fn.weight.grad is not None
