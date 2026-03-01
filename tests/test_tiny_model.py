import numpy as np
import pytest
from tiny_model.model import TinyModel


def test_forward_shape():
    model = TinyModel(input_dim=8, output_dim=1)
    import torch
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 1)


def test_infer_returns_numpy():
    model = TinyModel(input_dim=8, output_dim=1)
    raw = np.random.randn(64).astype(np.float32)
    result = model.infer(raw)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)


def test_predict_batch():
    model = TinyModel(input_dim=5, output_dim=1)
    X = np.random.randn(10, 5).astype(np.float32)
    preds = model.predict_batch(X)
    assert preds.shape == (10, 1)


def test_param_count_micro():
    model = TinyModel(input_dim=8, output_dim=1, size="micro")
    assert model.param_count() < 50_000


def test_sizes():
    for size in ("micro", "small", "medium"):
        m = TinyModel(8, 1, size=size)
        assert m.param_count() > 0
