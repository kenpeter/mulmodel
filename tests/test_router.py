import numpy as np
import pytest
from core.problem import Problem
from router.router import Router
from router.model_index import ModelIndex
from router.encoder import ProblemEncoder, compute_specialty_embedding
from tiny_model.model import TinyModel


def _make_problem(support_len: int = 3) -> Problem:
    sx = np.random.rand(support_len, 8).astype(np.float32)
    sy = np.random.rand(support_len, 1).astype(np.float32)
    raw = np.zeros(64, dtype=np.float32)
    raw[:8] = sx[0]
    return Problem(raw_input=raw, support_X=sx, support_y=sy)


def test_empty_bank_returns_inf():
    idx = ModelIndex()
    router = Router(idx)
    p = _make_problem()
    mid, model, loss = router.try_existing(p)
    assert model is None
    assert loss == float("inf")


def test_register_and_retrieve():
    idx = ModelIndex()
    router = Router(idx)
    model = TinyModel(input_dim=8, output_dim=1)
    emb = np.random.rand(64).astype(np.float32)
    router.register_model("m0", model, emb)
    p = _make_problem()
    mid, best, loss = router.try_existing(p)
    assert best is not None
    assert isinstance(loss, float)


def test_needs_new_model():
    idx = ModelIndex()
    router = Router(idx)
    assert router.needs_new_model(0.10)
    assert not router.needs_new_model(0.01)


def test_encoder_output_shape():
    enc = ProblemEncoder()
    p = _make_problem()
    emb = enc.encode(p)
    assert emb.shape == (64,)
    assert emb.dtype == np.float32


def test_specialty_embedding_shape():
    sx = np.random.rand(10, 8).astype(np.float32)
    sy = np.random.rand(10, 1).astype(np.float32)
    emb = compute_specialty_embedding(sx, sy)
    assert emb.shape == (64,)
