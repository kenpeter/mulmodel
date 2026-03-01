import numpy as np
import pytest
from bank.model_bank import ModelBank
from tiny_model.model import TinyModel
from curriculum.feeling import FeelingTracker


def _dummy_model() -> TinyModel:
    return TinyModel(input_dim=8, output_dim=1)


def _dummy_emb() -> np.ndarray:
    return np.random.rand(64).astype(np.float32)


def test_register_and_get():
    bank = ModelBank()
    m = _dummy_model()
    emb = _dummy_emb()
    mid = bank.register(m, emb)
    assert mid == "model_0"
    assert bank.get_model(mid) is m
    assert bank.get_embedding(mid) is not None


def test_register_increments_id():
    bank = ModelBank()
    ids = [bank.register(_dummy_model(), _dummy_emb()) for _ in range(3)]
    assert ids == ["model_0", "model_1", "model_2"]


def test_register_with_feeling():
    bank = ModelBank()
    f = FeelingTracker()
    f.record(0, 0, 0.3, 0.2, 0.9)
    mid = bank.register(_dummy_model(), _dummy_emb(), feeling=f)
    retrieved = bank.get_feeling(mid)
    assert retrieved is f


def test_len_and_contains():
    bank = ModelBank()
    assert len(bank) == 0
    mid = bank.register(_dummy_model(), _dummy_emb())
    assert len(bank) == 1
    assert mid in bank


def test_all_model_ids():
    bank = ModelBank()
    for _ in range(4):
        bank.register(_dummy_model(), _dummy_emb())
    ids = bank.all_model_ids()
    assert len(ids) == 4
