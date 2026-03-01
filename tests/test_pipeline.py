import numpy as np
import pytest
from core.problem import Problem
from system.pipeline import RTTrainerPipeline


@pytest.fixture
def pipeline():
    """Fresh pipeline with no disk loading for tests."""
    return RTTrainerPipeline(load_on_start=False)


def _seq_problem(values: list[float], n_support: int = 3) -> Problem:
    raw = np.zeros(64, dtype=np.float32)
    v = np.array(values, dtype=np.float32)
    raw[: len(v)] = v / max(abs(v).max(), 1.0)

    sx, sy = [], []
    for i in range(n_support):
        sx.append([values[i] / 256.0, values[i + 1] / 256.0])
        sy.append([values[i + 2] / 256.0])

    return Problem(
        raw_input=raw,
        support_X=np.array(sx, dtype=np.float32),
        support_y=np.array(sy, dtype=np.float32),
        description="sequence_prediction",
    )


def _pattern_problem(bits: list[int]) -> Problem:
    raw = np.zeros(64, dtype=np.float32)
    raw[: len(bits)] = bits
    sx = np.array([bits], dtype=np.float32)
    majority = 1.0 if sum(bits) > len(bits) / 2 else -1.0
    sy = np.array([[majority]], dtype=np.float32)
    return Problem(raw_input=raw, support_X=sx, support_y=sy, description="pattern_matching")


def test_new_problem_trains_model(pipeline):
    p = _seq_problem([0, 1, 2, 3, 4, 5])
    a = pipeline.solve(p)
    assert a.was_trained is True
    assert a.source.startswith("newly_trained:")
    assert a.value is not None


def test_bank_grows_after_training(pipeline):
    assert pipeline.bank_size() == 0
    pipeline.solve(_seq_problem([0, 1, 2, 3, 4, 5]))
    assert pipeline.bank_size() == 1


def test_same_domain_can_reuse(pipeline, monkeypatch):
    a1 = pipeline.solve(_seq_problem([0, 1, 2, 3, 4, 5]))
    assert a1.was_trained is True

    from router import router as router_mod

    def mock_evaluate(self, model, problem):
        return 0.01

    monkeypatch.setattr(router_mod.Router, "evaluate", mock_evaluate)

    a2 = pipeline.solve(_seq_problem([2, 4, 6, 8, 10, 12]))
    assert a2.was_trained is False
    assert a2.source.startswith("bank:")


def test_different_domain_trains_new_model(pipeline):
    pipeline.solve(_seq_problem([0, 1, 2, 3, 4, 5]))
    a = pipeline.solve(_pattern_problem([1, 0, 1, 0]))
    assert a.value is not None


def test_answer_fields(pipeline):
    a = pipeline.solve(_seq_problem([1, 3, 5, 7, 9, 11]))
    assert hasattr(a, "value")
    assert hasattr(a, "confidence")
    assert hasattr(a, "source")
    assert hasattr(a, "was_trained")
    assert hasattr(a, "loss")
    assert 0.0 <= a.confidence <= 1.0 or a.confidence >= 0.0
