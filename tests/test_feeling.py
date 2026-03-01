import numpy as np
import pytest
from curriculum.feeling import FeelingTracker


def test_initial_state():
    f = FeelingTracker()
    assert f.epochs_at_level(0) == 0
    assert f.best_acc(0) == 0.0
    assert f.last_loss(0) == 1.0


def test_record_and_advance_on_high_acc():
    f = FeelingTracker()
    f.record(0, 0, 0.5, 0.4, 0.90)  # acc=0.90 > threshold 0.85
    assert f.is_ready_to_advance(0)


def test_not_ready_low_acc_few_epochs():
    # Only 2 epochs (below MIN_EPOCHS=3), no acc threshold met → not ready
    f = FeelingTracker()
    for ep in range(2):
        f.record(0, ep, 0.5, 0.4, 0.50)
    assert not f.is_ready_to_advance(0)


def test_advance_on_max_epochs():
    f = FeelingTracker()
    for ep in range(20):  # MAX_EPOCHS[0] = 20
        f.record(0, ep, 0.5, 0.4, 0.50)
    assert f.is_ready_to_advance(0)


def test_feeling_vector_shape():
    f = FeelingTracker()
    f.record(0, 0, 0.3, 0.2, 0.7)
    f.record(1, 0, 0.2, 0.15, 0.8)
    f.record(2, 0, 0.1, 0.08, 0.78)
    vec = f.get_feeling_vector()
    assert vec.shape == (9,)
    assert vec.dtype == np.float32


def test_stagnation_advance():
    f = FeelingTracker()
    # Record 5+ epochs with tiny improvement → stagnation
    for ep in range(8):
        f.record(0, ep, 0.5, 0.4, 0.600 + ep * 0.0001)
    assert f.is_ready_to_advance(0)
