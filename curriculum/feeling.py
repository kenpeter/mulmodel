from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


_ACC_THRESHOLDS = {0: 0.85, 1: 0.80, 2: 0.75}
_MAX_EPOCHS = {0: 20, 1: 30, 2: 50}
_MIN_EPOCHS = {0: 3, 1: 5, 2: 5}
_STAGNATION_DELTA = 0.005  # 0.5% improvement threshold


@dataclass
class LevelRecord:
    level: int
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float


class FeelingTracker:
    """
    Tracks learning progress per curriculum level.
    Gates advancement when accuracy is high enough or stagnation detected.

    get_feeling_vector() → shape (9,)
      [easy_loss, easy_acc, med_loss, med_acc, hard_loss, hard_acc,
       epochs_easy, epochs_med, epochs_hard]
    """

    def __init__(self) -> None:
        self.records: list[list[LevelRecord]] = [[], [], []]

    def record(
        self,
        level: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        self.records[level].append(
            LevelRecord(level, epoch, train_loss, val_loss, val_acc)
        )

    def is_ready_to_advance(self, level: int) -> bool:
        recs = self.records[level]
        if not recs:
            return False
        epochs_done = len(recs)
        last_acc = recs[-1].val_acc

        if last_acc >= _ACC_THRESHOLDS[level]:
            return True

        if epochs_done >= _MAX_EPOCHS[level]:
            return True

        if epochs_done >= _MIN_EPOCHS[level] and epochs_done >= 2:
            recent_accs = [r.val_acc for r in recs[-5:]]
            improvement = max(recent_accs) - min(recent_accs)
            if improvement < _STAGNATION_DELTA:
                return True

        return False

    def epochs_at_level(self, level: int) -> int:
        return len(self.records[level])

    def best_acc(self, level: int) -> float:
        if not self.records[level]:
            return 0.0
        return max(r.val_acc for r in self.records[level])

    def last_loss(self, level: int) -> float:
        if not self.records[level]:
            return 1.0
        return self.records[level][-1].val_loss

    def get_feeling_vector(self) -> np.ndarray:
        vec = np.zeros(9, dtype=np.float32)
        for level in range(3):
            vec[level * 2] = self.last_loss(level)
            vec[level * 2 + 1] = self.best_acc(level)
        vec[6] = self.epochs_at_level(0)
        vec[7] = self.epochs_at_level(1)
        vec[8] = self.epochs_at_level(2)
        return vec
