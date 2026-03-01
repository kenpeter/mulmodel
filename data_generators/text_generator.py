from __future__ import annotations
import numpy as np
from curriculum.generator import CurriculumGenerator
from data_generators.binary_tokenizer import BinaryTokenizer
from data_generators.corpus_loader import get_corpus

_tokenizer = BinaryTokenizer()

# Noise added to labels at hard level
_LEVEL_LABEL_NOISE = {0: 0.0, 1: 0.05, 2: 0.15}
# Fraction of "hard" (close-to-boundary) examples per level
_LEVEL_HARD_FRAC   = {0: 0.0, 1: 0.10, 2: 0.25}


class TextDataGenerator(CurriculumGenerator):
    """
    Generates curriculum text training data using a REAL corpus.

    When a text problem arrives (e.g. sentiment, topic), this generator:
      1. Loads a real corpus (UCI sentiment dataset — ~2748 reviews from
         Amazon / IMDB / Yelp, downloaded once and cached locally)
      2. Splits corpus into positive and negative pools
      3. Samples from those pools to build (X, y) training batches
      4. Tokenizes each text via BinaryTokenizer → 64-dim float vector

    Curriculum levels:
      Level 0 (easy):   sample only high-confidence examples (clear signal)
      Level 1 (medium): full corpus, balanced sampling
      Level 2 (hard):   full corpus + some ambiguous/noisy examples

    Why real data beats templates:
      Templates produce "great movie", "terrible film" — simple but repetitive.
      The corpus has 2748 naturally varied sentences the TinyModel trains on,
      giving it exposure to real vocabulary bit-patterns.
    """

    def __init__(
        self,
        support_texts:  list[str],
        support_labels: list[float],
    ) -> None:
        self.support_texts  = support_texts
        self.support_labels = support_labels
        self._in_dim        = BinaryTokenizer.EMB_DIM

        # Load and split corpus into positive / negative pools
        corpus = get_corpus()
        self._pos_pool = [text for text, lbl in corpus if lbl > 0]
        self._neg_pool = [text for text, lbl in corpus if lbl <= 0]

        # If corpus is tiny / all one class, supplement with support texts
        if len(self._pos_pool) < 5:
            self._pos_pool += [t for t, l in zip(support_texts, support_labels) if l > 0]
        if len(self._neg_pool) < 5:
            self._neg_pool += [t for t, l in zip(support_texts, support_labels) if l <= 0]

    def input_dim(self, level: int) -> int:
        return self._in_dim

    def output_dim(self) -> int:
        return 1

    def generate(
        self,
        level:     int,
        n_samples: int = 500,
        seed:      int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng      = np.random.default_rng(seed)
        X_list:  list[np.ndarray] = []
        y_list:  list[float]      = []

        n_hard  = int(n_samples * _LEVEL_HARD_FRAC[level])
        n_clean = n_samples - n_hard

        # ── clean examples ────────────────────────────────────────────────────
        for _ in range(n_clean):
            positive = rng.random() > 0.5
            pool     = self._pos_pool if positive else self._neg_pool
            label    = 1.0 if positive else -1.0

            if level == 0:
                # Easy: only short, simple sentences (< 8 words)
                short = [t for t in pool if len(t.split()) <= 8]
                pool  = short if len(short) >= 5 else pool

            text = str(rng.choice(pool))
            X_list.append(_tokenizer.tokenize(text))
            y_list.append(label)

        # ── hard examples (ambiguous — label noise) ───────────────────────────
        for _ in range(n_hard):
            positive = rng.random() > 0.5
            pool     = self._pos_pool if positive else self._neg_pool
            label    = 1.0 if positive else -1.0
            # Add label noise: occasionally flip
            if rng.random() < _LEVEL_LABEL_NOISE[level]:
                label = -label
            text = str(rng.choice(pool))
            X_list.append(_tokenizer.tokenize(text))
            y_list.append(label)

        X = np.stack(X_list).astype(np.float32)
        y = np.array([[l] for l in y_list], dtype=np.float32)
        return X, y
