import numpy as np
import pytest
from data_generators.sequence_prediction import SequencePredictionGenerator
from data_generators.arithmetic import ArithmeticGenerator
from data_generators.pattern_matching import PatternMatchingGenerator
from curriculum.trainer import CurriculumTrainer
from curriculum.feeling import FeelingTracker


def test_sequence_generator_shapes():
    gen = SequencePredictionGenerator()
    for level in (0, 1, 2):
        X, y = gen.generate(level, n_samples=10)
        assert X.shape == (10, gen.input_dim(level))
        assert y.shape == (10, gen.output_dim())


def test_arithmetic_generator_shapes():
    gen = ArithmeticGenerator()
    for level in (0, 1, 2):
        X, y = gen.generate(level, n_samples=10)
        assert X.shape[1] == 5
        assert y.shape[1] == 1


def test_pattern_generator_shapes():
    gen = PatternMatchingGenerator()
    for level in (0, 1, 2):
        X, y = gen.generate(level, n_samples=10)
        assert X.shape[1] == gen.input_dim(level)
        assert y.shape[1] == 1


def test_curriculum_trainer_runs():
    gen = SequencePredictionGenerator()
    trainer = CurriculumTrainer(gen, model_size="micro")
    model, feeling = trainer.train(seed=42)
    assert model is not None
    # All 3 levels recorded
    assert len(feeling.records) == 3
    for level in range(3):
        assert len(feeling.records[level]) >= 1


def test_curriculum_trainer_produces_valid_output():
    gen = SequencePredictionGenerator()
    trainer = CurriculumTrainer(gen, model_size="micro")
    model, feeling = trainer.train(seed=0)
    raw = np.zeros(64, dtype=np.float32)
    raw[:8] = [0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
    result = model.infer(raw)
    assert result.shape == (1,)
