import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask, GridPair


class TestGridPair:
    def test_has_output_true(self):
        pair = GridPair(
            input=Grid(np.array([[0, 1]])),
            output=Grid(np.array([[1, 0]]))
        )
        assert pair.has_output() is True

    def test_has_output_false(self):
        pair = GridPair(input=Grid(np.array([[0, 1]])))
        assert pair.has_output() is False


class TestArcTask:
    @pytest.fixture
    def sample_task(self):
        return ArcTask(
            task_id="test_001",
            source="synthetic",
            rule_type="reflect_horizontal",
            rule_params={},
            difficulty=1,
            train_pairs=[
                GridPair(
                    input=Grid(np.array([[1, 2], [3, 4]])),
                    output=Grid(np.array([[3, 4], [1, 2]]))
                ),
                GridPair(
                    input=Grid(np.array([[5, 6], [7, 8]])),
                    output=Grid(np.array([[7, 8], [5, 6]]))
                ),
            ],
            test_pairs=[
                GridPair(
                    input=Grid(np.array([[0, 1], [2, 3]])),
                    output=Grid(np.array([[2, 3], [0, 1]]))
                )
            ]
        )

    def test_task_properties(self, sample_task):
        assert sample_task.n_train == 2
        assert sample_task.n_test == 1
        assert sample_task.source == "synthetic"
        assert sample_task.rule_type == "reflect_horizontal"

    def test_content_hash_deterministic(self, sample_task):
        # Hash should be the same for same content
        h1 = sample_task.content_hash
        h2 = sample_task.content_hash
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_differs_for_different_tasks(self):
        t1 = ArcTask(
            task_id="t1",
            train_pairs=[GridPair(input=Grid(np.array([[0, 1]])), output=Grid(np.array([[1, 0]])))],
            test_pairs=[GridPair(input=Grid(np.array([[2, 3]])))])
        t2 = ArcTask(
            task_id="t2",
            train_pairs=[GridPair(input=Grid(np.array([[9, 8]])), output=Grid(np.array([[8, 9]])))],
            test_pairs=[GridPair(input=Grid(np.array([[7, 6]])))])
        assert t1.content_hash != t2.content_hash

    def test_to_dict_and_from_dict(self, sample_task):
        d = sample_task.to_dict()
        restored = ArcTask.from_dict(d)

        assert restored.task_id == sample_task.task_id
        assert restored.source == sample_task.source
        assert restored.rule_type == sample_task.rule_type
        assert restored.n_train == sample_task.n_train
        assert restored.n_test == sample_task.n_test

        for orig, rest in zip(sample_task.train_pairs, restored.train_pairs):
            assert orig.input == rest.input
            assert orig.output == rest.output

    def test_save_and_load(self, sample_task, tmp_path):
        path = tmp_path / "test_task.json"
        sample_task.save(path)

        loaded = ArcTask.load(path)
        assert loaded.task_id == sample_task.task_id
        assert loaded.n_train == sample_task.n_train
        assert loaded.content_hash == sample_task.content_hash

        for orig, loaded_pair in zip(sample_task.train_pairs, loaded.train_pairs):
            assert orig.input == loaded_pair.input
            assert orig.output == loaded_pair.output

    def test_json_format_matches_arc(self, sample_task):
        d = sample_task.to_dict()
        assert "train" in d
        assert "test" in d
        assert isinstance(d["train"], list)
        assert "input" in d["train"][0]
        assert "output" in d["train"][0]

    def test_task_with_no_test_output(self):
        task = ArcTask(
            task_id="no_output",
            train_pairs=[
                GridPair(
                    input=Grid(np.array([[1]])),
                    output=Grid(np.array([[2]]))
                )
            ],
            test_pairs=[
                GridPair(input=Grid(np.array([[3]])))
            ]
        )
        d = task.to_dict()
        assert d["test"][0]["output"] is None

        restored = ArcTask.from_dict(d)
        assert restored.test_pairs[0].output is None
