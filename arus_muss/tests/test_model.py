import shutil
import uuid

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import arus

from ..har_model import MUSSHARModel


class TestMUSSHARModel:
    @pytest.mark.parametrize('placements', [['DW'], ['DW', 'DA']])
    @pytest.mark.parametrize('task_name', ['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])
    def test_mussharmodel(self, placements, task_name):
        spades_lab = arus.ds.MHDataset(
            path=arus.ds.get_dataset_path('spades_lab'),
            name='spades_lab', input_type=arus.ds.InputType.MHEALTH_FORMAT)
        spades_1_ds = spades_lab.subset(
            name='spades_lab_spades_1', pids=['SPADES_1'])

        spades_1_ds.set_class_set_parser(arus.slab.class_set)
        spades_1_ds.set_placement_parser(arus.slab.get_sensor_placement)

        model = MUSSHARModel(
            mid=str(uuid.uuid4()), used_placements=placements, window_size=12.8, sr=80)

        model.load_dataset(spades_1_ds)

        model.compute_features()
        model.compute_class_set(
            task_names=['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])

        model.train(task_name=task_name, verbose=True)

        assert type(model.model) == Pipeline
        assert model.train_perf['acc'] > 0.9

        shutil.rmtree(model.get_processed_path(), ignore_errors=True)
