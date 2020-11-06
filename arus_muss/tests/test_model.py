import shutil
import uuid

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import arus

from ..har_model import MUSSHARModel
from ..import cli


class TestMUSSHARModel:
    @pytest.mark.parametrize('placements', [['DW'], ['DW', 'DA']])
    @pytest.mark.parametrize('task_name', ['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])
    def test_train_mussharmodel(self, placements, task_name):
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

        model.compute_features(pids=['SPADES_1'], n_cores=1)
        model.compute_class_set(
            task_names=['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])

        model.train(task_name=task_name, verbose=True)

        assert type(model.model) == Pipeline
        assert model.train_perf['acc'] > 0.9

        shutil.rmtree(model.get_processed_path(), ignore_errors=True)

    @pytest.mark.parametrize('task_name', ['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])
    def test_predict_mussharmodel(self, task_name):
        model_path = cli._get_model_path(
            placements=['NDW'], pids='ALL', target=task_name, model_path=None)
        predict_df = cli.predict_on_files(model_path,
                                          test_files=['test.signaligner.csv'], placements=[
                                              'NDW'], srs=[80], file_format='SIGNALIGNER')
        assert predict_df.shape[0] == 136
