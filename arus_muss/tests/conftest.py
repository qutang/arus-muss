import pandas as pd
import pytest
import arus


@pytest.fixture(scope="module")
def spades_lab_ds():
    return arus.ds.MHDataset(path=arus.ds.get_dataset_path('spades_lab'),
                             name='spades_lab', input_type=arus.ds.InputType.MHEALTH_FORMAT)


@pytest.fixture(scope="module")
def dw_data(spades_lab_ds):
    sensor = spades_lab_ds.get_sensors(pid='SPADES_1', placement='DW')[0]
    sensor_file = sensor.paths[0]
    data = pd.read_csv(sensor_file, parse_dates=[0]).iloc[1:1000, :]
    st = data.iloc[0, 0]
    et = data.iloc[-1, 0]
    sr = 80
    return data, st, et, sr


@pytest.fixture(scope="module")
def da_data(spades_lab_ds):
    sensor = spades_lab_ds.get_sensors(pid='SPADES_1', placement='DA')[0]
    sensor_file = sensor.paths[0]
    data = pd.read_csv(sensor_file, parse_dates=[0]).iloc[1:1000, :]
    st = data.iloc[0, 0]
    et = data.iloc[-1, 0]
    sr = 80
    return data, st, et, sr
