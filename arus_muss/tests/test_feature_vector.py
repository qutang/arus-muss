import arus
import numpy as np
import pandas as pd
import pytest

from ..feature_vector import compute as compute_fv
from ..feature_vector import FEATURE_NAMES


def test_feature_vector(dw_data, da_data):

    dw_df, _, _, dw_sr = dw_data
    da_df, _, _, da_sr = da_data

    raw_dfs = [dw_df, da_df]
    placements = ['DW', 'DA']
    srs = [dw_sr, da_sr]

    fv_df = compute_fv(
        *raw_dfs, st=None, et=None, srs=srs, placements=placements)

    fv_names = fv_df.columns[3:].values.tolist()

    feature_names = arus.fv.inertial.assemble_fv_names(
        selected=FEATURE_NAMES, use_vm=True, num_of_axes=3)

    true_fv_names = []
    for p in placements:
        true_fv_names += list(map(lambda name: f'{name}_{p}', feature_names))

    np.testing.assert_array_equal(fv_names, true_fv_names)

    assert fv_df.shape[0] == 1
    assert np.sum(fv_df.notna().values) == fv_df.shape[1]
