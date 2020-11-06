import arus

FEATURE_NAMES = [
    'MEAN',
    'STD',
    'MAX',
    'RANGE',
    'DOM_FREQ',
    'FREQ_POWER_RATIO_ABOVE_3DOT5',
    'DOM_FREQ_POWER_RATIO',
    'ACTIVE_SAMPLES',
    'ACTIVATIONS',
    'STD_ACTIVATION_DURATIONS',
    'MEDIAN_G_ANGLE',
    'RANGE_G_ANGLE'
]


def compute(*raw_dfs, st, et, srs, placements):
    fv_dfs = []
    st, et = arus.ext.pandas.get_common_timespan(*raw_dfs, st=st, et=et)
    for raw_df, sr in zip(raw_dfs, srs):
        fv_df, feature_names = arus.fv.inertial.single_triaxial(
            raw_df, sr, st, et, subwin_secs=2,
            ori_unit='rad', activation_threshold=0.2, use_vm=True, selected=FEATURE_NAMES)
        fv_dfs.append(fv_df)
    fv_df, _ = arus.ext.pandas.merge_all(
        *fv_dfs,
        suffix_names=placements,
        suffix_cols=feature_names,
        on=arus.mh.FEATURE_SET_TIMESTAMP_COLS,
        how='inner',
        sort=False
    )
    return fv_df
