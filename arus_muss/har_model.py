
import gc
import os
import typing
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GroupShuffleSplit, TimeSeriesSplit,
                                     cross_validate, learning_curve)
from sklearn.utils import shuffle

import arus

from .feature_vector import compute as compute_fv
from .classifier import estimator


@dataclass
class MUSSHARModel(arus.models.HARModel):
    name: str = "MUSS_HAR"
    window_size: float = 12.8
    step_size: float = None
    sr: int = 80
    used_placements: typing.List[str] = field(default_factory=lambda: ['DW'])

    def reset_model(self):
        self.model = None
        self.scaler = None
        self.used_placements = ['DW']

    def reset_processed_data(self):
        self.data_set.clear_processed()

    def load_dataset(self, data_set: arus.ds.MHDataset):
        for subj in data_set.subjects:
            # Exclude all other data types
            sensors = []
            for p in self.used_placements:
                sensors += data_set.get_sensors(
                    subj.pid, data_type='AccelerationCalibrated', placement=p)
            subj.sensors = sensors
        self.data_set = data_set

    def compute_features(self, pids=None):
        if pids is None:
            pids = self.data_set.get_pids()
        fss = []
        sch = arus.Scheduler(mode=arus.Scheduler.Mode.PROCESS,
                             scheme=arus.Scheduler.Scheme.SUBMIT_ORDER)
        sch.reset()
        for pid in pids:
            subj = self.data_set.get_subject_obj(pid)

            logger.info(f'Computing features for {pid}')
            start_time, stop_time = self.data_set.get_session_span(pid)

            sch.submit(
                self._compute_features_per_subj,
                subject=subj,
                input_type=self.data_set.input_type,
                st=start_time,
                et=stop_time
            )
            gc.collect()

        results = sch.get_all_remaining_results()

        for subj_fs, pid in zip(results, pids):
            subj = self.data_set.get_subject_obj(pid)
            if subj_fs is not None:
                subj_fs_names = subj_fs.columns[3:]
                subj.processed = {**subj.processed,
                                  'fs': subj_fs, 'fs_names': subj_fs_names}
                fs = subj_fs.copy()
                fs['PID'] = pid
                fss.append(fs)
                fs_names = subj_fs_names
            else:
                logger.warning(
                    f'Subject {pid} failed to compute features, this may due to the sensor data is out of range, incorrect sensor data, or invalid sensor data values. Ignore it from the feature set.')
        if len(fss) > 0:
            fs = pd.concat(fss, axis=0, ignore_index=True, sort=False)
            fs_names = subj_fs_names
            self.data_set.processed = {
                **self.data_set.processed, 'fs': fs, 'fs_names': fs_names}
        else:
            logger.warning('No feature set is computed.')

    def compute_class_set(self, task_names, pids=None):
        if pids is None:
            pids = self.data_set.get_pids()
        css = []
        for pid in pids:
            subj = self.data_set.get_subject_obj(pid)
            logger.info(f'Compute class set for {pid}')
            start_time, stop_time = self.data_set.get_session_span(pid)

            subj_class_set = self.data_set.get_class_set(
                subj.pid, task_names=task_names, window_size=self.window_size, step_size=self.step_size, start_time=start_time, stop_time=stop_time)

            if subj_class_set is not None:
                subj.processed = {**subj.processed,
                                  'cs': subj_class_set, 'cs_names': task_names}
                cs = subj_class_set.copy()
                cs['PID'] = pid
                css.append(cs)
            else:
                logger.warning(
                    f'Subject {pid} failed to compute class set, this may due to the annotation data is out of range, incorrect annotation data, or invalid annotation data values. Ignore it from the class set.')
        if len(css) > 0:
            cs = pd.concat(css, axis=0, ignore_index=True, sort=False)
            self.data_set.processed = {
                **self.data_set.processed, 'cs': cs, 'cs_names': task_names}
        else:
            logger.warning('No class set is computed.')

    def get_feature_names(self):
        if 'fs_names' in self.data_set.processed:
            fs_names = self.data_set.processed['fs_names']
        else:
            logger.warning(
                'No feature set available, please run compute_features at first.')
            fs_names = None
        return fs_names

    def get_processed_path(self):
        output_path = os.path.join(
            self.data_set.path, arus.mh.PROCESSED_FOLDER, self.mid)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def get_training_acc(self):
        return self.train_perf['acc']

    def train(self, task_name, ignore_classes=["Unknown", "Transition"], pids=None, verbose=False, **kwargs):
        if pids is None:
            pid_info = 'all participants'
        else:
            pid_info = ','.join(pids)
        logger.info(f'Train MUSS model of {task_name} for {pid_info}')
        fcs, fs_names, cs_names = self._preprocess_feature_class_set(pids)
        if task_name not in cs_names:
            logger.error(
                f'{task_name} is not in the class set, aborting training.')
            return

        fcs = MUSSHARModel.ignore_classes(
            fcs, task_name, remove_classes=ignore_classes)

        X = fcs.loc[:, fs_names].values
        y = fcs.loc[:, task_name].values

        self.model, train_acc = self._train_classifier(
            X, y, verbose=verbose, **kwargs)
        self.train_perf = {**self.train_perf, 'acc': train_acc}
        logger.info(f'Training accuracy: {train_acc}')
        self.train_pids = pids
        self.train_target = task_name

    def save_fcs(self, pids=[]):
        fcs, fs_names, cs_names = self._preprocess_feature_class_set(pids)
        if fcs is not None:
            output_path = os.path.join(
                self.get_processed_path(), 'fcs.csv')
            fcs.to_csv(output_path, float_format='%.6f')
            logger.info(f'Saved feature and class set to {output_path}')
        else:
            logger.warning('No available feature and class set to save!')

    def get_model_filename(self):
        file_name = MUSSHARModel.build_model_filename(
            name=self.name,
            placements=self.used_placements,
            pids=self.train_pids,
            target=self.train_target,
            dataset_name=self.data_set.name.upper())
        return file_name

    def save_model(self, save_raw=True, save_fcs=True, output_folder=None):
        output_filepath = os.path.join(
            output_folder or self.get_processed_path(),
            self.get_model_filename())
        logger.info(f'Save trained MUSS model to {output_filepath}')
        bundle = {
            'mid': self.mid,
            'placements': self.used_placements,
            'model': self.model,
            'dataset_name': self.data_set.name,
            'dataset': self.data_set if save_raw else None,
            'fs': None if not save_fcs else self.data_set.processed['fs'],
            'cs': None if not save_fcs else self.data_set.processed['cs'],
            'fs_names': None if not save_fcs else self.data_set.processed['fs_names'],
            'cs_names': None if not save_fcs else self.data_set.processed['cs_names']
        }
        joblib.dump(bundle, filename=output_filepath)

    @staticmethod
    def build_model_filename(name, placements, pids, target, dataset_name):
        if pids is None:
            pid_info = 'ALL'
        else:
            pid_info = '_'.join(pids)
        return f'{name}-{target}-{pid_info}-{"_".join(placements)}-{dataset_name}.har'

    @staticmethod
    def load_model(path):
        bundle = joblib.load(path)
        model = MUSSHARModel(
            mid=bundle['mid'], used_placements=bundle['placements'])
        model.model = bundle['model']
        model.data_set = bundle['dataset']
        if hasattr(model.data_set, "processed"):
            model.data_set.processed['fs'] = bundle['fs']
            model.data_set.processed['cs'] = bundle['cs']
            model.data_set.processed['fs_names'] = bundle['fs_names']
            model.data_set.processed['cs_names'] = bundle['cs_names']
        return model

    def predict(self, *input_objs, **kwargs):
        result = None
        if type(input_objs[0]) is arus.ds.SensorObj:
            result = self._predict_sensor(*input_objs)
        elif type(input_objs[0]) is arus.ds.SubjectObj:
            result = self._predict_subj(*input_objs)
        elif type(input_objs[0]) is pd.DataFrame and len(input_objs) == 1 and 'fs_names' in kwargs:
            result = self._predict_fs(input_objs[0], kwargs['fs_names'])
        elif type(input_objs[0]) is pd.DataFrame and 'placements' in kwargs and 'srs' in kwargs:
            result = self._predict_raw_dfs(*input_objs, **kwargs)
        elif type(input_objs[0]) is str and 'placements' in kwargs:
            result = self._predict_raw_files(input_objs, **kwargs)
        else:
            raise ValueError(
                f'Unrecognized input type {type(input_objs[0])} for predict function.')
        return result

    def cross_validation(self, task_name, pids=None, n_fold=5, **kwargs):
        if pids is None:
            pids = self.data_set.get_pids()
        fcs, fs_names, fc_names = self._preprocess_feature_class_set(pids=pids)
        fcs = self.ignore_classes(fcs, task_name=task_name, remove_classes=[
                                  'Unknown', 'Transition', 'Transition-Left', 'Transition-Right', 'Transition-Both', 'Unknown-Left', 'Unknown-Right', 'Unknown-Both'])
        fcs.reset_index(drop=True, inplace=True)
        splitter = TimeSeriesEpisodeSplit(
            fcs, fs_names, task_name, n_split=n_fold)
        clf = estimator(**kwargs)
        label_names = fcs.loc[:, task_name].unique().tolist()

        logger.info(f"Run {n_fold} cross validation")
        scores = cross_validate(clf, fcs.loc[:, fs_names].values,
                                y=fcs.loc[:, task_name].values, cv=splitter.split(), scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'], n_jobs=4, return_train_score=True)

        splitter = TimeSeriesEpisodeSplit(
            fcs, fs_names, task_name, n_split=n_fold)
        clf = estimator(**kwargs)
        tests, preds = ext.sklearn.cross_val_predict(clf,
                                                     fcs.loc[:,
                                                             fs_names].values,
                                                     y=fcs.loc[:, task_name].values, cv=splitter.split(), n_jobs=4)
        cm_df = self.confusion_matrix(tests, preds, label_names=label_names)
        cv_df = pd.DataFrame.from_dict(scores)
        self.data_set.processed = {
            **self.data_set.processed,
            'cv': cv_df,
            'cv_cm': cm_df
        }
        return cv_df, cm_df

    def logo_validation(self, task_name, pids=None, group_col=None, random_state=None, **kwargs):
        if pids is None:
            pids = self.data_set.get_pids()
        fcs, fs_names, fc_names = self._preprocess_feature_class_set(pids=pids)
        fcs = self.ignore_classes(fcs, task_name=task_name, remove_classes=[
                                  'Unknown', 'Transition', 'Transition-Left', 'Transition-Right', 'Transition-Both', 'Unknown-Left', 'Unknown-Right', 'Unknown-Both'])
        fcs.reset_index(drop=True, inplace=True)

        n_splits = len(fcs[group_col].unique())
        logo = GroupShuffleSplit(
            n_splits=n_splits, test_size=1, random_state=random_state)

        clf = estimator(**kwargs)
        label_names = fcs.loc[:, task_name].unique().tolist()

        logger.info(f"Run leave one group out validation on {group_col}")
        X = fcs.loc[:, fs_names].values
        y = fcs.loc[:, task_name].values
        groups = fcs.loc[:, group_col].values

        scores = cross_validate(clf, X,
                                y=y, groups=groups, cv=logo, scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'], n_jobs=4, return_train_score=True)

        clf = estimator(**kwargs)
        tests, preds = arus.ext.sklearn.cross_val_predict(clf,
                                                          X,
                                                          y=y, groups=groups, cv=logo, n_jobs=4)
        cm_df = self.confusion_matrix(tests, preds, label_names=label_names)
        logo_df = pd.DataFrame.from_dict(scores)
        self.data_set.processed = {
            **self.data_set.processed,
            'logo': logo_df,
            'logo_cm': cm_df
        }
        return

    def learning_curve_logo(self, task_name, pids=None, group_col=None, random_state=None, n_steps=10, **kwargs):
        if pids is None:
            pids = self.data_set.get_pids()
        fcs, fs_names, fc_names = self._preprocess_feature_class_set(pids=pids)
        fcs = self.ignore_classes(fcs, task_name=task_name, remove_classes=[
                                  'Unknown', 'Transition', 'Transition-Left', 'Transition-Right', 'Transition-Both', 'Unknown-Left', 'Unknown-Right', 'Unknown-Both'])
        fcs.reset_index(drop=True, inplace=True)

        n_splits = len(fcs[group_col].unique())
        logo = GroupShuffleSplit(
            n_splits=n_splits, test_size=1, random_state=random_state)

        clf = estimator(**kwargs)
        label_names = fcs.loc[:, task_name].unique().tolist()

        logger.info(
            f"Compute learning curve using leave one group out validation on {group_col}")
        X = fcs.loc[:, fs_names].values
        y = fcs.loc[:, task_name].values
        groups = fcs.loc[:, group_col].values
        train_sizes = np.linspace(0.1, 1, num=n_steps, endpoint=True)

        train_sizes_abs, train_scores, test_scores, fit_times, score_times = learning_curve(
            clf, X, y=y, groups=groups, train_sizes=train_sizes, cv=logo, scoring='f1_macro', n_jobs=4, shuffle=True, verbose=1, random_state=random_state, return_times=True)

        train_scores_means = np.mean(train_scores, axis=1)
        train_scores_stds = np.std(train_scores, axis=1)
        test_scores_means = np.mean(test_scores, axis=1)
        test_scores_stds = np.std(test_scores, axis=1)
        fit_times_means = np.mean(fit_times, axis=1)
        fit_times_stds = np.std(fit_times, axis=1)

        result_dict = {
            'train_sizes': train_sizes_abs,
            'train_scores_means': train_scores_means,
            'train_scores_stds': train_scores_stds,
            'test_scores_means': test_scores_means,
            'test_scores_stds': test_scores_stds,
            'fit_times_means': fit_times_means,
            'fit_times_stds': fit_times_stds,
        }
        print(result_dict)
        lc_logo_df = pd.DataFrame.from_dict(result_dict)
        self.data_set.processed = {
            **self.data_set.processed,
            'logo_lc': lc_logo_df
        }

    def _predict_subj(self, *subj_objs: typing.List[arus.ds.SubjectObj]):
        results = []
        for subj in subj_objs:
            if 'fs' in subj.processed:
                result = self._predict_fs(
                    subj.processed['fs'], subj.processed['fs_names'])
            else:
                sensors = []
                for p in self.used_placements:
                    sensor = subj.get_sensor('placement', p)
                    sensors.append(sensor)
                result = self._predict_sensor(*sensors)
            results.append(result)
        return results

    def _predict_sensor(self, *sensor_objs):
        raw_dfs = [sensor.get_data() for sensor in sensor_objs]
        placements = [sensor.placement for sensor in sensor_objs]
        srs = [sensor.sr for sensor in sensor_objs]
        return self._predict_raw_dfs(*raw_dfs, placements=placements, srs=srs)

    def _predict_raw_files(self, raw_files, placements, file_format, srs=None):
        if srs is None:
            srs = [None] * len(raw_files)
        sensors = []
        file_format = arus.ds.InputType[file_format]
        for p in self.used_placements:
            i = placements.index(p)
            sensors.append(arus.ds.SensorObj(paths=[raw_files[i]],
                                             sr=srs[i], placement=p, input_type=file_format))
        return self._predict_sensor(*sensors)

    def _predict_raw_dfs(self, *raw_dfs, placements, srs, **kwargs):
        start_time = min([raw_df.iloc[0, 0] for raw_df in raw_dfs])
        logger.info(f'The start time of test data is: {start_time}')
        logger.info('Compute features for test data')
        fs = self._compute_features_from_raw_dfs(
            raw_dfs, placements=placements, srs=srs, st=start_time, show_progress=True)
        fs_names = fs.columns[3:].values.tolist()
        logger.info('Make predictions for test data')
        predicts = self._predict_fs(fs, fs_names)
        return predicts

    def _predict_fs(self, fs, fs_names):
        fs = fs.dropna()
        predicts = self.model.predict(fs.loc[:, fs_names].values)
        predict_df = pd.DataFrame.from_dict({
            'HEADER_TIME_STAMP': fs['HEADER_TIME_STAMP'],
            'START_TIME': fs['START_TIME'],
            'STOP_TIME': fs['STOP_TIME'],
            'PREDICTION': predicts
        })
        return predict_df

    def _preprocess_feature_class_set(self, pids=None):
        if 'fs' in self.data_set.processed:
            if pids is not None:
                fss = []
                for pid in pids:
                    fs = self.data_set.get_subject_obj(pid).processed['fs']
                    fs['PID'] = pid
                    fss.append(fs)
                fs = pd.concat(fss, axis=0, sort=False, ignore_index=True)
                fs_names = self.data_set.get_subject_obj(
                    pids[0]).processed['fs_names']
            else:
                fs = self.data_set.processed['fs']
                fs_names = self.data_set.processed['fs_names']
        else:
            logger.error(
                'No feature set available, please run compute_features at first.')
            return
        if 'cs' in self.data_set.processed:
            if pids is not None:
                css = []
                for pid in pids:
                    cs = self.data_set.get_subject_obj(pid).processed['cs']
                    cs['PID'] = pid
                    css.append(cs)
                cs = pd.concat(css, axis=0, sort=False, ignore_index=True)
                cs_names = self.data_set.get_subject_obj(
                    pids[0]).processed['cs_names']
            else:
                cs = self.data_set.processed['cs']
                cs_names = self.data_set.processed['cs_names']
        else:
            logger.error(
                'No class set available, please run compute_class_set at first.')
            return

        if 'PID' in fs:
            on = arus.mh.FEATURE_SET_TIMESTAMP_COLS + ['PID']
        else:
            on = arus.mh.FEATURE_SET_TIMESTAMP_COLS
        fcs = pd.merge(fs, cs, how='inner',
                       on=on, sort=False)
        return fcs, fs_names, cs_names

    def _train_classifier(self, X, y, **kwargs):
        input_matrix, input_classes = shuffle(X, y)
        pipe = estimator(**kwargs)
        logger.debug(pipe)
        pipe.fit(input_matrix, input_classes)
        train_accuracy = pipe.score(input_matrix, input_classes)
        return pipe, train_accuracy

    def _import_data_per_subj(self, subj, input_type: arus.ds.InputType):
        sensor_dfs = []
        placements = []
        srs = []
        for sensor in subj.sensors:
            if input_type == arus.ds.InputType.MHEALTH_FORMAT:
                sensor.data = arus.mh.MhealthFileReader.read_csvs(
                    *sensor.paths)
            else:
                logger.error(
                    f'Unrecognized dataset input type: {input_type.name}')
            sensor_dfs.append(sensor.data)
            placements.append(sensor.placement)
            srs.append(sensor.sr or self.sr)
        return sensor_dfs, placements, srs

    def _compute_features_from_raw_dfs(self, sensor_dfs, placements, srs, st=None, et=None, show_progress=False):
        fs_df = arus.ext.pandas.fixed_window_slider(
            *sensor_dfs,
            slider_fn=compute_fv,
            window_size=self.window_size,
            step_size=self.step_size,
            st=st,
            et=et,
            show_progress=show_progress, placements=placements, srs=srs
        )
        return fs_df

    def _compute_features_per_subj(self, subject, input_type=arus.ds.InputType.MHEALTH_FORMAT, st=None, et=None):

        sensor_dfs, placements, srs = self._import_data_per_subj(
            subject, input_type)

        fs_df = self._compute_features_from_raw_dfs(
            sensor_dfs, placements, srs, st=st, et=et, show_progress=False)

        del sensor_dfs

        return fs_df
