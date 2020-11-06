"""arus-muss command line application

Usage:
  arus-muss update -p=<places> -t=<target> [--pids=<pids>] [-c=<cores>] [-d]
  arus-muss predict (-m=<model> | -t=<target>) -p=<places> -f=<format> -o=<output> [-s=<srs>] [--pids=<pids>] [-d] FILE_PATH...

Arguments:
  FILE_PATH     Test file paths stored in signaligner sensor file format, the order should

Options:
  -s <srs>, --srs <srs>                   Set the sampling rates.
  -p <places>, --placements <places>      Set placements. E.g., "DW,DA".
  -t <target>, --target <target>          Set target task name. E.g."INTENSITY".
  -m <model>, --model <model>             Set model path.
  -f <format>, --format <format>          Set input file format. E.g. "MHEALTH_FORMAT" or "SIGNALIGNER".
  -c <cores>, --cores <cores>             Set the number of cores used. E.g., 4.
  -o <output>, --output <output>          Set output folder.
  --pids <pids>                           Set the participant IDs. E.g., "ALL" or "SPADES_1,SPADES_2".
  -d, --debug                             Turn on debug messages.
"""
import sys
import uuid
import os
import pkg_resources

from docopt import docopt
from loguru import logger
from joblib import Memory

import arus

from .har_model import MUSSHARModel

CACHE_DIR = os.path.join(arus.env.get_cache_home(), 'muss')
memory = Memory(location=CACHE_DIR, verbose=0)
BUILTIN_DIR = pkg_resources.resource_filename('arus_muss', 'models')


def cli():
    ver = pkg_resources.get_distribution('arus-muss').version
    arguments = docopt(__doc__, version=f'arus-muss {ver}')
    if arguments['--debug']:
        logger.remove()
        logger.add(sys.stderr, level='DEBUG')
    else:
        logger.remove()
        logger.add(sys.stderr, level='INFO')
    logger.debug(arguments)

    if arguments['update']:
        target, placements, pids, n_cores = _parse_for_update(arguments)
        model = update_builtin_model(target, placements, pids, n_cores)
        model.save_model(save_raw=False, save_fcs=False,
                         output_folder=BUILTIN_DIR)
    elif arguments['predict']:
        model_path, test_files, placements, srs, file_format, output_path = _parse_for_predict(
            arguments)

        predict_df = predict_on_files(model_path, test_files, placements,
                                      srs, file_format)

        logger.info(f'Saving predictions to {output_path}')
        predict_df.to_csv(output_path, index=False)


def _parse_options(option, as_number=False):
    result = option.split(',')
    if as_number:
        result = [float(item) for item in result]
    return result


def _parse_for_update(arguments):
    placements = _parse_options(arguments['--placements'])
    pids = 'ALL' if arguments['--pids'] == 'ALL' or arguments['--pids'] is None else _parse_options(
        arguments['--pids'])
    target = arguments['--target']
    n_cores = int(arguments['--cores'])
    return target, placements, pids, n_cores


def _parse_for_predict(arguments):
    placements = _parse_options(arguments['--placements'])
    target = arguments['--target']
    model_path = arguments['--model']
    srs = None if arguments['--srs'] is None else _parse_options(
        arguments['--srs'], as_number=True)
    file_format = arguments['--format']
    test_files = arguments['FILE_PATH']
    pids = None if arguments['--pids'] is None else _parse_options(
        arguments['--pids'])
    output_folder = arguments['--output']
    model_path = _get_model_path(
        placements=placements, pids=pids, target=target, model_path=model_path)
    output_path = os.path.join(output_folder, os.path.basename(
        model_path).replace('.har', '.prediction.csv'))
    return model_path, test_files, placements, srs, file_format, output_path


def _get_model_path(placements=None, pids='ALL', target=None, model_path=None):
    model_path = model_path or os.path.join(BUILTIN_DIR, MUSSHARModel.build_model_filename(
        MUSSHARModel.name, placements=placements, pids=pids, target=target, dataset_name='SPADES_LAB'))
    return model_path


def update_builtin_model(target, placements, pids='ALL', n_cores=4):
    spades_lab = arus.ds.MHDataset(
        path=arus.ds.get_dataset_path('spades_lab'),
        name='spades_lab', input_type=arus.ds.InputType.MHEALTH_FORMAT)

    spades_lab.set_class_set_parser(arus.slab.class_set)
    spades_lab.set_placement_parser(arus.slab.get_sensor_placement)

    model = MUSSHARModel(
        mid=str(uuid.uuid4()), used_placements=placements, window_size=12.8, sr=80)

    model.load_dataset(spades_lab)

    model.compute_features(pids=pids, n_cores=n_cores)
    model.compute_class_set(
        task_names=[target], pids=pids, n_cores=n_cores)

    model.train(task_name=target, pids=pids, verbose=True)
    return model


def predict_on_files(model_path, test_files, placements, srs, file_format):
    logger.info(f'Loading MUSS model from {os.path.basename(model_path)}')
    model = MUSSHARModel.load_model(model_path)

    logger.info('Making predictions for the input files')
    predict_df = model.predict(*test_files, placements=placements,
                               srs=srs, file_format=file_format)

    return predict_df


if __name__ == "__main__":
    cli()
