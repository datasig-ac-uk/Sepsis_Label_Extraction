"""
This file contains basic variables and definitions that we wish to make easily
accessible for any script that requires it.
"""
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])

DATA_DIR = ROOT_DIR + '/data/'
MODELS_DIR = ROOT_DIR + '/models/'
OUTPUT_DIR = ROOT_DIR + '/outputs/'


DATA_processed = DATA_DIR + 'processed/'
DATA_raw = DATA_DIR + 'raw/'

xy_pairs = [(48, 24), (24, 12), (12, 6), (6, 3)]
FEATURES = ['t_sofa', 't_suspicion', 't_sepsis_min']
T_list = [12, 8, 6, 4]

MODELS = ['LGBM', 'LSTM', 'CoxPHM']
models = ['lgbm', 'lstm', 'coxph']
exclusion_rules = ['blood_only','no_gcs','all_cultures','absolute_values','strict_exclusion']

MIMIC_DATA_DIRS = {}
MIMIC_DATA_DIRS['strict_exclusion'] = {'train': '/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data',
                                       'test': DATA_DIR + 'raw/test/metavision_sepsis_blood_only_data'}
MIMIC_DATA_DIRS['blood_only'] = MIMIC_DATA_DIRS['strict_exclusion']

MIMIC_DATA_DIRS['no_gcs'] = {'train': '/scratch/mimiciii/training_data/additional_experiments/no_gcs',
                             'test': DATA_DIR + 'raw/test/no_gcs'}

MIMIC_DATA_DIRS['all_cultures'] = {'train': '/scratch/mimiciii/training_data/additional_experiments/all_cultures',
                                   'test': DATA_DIR + 'raw/test/all_cultures'}

MIMIC_DATA_DIRS['absolute_values'] = {'train': '/scratch/mimiciii/training_data/additional_experiments/absolute_values',
                                      'test': DATA_DIR + 'raw/test/absolute_values'}
