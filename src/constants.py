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
exclusion_rules = ['blood_only/','no_gcs/','all_cultures/','absolute_values/','other_cultures/','strict_exclusion/']

MIMIC_DATA_DIRS = {}
MIMIC_DATA_DIRS['strict_exclusion'] = {'train': DATA_DIR+'raw/train/blood_only',
                                       'test': DATA_DIR + 'raw/test/blood_only'}
MIMIC_DATA_DIRS['blood_only'] = MIMIC_DATA_DIRS['strict_exclusion']

MIMIC_DATA_DIRS['no_gcs'] = {'train': DATA_DIR+'raw/train/no_gcs',
                             'test': DATA_DIR + 'raw/test/no_gcs'}
MIMIC_DATA_DIRS['other_cultures'] = {'train': DATA_DIR+'raw/train/other_cultures',
                                     'test': DATA_DIR + 'raw/test/other_cultures'}
MIMIC_DATA_DIRS['all_cultures'] = {'train': DATA_DIR+'raw/train/all_cultures',
                                   'test': DATA_DIR + 'raw/test/all_cultures'}

MIMIC_DATA_DIRS['absolute_values'] = {'train': DATA_DIR+'raw/train/absolute_values',
                                      'test': DATA_DIR + 'raw/test/absolute_values'}
