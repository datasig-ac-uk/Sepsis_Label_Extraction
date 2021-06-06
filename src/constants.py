"""
This file contains basic variables and definitions that we wish to make easily
accessible for any script that requires it.
"""
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
data_location= str(Path(__file__).resolve().parents[3])+'/mimiciii/data_27_05_21/'


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

exclusion_rules = ['blood_only', 'no_gcs', 'absolute_values', 'other_cultures', 'strict_exclusion']

MIMIC_DATA_DIRS = {}
MIMIC_DATA_DIRS['strict_exclusion'] = {'train': data_location+'blood_only_data',
                                       'test': data_location+'blood_only_data'}
MIMIC_DATA_DIRS['blood_only'] = MIMIC_DATA_DIRS['strict_exclusion']

MIMIC_DATA_DIRS['no_gcs'] = {'train': data_location+'additional_experiments/no_gcs_cultures',
                             'test': data_location + 'additional_experiments/no_gcs_cultures'}
MIMIC_DATA_DIRS['other_cultures'] = {'train': data_location+'additional_experiments/other_cultures',
                                     'test': data_location + 'additional_experiments/other_cultures'}
# MIMIC_DATA_DIRS['all_cultures'] = {'train': data_location+'additional_experiments/all_cultures',
#                                    'test': data_location + 'additional_experiments/all_cultures'}

MIMIC_DATA_DIRS['absolute_values'] = {'train': data_location+'additional_experiments/absolute_cultures',
                                      'test': data_location + 'additional_experiments/absolute_cultures'}


