"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path

# Packages/functions used everywhere
from src.omni.functions import *


ROOT_DIR = str(Path(__file__).resolve().parents[0])

DATA_DIR = ROOT_DIR + '/data/'
MODELS_DIR = ROOT_DIR + '/models/'
OUTPUT_DIR=ROOT_DIR + '/outputs/'


DATA_processed=DATA_DIR + 'processed/'
DATA_raw=DATA_DIR + 'raw/'

xy_pairs=[(48,24),(24,12),(12,6),(6,3)]
definitions=['t_sofa','t_suspicion','t_sepsis_min']
T_list=[12,8,6,4]

MODELS=['LGBM','LSTM','CoxPHM']
models=['lgbm','lstm','coxph']
