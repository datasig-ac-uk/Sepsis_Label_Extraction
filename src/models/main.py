import os
import argparse
import sys

sys.path.insert(0, '/')

import constants

MODELS = ['LGBM', 'LSTM', 'CoxPHM']
STEPS = ['tune', 'train', 'eval']

model_parser = argparse.ArgumentParser(description='Execute model training and testing pipeline. When called '
                                       'without any arguments, execute all pipeline steps for all models.')
model_parser.add_argument('--model', metavar='MODEL_NAME', choices=MODELS,
                          help='specify a model (must be one of {LGBM, LSTM, CoxPHM})')
model_parser.add_argument('--step', metavar='STEP_NAME', choices=STEPS,
                          help='the step in the pipeline to execute (must be one of {tune, train, eval})')
model_parser.add_argument('--n_cpus', metavar='N_CPUS', type=int, default=1,
                          help='number of CPUs to use (default 1)')
model_parser.add_argument('--n_gpus', metavar='N_GPUS', type=int, default=1,
                          help='number of GPUs to use (default 1)')

args = model_parser.parse_args()

if args.model is None:
    models_to_include = MODELS
else:
    models_to_include = [args.model]

if args.step is None:
    steps_to_include = STEPS
else:
    steps_to_include = [args.step]

os.environ['N_CPUS'] = str(args.n_cpus)
os.environ['N_GPUS'] = str(args.n_gpus)

for model in models_to_include:
    for step in steps_to_include:
        os.system('python3 src/models/{}/{}_{}.py'.format(model, step, model))
