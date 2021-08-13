import os
import argparse
import sys

sys.path.insert(0, '/')

MODELS = ['LGBM', 'LSTM', 'CoxPHM']
STEPS = ['tune', 'train', 'eval']

model_parser = argparse.ArgumentParser(description='Execute model training and testing pipeline. When called '
                                       'without any arguments, execute all pipeline steps for all models.')
model_parser.add_argument('--model', metavar='MODEL_NAME', choices=MODELS,
                          help='specify a model (must be one of {LGBM, LSTM, CoxPHM})')
model_parser.add_argument('--step', metavar='STEP_NAME', choices=STEPS,
                          help='the step in the pipeline to execute (must be one of {tune, train, eval})')
args = model_parser.parse_args()

if args.model is None:
    models_to_include = MODELS
else:
    models_to_include = [args.model]

if args.step is None:
    steps_to_include = STEPS
else:
    steps_to_include = [args.step]

for model in models_to_include:
    for step in steps_to_include:
        os.system('python3 src/models/{}/{}_{}.py'.format(model, step, model))
