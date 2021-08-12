import os
import argparse
import sys
sys.path.insert(0, '/')
model_parser = argparse.ArgumentParser(
    description='Argument for specifying model types and process')
model_parser.add_argument('--model', type=str, required=True, default='LGBM')
model_parser.add_argument('--process', type=str,
                          required=True, default='train')
args = model_parser.parse_args()
os.system('python3 '+'models/'+args.model +'/'+args.process+'_'+args.model+'.py')
