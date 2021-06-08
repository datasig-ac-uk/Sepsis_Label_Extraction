import os
import argparse
model_parser = argparse.ArgumentParser(
    description='Argument for specifying model types and process')
model_parser.add_argument('--model', type=str, required=True, default='lgbm')
model_parser.add_argument('--process', type=str,
                          required=True, default='train')
args = model_parser.parse_args()
if args.model == 'LSTM':
    os.system('CUBLAS_WORKSPACE_CONFIG =:16: 8 python3 ' + 'models/' + args.model +
              '/' + args.process + '_' + args.model + '.py')
else:
    os.system('python3 '+'models/'+args.model +'/'+args.process+'_'+args.model+'.py')
