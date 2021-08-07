Variation of sepsis-III definitions influences predictive performance of machine learning
==============================

The early detection of sepsis is a key research priority to help facilitate timely intervention.  Criteria used to identify the onset time of sepsis from health records vary, hindering comparison and progress in this field. We considered the effects of variations in sepsis onset definition on the predictive performance of three representive models (i.e. Light gradient boosting machine (LGBM), Long short term memory (LSTM) and Cox proportional-harzard models (CoxPHM)) for early sepsis detection.

This is the code for the paper entitled "Variation of sepsis-III definitions influences predictive performance of machine learning".

The code is composed with the following parts:
1. Extracting the sepsis labelling from the MIMIC-III data based on three sepsis criteria H1-3 and their variants;
2. Training three types of models (i.e. LGBM, LSTM and CoxPHM) for the early sepsis prediction on the datasets produced in Step 1.
3. Evaluating each trained model using the test metrics (e.g. AUROC) and producing the visulaziation plots.

Data
------------
Data is extracted from the MIMIC-III database.
------------

TODO list:
- [ ] connection between database code and analysis code: one line command output the all pivot csv files in /data/raw/.
- [ ] edit readme, probably move the database readme to the main readme.
- [ ] final check analysis code compatibility
- [ ] bash script to download all existing models to model directory.
- [x] code cleaning: delete all unnecessary codes, code linting, code commenting.
- [ ] add notebooks.

------------

Create a new environment and run
```
pip install -r requirements.txt
```
Export PYTHONPATH
```
source pythonpath.sh
```
Raw Data  
------------
You may indicate where the raw data is stored by making changes to `MIMIC_DATA_DIRS` in `src/constants.py`.


Feature extraction
------------

```
python3 features/generate_features.py
```
This commmand will generate list of features which are required for model implementaion and the they will be saved in data/processed.   

Model tuning/training/evaluation 
------------
run the main.py script in src/models with two arguments: model:'LGMB','LSTM','CoxPHM' and process:'train','tune','eval', e.g. For train a LGBM model:
```
python3 models/main.py --model 'LGBM' --process 'train'
```
Hyperparameter tuning, model training and evaluation should be done in sequence as follows:
1. Running the tuning step will compute and save the optimised hyperparameter for later use on model training and evaluation.
2. Then model is trained and saved in /model/ directory for later use on evaluation.
3. Evaluation will produce numerical results and predictions, which are saved in outputs/results and outputs/predictions respectively. 


Note:
In order to reproduce our results with our trained model, you can download all the trained model by
```
bash download_models.sh
```
this will automatically set up the models folder and then run the evaluation for all models, e.g. for LGBM.
```
python3 models/main.py --model 'LGBM' --process 'eval'
```

Illustration 
------------
In order to reproduce all the plots in the paper, run the following command after obtaining all predictions from model evaluation step.   
```
python3 visualization/main_plots.py
```
