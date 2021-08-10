Subtle variation in sepsis-III definitions influences predictive performance of machine learning
==============================


The early detection of sepsis is a key research priority to help facilitate timely intervention.  Criteria used to identify the onset time of sepsis from health records vary, hindering comparison and progress in this field. We considered the effects of variations in sepsis onset definition on the predictive performance of three representive models (i.e. Light gradient boosting machine (LGBM), Long short term memory (LSTM) and Cox proportional-harzard models (CoxPHM)) for early sepsis detection.

This repository is the official implementation of the paper entitled "Variation of sepsis-III definitions influences predictive performance of machine learning".

The code is composed with the following parts:
1. Extracting the sepsis labelling from the MIMIC-III data based on three sepsis criteria H1-3 and their variants;
2. Training three types of models (i.e. LGBM, LSTM and CoxPHM) for the early sepsis prediction on the datasets produced in Step 1.
3. Evaluating each trained model using the test metrics (e.g. AUROC) and producing the visulaziation plots.




# Environment setup
```console
pip install -r requirements.txt
source pythonpath.sh
```


# Data Extraction Pipeline

To train and evaluate our models, we will change the relational format of the [MIMIC-III database](https://mimic.mit.edu/iii/gettingstarted/overview/) to a pivoted view which includes key demographic information, vital signs, and laboratory readings. We will also create tables for the possible sepsis onset times of each patient. We will subsequently output the pivoted data to comma-separated value (CSV) files, which serve as input for model training and evaluation. 
We provide more detailed instructions under **/src/database** subdirecotry, make sure you run all all data extraction pipeline under that subdirecotry:
```console
cd /src/database
```
Depending on your preferred choice of installing PostgreSQL on your machine yourself or using a Docker container, please proceed with the relevant section in **/src/database/README.md**


# Model Training and Testing Pipeline

Feature Extraction
------------
```console
python3 features/generate_features.py
```
This commmand will generate list of features which are required for model implementaion and the they will be saved in data/processed.   


Model tuning/training/evaluation 
------------
Run the main.py script in src/models with two arguments: model:'LGMB','LSTM','CoxPHM' and process:'train','tune','eval'.
```console
python3 models/main.py --model [Model_name] --process [Process_name]
```
where [Model_name] = 'LGBM', 'LSTM', 'CoxPHM', [Process_name] = 'train', 'eval', 'tune'

Hyperparameter tuning, model training and evaluation should be done in sequence as follows:
1. Running the tuning step will compute and save the optimised hyperparameter for later use on model training and evaluation.
2. Then model is trained and saved in /model/ directory for later use on evaluation.
3. Evaluation will produce numerical results and predictions, which are saved in outputs/results and outputs/predictions respectively. 


Visualizations
------------
In order to reproduce all the plots in the paper, run the following command after obtaining all predictions from model evaluation step.   
```console
python3 visualization/main_plots.py
```

