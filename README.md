Subtle Variation in Sepsis-III Definitions Influences Predictive Performance of Machine Learning
==============================

The early detection of sepsis is a key research priority to help facilitate timely intervention.  Criteria used to identify the onset time of sepsis from health records vary, hindering comparison and progress in this field. We considered the effects of variations in sepsis onset definition on the predictive performance of three representative models (i.e. Light gradient boosting machine (LGBM), Long short term memory (LSTM) and Cox proportional-harzard models (CoxPHM)) for early sepsis detection.

This repository is the official implementation of the paper entitled "Variation of Sepsis-III Definitions Influences Predictive Performance of Machine Learning".

This repository contains code for the following parts in our experimental pipeline:
1. Extracting the sepsis labelling from the MIMIC-III data based on three sepsis criteria H1-3 and their variants (see [src/database](src/database))
2. Training three types of models (i.e. LGBM, LSTM and CoxPHM) for the early sepsis prediction on the datasets produced in Step 1. (see [src/models](src/models))
3. Evaluating each trained model using the test metrics (e.g. AUROC) and producing the visualization plots (see [src/visualization](src/visualization))

# Environment Setup
The code has been tested successfully using Python 3.7; thus we suggest using this version or a later version of Python. A typical process for installing the package dependencies involves creating a new Python virtual environment.

To install the required packages, run the following:
```console
pip install -r requirements.txt
```

Finally, to prepare the environment for running the code, run the following:
```console
source pythonpath.sh
```

# Data Extraction Pipeline
To train and evaluate our models, we will change the relational format of the [MIMIC-III database](https://mimic.mit.edu/iii/gettingstarted/overview/) to a pivoted view which includes key demographic information, vital signs, and laboratory readings. We will also create tables for the possible sepsis onset times of each patient. We will subsequently output the pivoted data to comma-separated value (CSV) files, which serve as input for model training and evaluation. 

Prior to running any of the data extraction commands, make sure to change to the [src/database](src/database) subdirectory:
```console
cd src/database
```

Next, please follow the instructions in the [data extraction README.md](src/database/README.md). (Depending on your preferred choice of installing PostgreSQL on your machine yourself or using a Docker container, please follow the relevant sections in the [data extraction README.md](src/database/README.md).)

# Model Training and Testing Pipeline

Feature Extraction
------------
To generate the derived features mentioned in our paper, simply run the following:
```console
python3 src/features/generate_features.py
```
The preceding command will save features required for model training/tuning/evaluation to [data/processed](data/processed).

Model tuning/training/evaluation 
------------
Initiate model tuning, training and evaluation using the [main.py](src/models/main.py) script. This script takes four optional arguments: `--model`, `--process`, `--n_cpus`, and `--n_gpus`:
```console
python3 src/models/main.py --model MODEL_NAME --step STEP_NAME --n_cpus N_CPUS --n_gpus N_GPUS 
```
where `MODEL_NAME` is either `LGBM`, `LSTM`, or `CoxPHM` and where `STEP_NAME` is either `tune` `train`, or `eval`. Furthermore, `N_CPUS` is the number of CPUs and `N_GPUs` is the number of GPUs.

For each of the three models (`LGBM`, `LSTM`, and `CoxPHM`), the required sequence of steps is `tune`, `train`, `eval`:
1. `tune`: For a given model, running the tuning step computes and saves optimal hyperparameters for subsequent training and evaluation.
2. `train`: The model is trained and saved to the [model/](model/) directory for subsequent evaluation.
3. `eval`: Evaluation involves generating numerical results and predictions, which are respectively saved to [outputs/results](outputs/results) and [outputs/predictions](outputs/predictions). 

**Note:** To run all three above steps in the required order for all three models on 1 CPU and on 1 GPU, simply run [main.py](src/models/main.py) without any arguments, i.e.
```console
python3 src/models/main.py 
```
**Note:** The full pipeline may take several hours to complete, therefore you can alternatively download our pretrained model and obtain the results directly by the following commands:
```console
bash pretrained_models.sh
python3 src/models/main.py --model MODEL_NAME --step 'eval'
```

Visualizations
------------
To reproduce all the plots in the paper, after having run the model evaluation step run the following command:  
```console
python3 src/visualization/main_plots.py
```
