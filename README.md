Variation of sepsis-III definitions influencespredictive performance of machine learning
==============================

We consider the effects of variations in onset definition on performance of models for early sepsis detection, namely, LGBM, LSTM and Cox proportional-harzard models. 

Data
------------
Data is extracted from the MIMIC-III database.




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Enviroment Setup
------------

Create a new environment and run
```
pip install -r requirements.txt
```
Raw Data  
------------
You may indicate where the raw data is stored by making changes to `MIMIC_DATA_DIRS` in `src/constants.py`.


Feature extraction and model tuning/evaluation (LGBM for example)
------------

```
python3 src/features/generate_features.py
```
This commmand will generate list of features which are required for model implementaion and the they will be saved in data/processed.   

model tuning/training/evaluation 
------------
run the main.py script in src/models with two arguments: model:'LGMB','LSTM','CoxPHM' and process:'train','tune','eval', e.g. For train a LGBM model:
```
python3 src/models/main.py --model 'LGBM' --process 'train'
```
Hyperparameter tuning, model training and evaluation should be done in sequence as follows:
1. Running the tuning step will compute and save the optimised hyperparameter for later use on model training and evaluation.
2. Then model is trained and saved in /model/ directory for later use on evaluation.
3. Evaluation will produce numerical results and predictions, which are saved in outputs/results and outputs/predictions respectively. 

Illustration 
------------
In order to reproduce all the plots in the paper, run the following command after obtaining all predictions from model evaluation step.   
```
python3 src/visualization/main_plots.py 
```
