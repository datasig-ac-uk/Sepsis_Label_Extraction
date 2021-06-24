Variation of sepsis-III definitions influences predictive performance of machine learning
==============================

We consider the effects of variations in onset definition on performance of three representative models, i.e. Light gradient boosting machine (LGBM), Long short term memory (LSTM) and Cox proportional-harzard models (CoxPHM) for early sepsis detection.

# Environment setup
```console
pip install -r requirements.txt
source pythonpath.sh
```

# Data Extraction Pipeline

To train and evaluate our models, we will change the relational format of the [MIMIC-III database](https://mimic.mit.edu/iii/gettingstarted/overview/) to a pivoted view which includes key demographic information, vital signs, and laboratory readings. We will also create tables for the possible sepsis onset times of each patient. We will subsequently output the pivoted data to comma-separated value (CSV) files, which serve as input for model training and evaluation. The scripts in this directory accomplish the following steps, which we refer to as our data extraction pipeline:

* Initialise a PostgreSQL installation with a new database for storing MIMIC-III data
* Populate the database using MIMIC-III data files
* Change the relational format of the database
* Generate CSV files containing IDs of exemplars in training and testing subsets
* Generate CSV files containing data for model training and testing

# Preliminaries

Before executing any of the steps in the pipeline, it is necessary that you complete the following tasks [described on the MIMIC-III website](https://mimic.mit.edu/iii/gettingstarted/):

* [Become a credentialed user on PhysioNet](https://physionet.org/settings/credentialing/). This involves completion of a half-day online training course in human subjects research.
* Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
* Download the MIMIC-III data locally by [navigating to the Files section on the PhysioNet MIMIC-III project page](https://physionet.org/content/mimiciii/1.4/#files)
* You may alternatively download the files using the `wget` utility:
```console
wget -r -N -c -np --user username --ask-password https://physionet.org/files/mimiciii/1.4/
```

# Requirements

Running the data extraction pipeline requires a Bash command line environment. We have successfully tested the pipeline using Bash 3.2 under MacOS Catalina 10.15 and Bash 4.4.20 under Ubuntu Linux 18.04.

As a prerequisite for running the Jupyter notebooks which are part of the data extraction pipeline, please install the Python dependencies listed in [requirements.txt](requirements.txt). A typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing
```console
pip install -r requirements.txt
```

Finally, you will also need a running local PostgreSQL installation. As an alternative to installing Postgresql manually on your local machine, we provide scripts for automatically deploying PostgreSQL inside a Docker container (and subsequently removing the container). We have successfully tested the latter approach using Docker Desktop 2.2.0.3 under MacOS Catalina 10.15.

The populated PostgreSQL database required around 65GB of storage space.

Depending on your preferred choice of installing PostgreSQL on your machine yourself or using a Docker container, please proceed with the relevant section below.

# Local PostgreSQL Installation
## Option 1: Install from Package
To install PostgreSQL locally, we recommend using a package manager if possible, e.g. for Ubuntu Linux
```console
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
```
or for Mac
```console
brew install postgres
brew services start postgresql
```

## Option 2: Install from Source

Another possibility is to install from source by cloning the repository at https://github.com/postgres/postgres. If your operating system does not permit you to follow the steps, then you should consider installing from source too.
Steps to install from source:
* Make a new directory at a suitable location,  e.g. 
```console
mkdir source
```
* Clone repository 
```console
git clone git@github.com:postgres/postgres.git
```
* Go to the cloned directory with `cd ...` 
* Next execute the following commands
```console
./configure --prefix=/path_to_installation_directory
make
make install
```
* Add the line `export PATH=/path_of_directory_of_installation` to your .bashrc file.
* Source this file
```console
source .bashrc
```
* Then change directory again e.g.`cd path_of_directory_of_installation`
* Then make a new folder and move there:
```console
mkdir data
cd data
```
* Initialise the server with 
```console
initdb
```
* Now go back to the .bashrc file and add the line `export PGDATA=path_of_directory_of_installation/data` and again source the file with
```console
source .bashrc
```
* Now you should be able to initialise the database server with (changing the placeholder name)
```console
path_of_directory_of_installation/bin/pg_ctl -D path_of_directory_of_installation/data/ -l logfile start
```

## Test the PostgreSQL Installation
Check that you have psql installed with
```console
psql -V
```
#

## Initialise PostgreSQL installation with a new database for storing MIMIC-III data

The script `00_define_database_environment.sh` contains relevant environment variables for connecting to PostgreSQL. It should not be necessary to change the value of any of these variables, with the exception of MIMIC_DATA_PATH, which you should change to the path containing the MIMIC-III data files you downloaded previously.

To initialise the PostgreSQL installation with a new database for storing MIMIC-III data, invoke the script
```console
./10_initialise_mimic_database.sh
```
Note that this script and all subsequent scripts in the pipeline should be invoked from within this directory. If you are asked to provide a password, try authenticating using the default password `postgres`.

## Populate the database using MIMIC-III data files

Next, populate the newly created database by invoking the script
```console
./20_load_mimic_database.sh
```
Note that the preceding two steps are based on the instructions for installing MIMIC-III manually [available on the MIMIC website](https://mimic.mit.edu/iii/tutorials/install-mimic-locally-ubuntu/).

## Change the relational format of the database

Next, change the relational format of the database by invoking the script
```console
./30_sepsis_time.sh
```
NB: Please provision for several hours for `./20_load_mimic_database.sh` and `./30_sepsis_time.sh` to complete.

Note also that our adaptations and database loading scripts are based on the [mimic-code repository](https://github.com/MIT-LCP/mimic-code/tree/5f563bd40fac781eaa3d815b71503a6857ce9599) at commit 5f563bd40fac781eaa3d815b71503a6857ce9599. We include all required scripts as part of this repository, therefore it is not necessary to check out any of the aforementioned repository separately.

## Generate CSV files containing IDs of exemplars in training and testing subsets

To generate the CSV files containing IDs of exemplars in training and testing subsets, open the following notebook in Jupyter and execute all cells
```console
40_patient_split.ipynb
```
To avoid issues with working memory consumption, after executing `40_patient_split.ipynb` we recommend shutting down the notebook.

## Generate CSV files containing data for model training and testing

This step requires first executing the script
```console
./50_make_ids.sh
```
Once the script has completed, open the following notebook in Jupyter and execute all cells
```console
60_tables_to_csvs_final.ipynb
```
The result of executing 60_tables_to_csvs_final.ipynb should be that the CSV files for subsequent model training and testing are reproduced and output to the directory [../../data/raw/](../../data/raw/).

# Deploy PostgreSQL using Docker

The scripts for executing the analogous pipeline based on Docker are located inside the directory [docker/](docker/). In common with the scripts described in the preceding section, it is required that all scripts are executed from inside te aforementioned directory. This is the reason why in the following, we execute scripts inside a subshell, e.g. `(cd docker && ./10_initialise_mimic_database.sh)`. As an alternative, you may simply use `cd docker`. However, please note that the Jupyter notebooks are located inside the parent directory.

## Initialise PostgreSQL installation with a new database for storing MIMIC-III data

The script `./docker/00_define_database_environment.sh` contains relevant environment variables for connecting to PostgreSQL. It should not be necessary to change the values of MIMIC_POSTGRES_PORT and MIMIC_POSTGRES_PASSWORD. However, you should change MIMIC_DATA_PATH to the path containing the MIMIC-III data files you downloaded previously. In addition, you should change POSTGRESQL_DATA_PATH to a newly created directory which will be used to store the database.

To initialise the PostgreSQL installation with a new database for storing MIMIC-III data, invoke the script with
```console
(cd docker && ./10_initialise_mimic_database.sh)
```
## Populate the database using MIMIC-III data files

Next, populate the newly created database by invoking
```console
(cd docker && ./20_load_mimic_database.sh)
```
## Change the relational format of the database

Next, change the relational format of the database by invoking
```console
(cd docker && ./30_sepsis_time.sh)
```
NB: Please provision for several hours for `20_load_mimic_database.sh` and `30_sepsis_time.sh` to complete.
## Generate CSV files containing IDs of exemplars in training and testing subsets

To generate the CSV files containing IDs of exemplars in training and testing subsets, open the following notebook in Jupyter and execute all cells
```console
40_patient_split.ipynb
```
To avoid issues with working memory consumption, after executing `40_patient_split.ipynb` we recommend shutting down the notebook.
## Generate CSV files containing data for model training and testing

This step requires first executing
```console
(cd docker && ./50_make_ids.sh)
```
Once the script has completed, open the following notebook in Jupyter and execute all cells
```console
60_tables_to_csvs_final.ipynb
```
As was described in the preceding section, the result of executing 60_tables_to_csvs_final.ipynb should be that the CSV files for subsequent model training and testing are reproduced and output to the directory [../../data/raw/](../../data/raw/).

## Remove the Docker container

To remove the Docker container, execute
```console
(cd docker && ./70_remove_postgres_container.sh)
```
You may also wish to clean up by deleting the contents of the directory you assigned to POSTGRESQL_DATA_PATH.


# Model Training and Testing Pipeline

Feature Extraction
------------
```console
python3 features/generate_features.py
```
This commmand will generate list of features which are required for model implementaion and the they will be saved in data/processed.   

Model tuning/training/evaluation 
------------
run the main.py script in src/models with two arguments: model:'LGMB','LSTM','CoxPHM' and process:'train','tune','eval'.
```console
python3 models/main.py --model [Model_name] --process [Process_name]
```
Where [Model_name] = 'LGBM', 'LSTM', 'CoxPHM', [Process_name] = 'train', 'eval', 'tune'

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
