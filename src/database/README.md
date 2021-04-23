# Data extraction pipeline

To make analysis easier, we will change the relational format of the MIMIC-III database to a pivoted view which includes key demographic information, vital signs, and laboratory readings. This will be done in a similar way to the pivoted tables provided by the MIT team at 
https://github.com/MIT-LCP/mimic-code/concepts/pivot
Separately we will also create tables for the possible sepsis onset times of each patient.

The scripts in this folder is designed to be the complete pipeline of how we took the MIMIC-III data and produced the data and train/test split to use in our analyses.

Once MIMIC-III access is granted, set up a local PSQL database by following the guide in `mimic_setup.md`. We used the version of code found at:
https://github.com/MIT-LCP/mimic-code/tree/5f563bd40fac781eaa3d815b71503a6857ce9599
So please git pull that version to ensure our scripts run smoothly on top of built tables/views.

Once the database is set up and the additional tables/views from the mimic-code repository has been built on top, we are ready to create our pivoted table and sepsis times by running the `sepsis_time.sh` file  
```console
~$ bash sepsis_time.sh
```

Then view the `patient_split.ipynb` to get the train/test split, you can choose whether to create a new split or copy over the original split we used (which can be found in `original_ids`). Build the tables of these ids with
```console
~$ bash make_ids.sh
```

Once this is done, run through `tables_to_csvs.ipynb` notebook to create the csv files which are needed for our analysis.