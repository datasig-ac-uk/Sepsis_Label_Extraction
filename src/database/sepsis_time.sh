#!/bin/bash
set -e

cd $BASE_DIR
psql -f make-all.sql

psql -v v1=sepsis_cohort_time_blood_sensitivity_4824  -v v2="'24'" -v v3="'48'" -f sepsis-time-blood.sql
psql -v v1=sepsis_cohort_time_blood_sensitivity_2412  -v v2="'12'" -v v3="'24'" -f sepsis-time-blood.sql
psql -v v1=sepsis_cohort_time_blood_sensitivity_126  -v v2="'6'" -v v3="'12'" -f sepsis-time-blood.sql
psql -v v1=sepsis_cohort_time_blood_sensitivity_63  -v v2="'3'" -v v3="'6'" -f sepsis-time-blood.sql
