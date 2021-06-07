#!/bin/bash
set -e

echo 'POPULATING MIMIC DATABASE ... '

# check for the admissions to set the extension
if [ -e "$MIMIC_DATA_PATH"/ADMISSIONS.csv.gz ]; then
  COMPRESSED=1
  EXT='.csv.gz'
elif [ -e "$MIMIC_DATA_PATH"/ADMISSIONS.csv ]; then
  COMPRESSED=0
  EXT='.csv'
else
  echo "Unable to find a MIMIC data file (ADMISSIONS) in $MIMIC_DATA_PATH"
  exit 1
fi

# check for all the tables, exit if we are missing any
ALLTABLES='ADMISSIONS CALLOUT CAREGIVERS CHARTEVENTS CPTEVENTS DATETIMEEVENTS D_CPT DIAGNOSES_ICD D_ICD_DIAGNOSES D_ICD_PROCEDURES D_ITEMS D_LABITEMS DRGCODES ICUSTAYS INPUTEVENTS_CV INPUTEVENTS_MV LABEVENTS MICROBIOLOGYEVENTS NOTEEVENTS OUTPUTEVENTS PATIENTS PRESCRIPTIONS PROCEDUREEVENTS_MV PROCEDURES_ICD SERVICES TRANSFERS'

for TBL in $ALLTABLES; do
  if [ ! -e "$MIMIC_DATA_PATH"/$TBL$EXT ];
  then
    echo "Unable to find $TBL$EXT in $MIMIC_DATA_PATH"
    exit 1
  fi
  echo "Found all tables in $MIMIC_DATA_PATH - beginning import from $EXT files."
done

# checks passed - begin building the database
PG_MAJOR=`psql --version | rev | cut -d' ' -f1 | rev | cut -d. -f1`
if [ ${PG_MAJOR:0:1} -eq 1 ]; then
echo "$0: running postgres_create_tables_pg10.sql"
psql < $BASE_DIR/postgres/postgres_create_tables_pg10.sql
else
echo "$0: running postgres_create_tables.sql"
psql < $BASE_DIR/postgres/postgres_create_tables.sql
fi

if [ $COMPRESSED -eq 1 ]; then
echo "$0: running postgres_load_data_gz.sql"
cd $BASE_DIR/postgres/
psql -v mimic_data_dir=$MIMIC_DATA_PATH < postgres_load_data_gz.sql
else
echo "$0: running postgres_load_data.sql"
cd $BASE_DIR/postgres/
psql -v mimic_data_dir=$MIMIC_DATA_PATH < postgres_load_data.sql
fi

echo "$0: running postgres_add_indexes.sql"
cd $BASE_DIR/postgres/
psql < postgres_add_indexes.sql

echo "$0: running postgres_add_constraints.sql"
cd $BASE_DIR/postgres/
psql < postgres_add_constraints.sql

echo "$0: running postgres_checks.sql (all rows should return PASSED)"
cd $BASE_DIR/postgres/
psql < postgres_checks.sql

echo "$0: running make-concepts.sql"
cd $BASE_DIR/../concepts/
psql < make-concepts.sql

echo "$0: running pivoted-gcs.sql"
cd $BASE_DIR/../concepts/pivot/
psql < pivoted-gcs.sql

echo "$0: running pivoted-uo.sql"
cd $BASE_DIR/../concepts/pivot/
psql < pivoted-uo.sql

echo "$0: running pivoted-uo.sql"
cd $BASE_DIR/../concepts/durations/
psql < epinephrine-dose.sql

echo "$0: running pivoted-uo.sql"
cd $BASE_DIR/../concepts/durations
psql < norepinephrine-dose.sql

echo "$0: running dopamine-dose.sql"
cd $BASE_DIR/../concepts/durations
psql < dopamine-dose.sql

echo "$0: running dopamine-dose.sql"
cd $BASE_DIR/../concepts/durations
psql < dobutamine-dose.sql

echo 'DONE POPULATING MIMIC DATABASE'
