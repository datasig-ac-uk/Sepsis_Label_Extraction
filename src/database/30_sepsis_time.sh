#!/bin/bash
set -e

. 00_define_database_environment.sh
export PGHOST=$MIMIC_POSTGRES_HOSTNAME
export PGPORT=$MIMIC_POSTGRES_PORT
export PGDATABASE=$MIMIC_POSTGRES_DATABASE
export PGUSER=$MIMIC_POSTGRES_USER
export PGPASSWORD=$MIMIC_POSTGRES_PASSWORD
export PGOPTIONS='--search_path=mimiciii'

export BASE_DIR=$(cd sql && pwd)
( ./sepsis_time.sh )

