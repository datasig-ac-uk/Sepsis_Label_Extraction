#!/bin/bash
set -e

export PGDATABASE=mimic
export PGUSER=mimicuser
export PGOPTIONS='--search_path=mimiciii'

export MIMIC_DATA_PATH=/mimic_data
export BASE_DIR=/docker-entrypoint-initdb.d/buildmimic/

( /usr/local/bin/load_mimic_database.sh )
