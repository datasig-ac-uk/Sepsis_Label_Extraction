#!/bin/bash
set -e

export PGDATABASE=mimic
export PGUSER=mimicuser
export PGOPTIONS='--search_path=mimiciii'

export BASE_DIR=/docker-entrypoint-initdb.d/custom_sql
( /usr/local/bin/make_ids.sh )
