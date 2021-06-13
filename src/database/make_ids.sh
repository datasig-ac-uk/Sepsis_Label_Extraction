#!/bin/bash
set -e

cd $BASE_DIR
psql -f id_tables.sql
