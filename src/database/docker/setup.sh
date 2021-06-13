#!/bin/bash

if [ $BUILD_MIMIC -eq 1 ]
then
echo 'INITIALISING MIMIC DATABASE ... '

# this flag allows us to initialize the docker repo without building the data
echo 'INITIALISING MIMIC DATABASE ... '
echo "running create mimic user"

pg_ctl stop

pg_ctl -D "$PGDATA" \
	-o "-c listen_addresses='' -c checkpoint_timeout=600" \
	-w start

psql <<- EOSQL
    CREATE USER mimicuser WITH PASSWORD '$MIMIC_PASSWORD';
    CREATE DATABASE mimic OWNER mimicuser;
    \c mimic;
    CREATE SCHEMA mimiciii;
		ALTER SCHEMA mimiciii OWNER TO mimicuser;
EOSQL

echo 'DONE INITIALISING MIMIC DATABASE'
fi
