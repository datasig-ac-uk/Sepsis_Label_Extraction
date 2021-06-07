#!/bin/bash
set -e

. 00_define_database_environment.sh

echo "Building docker image"
docker build -t postgres/mimic -f Dockerfile ..

if [[ -e "$POSTGRESQL_DATA_PATH/postgresql.conf" ]]; then
    echo "Using already initialised postgresql database in $POSTGRESQL_DATA_PATH"
    BUILD_MIMIC=0
else
    echo "Initialising postgresql database in $POSTGRESQL_DATA_PATH"
    BUILD_MIMIC=1
fi

echo "Starting docker container"
docker run \
--name mimic \
--shm-size=1g \
-p $MIMIC_POSTGRES_PORT:$MIMIC_POSTGRES_PORT \
-e BUILD_MIMIC=$BUILD_MIMIC \
-e POSTGRES_PASSWORD=$MIMIC_POSTGRES_PASSWORD \
-e MIMIC_PASSWORD=$MIMIC_POSTGRES_PASSWORD \
-v $MIMIC_DATA_PATH:/mimic_data \
-v $POSTGRESQL_DATA_PATH:/var/lib/postgresql/data \
-v $(cd ../split_ids && pwd):/docker-entrypoint-initdb.d/split_ids \
-d postgres/mimic

echo "Waiting for postgresql server to start..."
until docker exec mimic pg_isready ;
    do sleep 5
done
