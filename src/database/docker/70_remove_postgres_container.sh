#!/bin/bash

docker_id=`docker ps -aqf "name=^mimic$"`

if [ -n "${docker_id+set}" ]; then

echo "Shutting down postgresql"
docker exec -u postgres mimic pg_ctl stop
echo "Stopping container"
docker stop "$docker_id" > /dev/null
echo "Removing container"
docker rm "$docker_id" > /dev/null
echo "Removing image"
docker image rm postgres/mimic > /dev/null

fi
