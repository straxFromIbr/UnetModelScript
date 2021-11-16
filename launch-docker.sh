#! /bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

#docker run --rm \
#	-p 8888:8888 \
#	--gpus all \
#	-v $(pwd)/home:/home  \
#    -it jupyterlab_tf:latest bash
#   # --shm-size=1g \
#   # --ulimit memlock=-1 \
#   # --ulimit stack=67108864 

docker-compose down
docker-compose up -d
docker cp ../dataset unetmodelscript_devs_1:/datasets
docker exec -it unetmodelscript_devs_1 bash

