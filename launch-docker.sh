docker run --rm \
	-p 8888:8888 \
	--gpus all \
	-v $(pwd)/home:/home  \
    -it jupyterlab_tf:latest bash
   # --shm-size=1g \
   # --ulimit memlock=-1 \
   # --ulimit stack=67108864 
