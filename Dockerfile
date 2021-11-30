FROM tensorflow/tensorflow:latest-gpu

# connect: ssh -L 8888:localhost:8888 ${user}@${host}
# build: `docker build -t jupyterlab_tf .`
# run: `docker-compose up`
# access localhost:8888 on local browser


## タイムゾーンの設定：ミラーサーバーの選択時に必要?
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN : INSTALL CV2                                                   && \
    apt-get update && apt-get upgrade -y                            && \
    apt-get install -y tzdata                                       && \
    apt-get install -y --no-install-recommends libopencv-dev        && \
    pip install -U pip                                              && \
    pip install opencv-python opencv-contrib-python                 && \
    :                                                               && \
    :                                                               && \
    : INSTALL PIP packages                                          && \
    pip install jupyterlab matplotlib black isort                      \
                scikit-learn scikit-image tensorflow-datasets       && \
    :                                                               && \
    :                                                               && \
    : INSTALL OTHER PACKAGES                                        && \
    apt-get install -y --no-install-recommends wget vim


CMD jupyter lab \
    --no-browser \
    --allow-root \
    --ip=0.0.0.0 \
    --ServerApp.password='sha1:96686e15b7d0:616a406a297c71a3f824aac48a9cae395639a1a2' \
    /home


