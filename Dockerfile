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
                jupyterlab_code_formatter tensorflow-datasets          \
                scikit-learn                                        && \
    :                                                               && \
    :                                                               && \
    : INSTALL NodeJS for JL extension                               && \
    curl -sL https://deb.nodesource.com/setup_12.x | bash -         && \
    apt-get install -y --no-install-recommends nodejs wget vim git  && \
    apt-get upgrade -y                                              && \
    :                                                               && \
    :                                                               && \
    : INSTALL JL Extensions                                         && \
    pip install JLDracula                                           && \
    jupyter labextension install '@axlair/jupyterlab_vim'           && \
    jupyter lab build                                               && \
    :                                                               && \
    :                                                               && \
    : INSTALL OTHER PACKAGES                                        && \
    apt-get install -y --no-install-recommends wget vim git


CMD jupyter lab \
    --no-browser \
    --allow-root \
    --ip=0.0.0.0 \
    --ServerApp.password='sha1:96686e15b7d0:616a406a297c71a3f824aac48a9cae395639a1a2' \
    /home


