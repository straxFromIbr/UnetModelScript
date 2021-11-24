---
title: "研究室のPCの環境構築メモ"
date: 2021-10-11T14:02:08+09:00
categories:
- 卒研
- 環境構築
tags:
- 卒研
- 環境構築
summary: "研究室のPCの環境構築メモ"
---
## 目次
- [目次](#目次)
- [1. Dockerのインストール前の準備](#1-dockerのインストール前の準備)
- [2. Dockerのインストール](#2-dockerのインストール)
- [3. インストール後の準備](#3-インストール後の準備)
- [4. TensorFlow2とJupyter Labを使う](#4-tensorflow2とjupyter-labを使う)
- [5. Dockerコンテナにリモート接続する](#5-dockerコンテナにリモート接続する)

ステップ3まではセットアップについて。4以降が使い方

## 1. Dockerのインストール前の準備
1. Windows 11にアップデート
1. WSL2(Ubuntu)をインストール
1. プロキシの設定
    WSL2上で次のようにファイルを編集する。
    - `/etc/apt/aptconf`(su権限が必要)
        ```
        Acquire::http::Proxy "http://po.cc.ibaraki-ct.ac.jp:3128";
        Acquire::https::Proxy "http://po.cc.ibaraki-ct.ac.jp:3128";
        ```
    - `~/.bashrc`に次を追記。
        ```bash
        export HTTPS_PROXY=http://po.cc.ibaraki-ct.ac.jp:3128
        export  HTTP_PROXY=http://po.cc.ibaraki-ct.ac.jp:3128
        export https_proxy=http://po.cc.ibaraki-ct.ac.jp:3128
        export  http_proxy=http://po.cc.ibaraki-ct.ac.jp:3128
        ```
## 2. Dockerのインストール
次のガイドに従う
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- おおまかに次の皇帝が含まれる
    - CUDAドライバーのインストール
    - WSLバックエンドのDockerのインストール

## 3. インストール後の準備
1. プロキシ設定
    - WSL2でファイル`/etc/default/docker`に次を追記(su権限必要)
    - これでDocker Hubからpullができる。
        ```bash
        export  http_proxy="http://po.cc.ibaraki-ct.ac.jp:3128"
        export https_proxy="http://po.cc.ibaraki-ct.ac.jp:3128"
        ```
2. GPUの確認
    - 次のコマンドを実行
        ```
        docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
        ```

## 4. TensorFlow2とJupyter Labを使う
1. `docker-compose.yaml`に次の内容を書く。(パスはどこでも良い)
    ```yaml
    version: "3.3"
    services:
      devs:
        ports:
          - "8888:8888"
          - "6006:6006"
        volumes:
          - ".:/home"
        image: jupyterlab_tf
        deploy:
          resources:
            reservations:
              devices:
                - capabilities: [gpu]
    ``` 
    (補足：`volumes`のコロン前(`.`)がホスト側のパス、後ろ(`/home`)がコンテナ内のパス)
2. 上記YAMLファイルがあるパスで次のコマンドを実行。
    ```
    docker-compose up
    ```
    Windows側のブラウザで[localhost:8888](http://localhost:8888) にアクセス。Dockerコンテナ側の`/home`からホスト側のファイルにアクセスできる。


## 5. Dockerコンテナにリモート接続する
1. WindowsにSSH接続する時、8888をポートフォワーディングする。
    ``` 
    ssh -L 8888:localhost:8888 r-kodama@172.16.161.107
    ```
2. SSHで接続した後`wsl`コマンドでWSL2を起動し...
3. `docker-compose up`でDockerコンテナを起動。
4. ブラウザで[localhost:8888](http://localhost:8888)にアクセス。
