import functools
import logging
import math
import pathlib
import random
from typing import List, Optional

import config
import tensorflow as tf
import tensorflow.keras as keras

from . import auguments, preprocess


def mk_baseds(sat_path_list: List[str], map_path_list: List[str]):
    base_load_image = functools.partial(
        preprocess.load_image, height=config.IMG_HEIGHT, width=config.IMG_WIDTH
    )

    # 衛星画像のデータセット
    load_image_rgb = functools.partial(base_load_image, channels=3)
    sat_path_list_ds = tf.data.Dataset.list_files(sat_path_list, shuffle=False)
    sat_dataset = sat_path_list_ds.map(
        load_image_rgb, num_parallel_calls=tf.data.AUTOTUNE
    )

    # 地図のデータセット
    load_image_gray = functools.partial(base_load_image, channels=1)
    map_path_list_ds = tf.data.Dataset.list_files(map_path_list, shuffle=False)
    map_dataset = map_path_list_ds.map(
        load_image_gray, num_parallel_calls=tf.data.AUTOTUNE
    )

    sat_map_ds = tf.data.Dataset.zip((sat_dataset, map_dataset))
    return sat_map_ds


def apply_cutmix(sat_map_ds: tf.data.Dataset, nbmix: int):
    # CutMixを適用する。nbmixでbatch, cutmix適用, unbatch.
    # ここが処理のボトルネックになりそう
    logging.info(f"cutmix: {nbmix}")
    sat_map_ds = sat_map_ds.batch(nbmix).map(auguments.cutmix_batch).unbatch()
    return sat_map_ds


def mkds(
    sat_path_list: List[str],
    map_path_list: List[str],
    batch_size: int,
    augment: Optional[keras.layers.Layer] = None,
    cutmix: bool = False,
    nbmix: int = 2,
    test: bool = False,
):

    sat_map_ds = mk_baseds(sat_path_list, map_path_list)
    logging.info(f"batch_size: {batch_size}")

    if test:
        return sat_map_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if cutmix:
        sat_map_ds = apply_cutmix(sat_map_ds, nbmix)

    batch_sat_map_ds = sat_map_ds.shuffle(config.BUFFER_SIZE).batch(batch_size)

    if augment is not None:
        logging.info(f"Apply DataAugment: {augment.name}")
        batch_sat_map_ds = batch_sat_map_ds.map(augment)

    return batch_sat_map_ds.cache().repeat().prefetch(tf.data.AUTOTUNE)


def mk_pathlist(
    ds_root: pathlib.Path, suffix: str, batch_size: int, dsrate: float = 1.0
):
    # * パスのリストを作る
    random.seed(1)
    pathlist = (ds_root / "map").glob(f"*.{suffix}")
    pathlist = sorted([path.name for path in pathlist])
    random.shuffle(pathlist)

    # * 訓練用と検証用に分割
    nb_tr = int(len(pathlist) * 0.8 * dsrate)
    nb_va = int(len(pathlist) * 0.2)
    tr_pathlist = pathlist[:nb_tr]
    va_pathlist = pathlist[nb_tr : nb_tr + nb_va]
    logging.info(f"nb_tr: {nb_tr}")
    logging.info(f"nb_va: {nb_va}")

    # * ステップ数を決める
    tr_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in tr_pathlist])
    tr_map_pathlist = sorted([str(ds_root / "map" / path) for path in tr_pathlist])
    tr_steps = math.ceil(nb_tr / batch_size)

    va_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in va_pathlist])
    va_map_pathlist = sorted([str(ds_root / "map" / path) for path in va_pathlist])
    va_steps = math.ceil(nb_va / batch_size)

    return (
        ((tr_sat_pathlist, tr_map_pathlist), tr_steps),
        ((va_sat_pathlist, va_map_pathlist), va_steps),
    )


def cutmix_ds(sat_map: tf.data.Dataset, nb_mix: int):
    sat_map_cum = (
        sat_map.batch(nb_mix)
        .map(auguments.cutmix_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch()
    )

    return sat_map_cum


def post_process_ds(ds: tf.data.Dataset, batch_size=config.BATCH_SIZE):
    return ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == "__main__":
    pathlist = config.TR_MAP_PATH.glob("*.png")
    pathlist = [path.name for path in pathlist]

    # tr_ds = mk_base_dataset(
    #     pathlist, sat_path=config.TR_SAT_PATH, map_path=config.TR_MAP_PATH
    # )
    # tr_ds = augument_ds(tr_ds, nb_mix=3)
    # tr_ds = post_process_ds(tr_ds)
