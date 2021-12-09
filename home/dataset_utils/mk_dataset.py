import functools
from typing import List

import tensorflow as tf

import config

from . import auguments, preprocess


def mk_base_dataset(sat_path_list: List[str], map_path_list: List[str]):

    image_loader_rgb = functools.partial(preprocess.load_image, channels=3)
    sat_path_list_ds = tf.data.Dataset.list_files(sat_path_list, shuffle=False)
    sat_dataset = sat_path_list_ds.map(
        image_loader_rgb, num_parallel_calls=tf.data.AUTOTUNE
    )

    image_loader_gry = functools.partial(preprocess.load_image, channels=1)
    map_path_list_ds = tf.data.Dataset.list_files(map_path_list, shuffle=False)
    map_dataset = map_path_list_ds.map(
        image_loader_gry, num_parallel_calls=tf.data.AUTOTUNE
    )

    # fmt:off
    sat_map = (
        tf.data.Dataset
        .zip((sat_dataset, map_dataset))
        .shuffle(config.BUFFER_SIZE)
        .repeat()
    )

    return sat_map


def cutmix_ds(sat_map: tf.data.Dataset, nb_mix: int):
    sat_map_cum = (
        sat_map
        .batch(nb_mix)
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
