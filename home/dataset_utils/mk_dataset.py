import pathlib
import functools
from typing import List

import tensorflow as tf

import config

from . import auguments, preprocess


def mk_base_dataset(
    path_list: List[str],
    sat_path: pathlib.Path,
    map_path: pathlib.Path,
):
    sat_path_list = sorted([str(sat_path / path) for path in path_list])
    map_path_list = sorted([str(map_path / path) for path in path_list])

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

    sat_map = (
        tf.data.Dataset.zip((sat_dataset, map_dataset))
        # .cache()
        .shuffle(config.BUFFER_SIZE).repeat()
    )

    return sat_map


def augument_ds(sat_map: tf.data.Dataset, nb_mix: int):
    sat_map_cum = (
        sat_map.batch(nb_mix)
        #! .cache() ## コイツが `kernel dead`の元凶。メモリ喰い。
        .map(auguments.Augment())
        .map(auguments.cutmix_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch()
    )

    return sat_map_cum


def post_process_ds(ds: tf.data.Dataset, batch_size=config.BATCH_SIZE):
    return ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == "__main__":
    pathlist = config.TR_MAP_PATH.glob("*.png")
    pathlist = [path.name for path in pathlist]

    tr_ds = mk_base_dataset(
        pathlist, sat_path=config.TR_SAT_PATH, map_path=config.TR_MAP_PATH
    )
    tr_ds = augument_ds(tr_ds, nb_mix=3)
    tr_ds = post_process_ds(tr_ds)
