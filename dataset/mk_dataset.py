import pathlib

import tensorflow as tf

import config
from . import cutmix
from . import preprocess


def mk_dataset(
    SAT_PATH: pathlib.Path,
    MAP_PATH: pathlib.Path,
    batch_size=config.BATCH_SIZE,
):
    # * sat paths
    sat_path_list = tf.data.Dataset.list_files(str(SAT_PATH / "*.png"), shuffle=False)
    # ** sat images
    sat_dataset = (
        sat_path_list.map(
            preprocess.load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            preprocess.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # .batch(1)
    )

    # * map paths
    map_path_list = tf.data.Dataset.list_files(str(MAP_PATH / "*.png"), shuffle=False)
    # * param myParam
    map_dataset = (
        map_path_list.map(
            preprocess.load_image_gray,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            preprocess.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # .batch(1)
    )

    # * zipped ds
    sat_map = tf.data.Dataset.zip((sat_dataset, map_dataset)).shuffle(
        config.BUFFER_SIZE
    )
    # * apply cutmix
    sat_map_cum = (
        sat_map
        # .map(
        #     preprocess.Augment(),
        #     num_parallel_calls=tf.data.AUTOTUNE,
        # )
        # .unbatch()
        .batch(config.NB_MIX)
        .map(
            cutmix.cutmix_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .unbatch()
        .repeat()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return sat_map_cum


if __name__ == "__main__":
    tr_ds = mk_dataset(SAT_PATH=config.TR_SAT_PATH, MAP_PATH=config.TR_MAP_PATH)
    val_ds = mk_dataset(SAT_PATH=config.VA_SAT_PATH, MAP_PATH=config.VA_MAP_PATH)
