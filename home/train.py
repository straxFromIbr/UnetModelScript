import functools
import logging
import math
import pathlib
import random
from typing import List

import tensorflow as tf
import tensorflow.keras as keras

import config
from dataset_utils import auguments, preprocess
from model import losses, unet
from utils import callbacks


def mk_baseds(sat_path_list: List[str], map_path_list: List[str], eq: bool):
    logging.info(f"eq: {eq}")
    base_load_image = functools.partial(
        preprocess.load_image, height=config.IMG_HEIGHT, width=config.IMG_WIDTH
    )

    # 衛星画像のデータセット
    load_image_rgb = functools.partial(base_load_image, channels=3, eq=eq)
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


def apply_da(
    batch_sat_map_ds: tf.data.Dataset,
    zoom: bool = True,
    flip: bool = True,
    rotate: bool = True,
):
    batch_sat_map_ds = batch_sat_map_ds.map(
        auguments.Augment(zoom=zoom, flip=flip, rotate=rotate),
        # num_parallel_calls=tf.data.AUTOTUNE,
        # deterministic=True,
    )
    logging.info(f"zoom: {zoom}, flip: {flip}, rotate: {rotate}")
    return batch_sat_map_ds


def apply_cutmix(sat_map_ds: tf.data.Dataset, nbmix: int):
    # CutMixを適用する。nbmixでbatch, cutmix適用, unbatch.
    # ここが処理のボトルネックになりそう
    logging.info(f"cutmix: {nbmix}")
    sat_map_ds = sat_map_ds.batch(nbmix).map(auguments.cutmix_batch).unbatch()
    return sat_map_ds


def mk_testds(sat_map: tf.data.Dataset, batch_size: int):
    sat_map = sat_map.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return sat_map


def mkds(
    sat_path_list: List[str],
    map_path_list: List[str],
    batch_size: int,
    eq: bool = False,
    aug: bool = False,
    zoom: bool = True,
    flip: bool = True,
    rotate: bool = True,
    cutmix: bool = False,
    nbmix: int = 2,
    test: bool = False,
):
    sat_map_ds = mk_baseds(sat_path_list, map_path_list, eq)
    if test:
        return mk_testds(sat_map_ds, batch_size)

    if cutmix and aug:
        sat_map_ds = apply_cutmix(sat_map_ds, nbmix)

    batch_sat_map_ds = sat_map_ds.shuffle(config.BUFFER_SIZE).batch(batch_size)
    logging.info(f"batch_size: {batch_size}")
    if aug:
        # fmt:off
        batch_sat_map_ds = apply_da(batch_sat_map_ds, zoom=zoom, flip=flip, rotate=rotate)
        # fmt:on

    return batch_sat_map_ds.repeat().prefetch(tf.data.AUTOTUNE)


# Define model
def compile_model(
    loss,
    freeze_enc=False,
    freeze_dec=False,
    xception=False,
):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE,
        output_channels=config.OUT_CH,
        freeze_enc=freeze_enc,
        freeze_dec=freeze_dec,
    )
    if xception:
        del model
        model = unet.xception_unet(img_size=config.INPUT_SIZE, num_classes=1)

    # Compile the model
    optimizer = keras.optimizers.Adam()
    metric_list = [keras.metrics.MeanIoU(num_classes=2)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    return model


# Define Callbacks
def get_callbacks(filename):
    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename))
    checkpoint_path = str(config.CHECKPOINT_PATH / filename / filename) + "_{epoch:03d}"

    checkpoint_cb = callbacks.get_checkpoint_callback(checkpoint_path)
    callback_list = [tboard_cb, checkpoint_cb]
    return callback_list


def getargs():
    # fmt:off
    parser = argparse.ArgumentParser(description="U-Netによる道路検出。CutMix")
    parser.add_argument("--logdir", help="ログ/チェックポイントのパス", type=str, required=True)

    # *モデル設定
    parser.add_argument("--xception", help="Xception", action="store_true", default=False)
    parser.add_argument("--pretrained", help="事前学習済みのモデル", type=str, required=False)
    parser.add_argument("--freeze_enc", help="エンコーダ部分を凍結", action="store_true", default=False)
    parser.add_argument("--freeze_dec", help="デコーダ部分を凍結", action="store_true", default=False)

    # *損失関数・学習の設定
    parser.add_argument("--loss",help='損失関数', choices=["DICE", "BCEwithDICE", "Tversky", "Focal"], default="DICE")
    parser.add_argument("--alpha", help="Tversky損失のalpha", type=float, default=0.7)
    parser.add_argument("--gamma", help="Focal損失のgamma", type=float, default=0.75)
    parser.add_argument("--epochs", help="エポック数", type=int, required=True)

    # *データセット設定
    parser.add_argument("--datadir", help="データセットのパス", type=str)
    parser.add_argument("--suffix", help="データの拡張子", type=str, required=True)
    parser.add_argument("--dsrate", help="データセットを使う割合", type=float, default=1.0)

    # *前処理設定
    parser.add_argument("--eq", help="ヒストグラム正規化", action="store_true")
    parser.add_argument("--use_cutmix", help="Cutmixを使う？", action="store_true")
    parser.add_argument("--nbmix", help="CutMix数", type=int, default=3)
    parser.add_argument("--augment", help="前処理",  action="store_true")
    parser.add_argument("--zoom", help="Zoom", action="store_true")
    parser.add_argument("--rotate", help="Rotate", action="store_true")
    parser.add_argument("--flip", help="Flip", action="store_true")

    # fmt:on
    args = parser.parse_args()

    loss_dic = {
        "DICE": losses.DICELoss("DICE"),
        "BCEwithDICE": losses.BCEwithDICELoss("BCEwithDICE"),
        "Tversky": losses.TverskyLoss("Tversky"),
        "Focal": losses.FocalTverskyLoss("Focal"),
    }
    # * 損失関数の設定をここでしちゃう
    args.loss = loss_dic[args.loss]
    del loss_dic
    return args


def main(**args):
    # * パスのリストを作る
    random.seed(1)
    pathlist = (pathlib.Path(args["datadir"]) / "map").glob(f"*.{args['suffix']}")
    pathlist = sorted([path.name for path in pathlist])
    random.shuffle(pathlist)

    # * 訓練用と検証用に分割
    nb_tr = int(len(pathlist) * 0.8 * args["dsrate"])
    nb_va = int(len(pathlist) * 0.2)
    tr_pathlist = pathlist[:nb_tr]
    va_pathlist = pathlist[nb_tr : nb_tr + nb_va]
    logging.info(f"nb_tr: {nb_tr}")
    logging.info(f"nb_va: {nb_va}")

    # * ステップ数を決める
    steps_per_epoch = math.ceil(nb_tr / config.BATCH_SIZE)
    va_steps = math.ceil(nb_va / config.BATCH_SIZE)

    # * tf.Dataの作成
    ds_root = pathlib.Path(args["datadir"])

    # fmt:off
    tr_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in tr_pathlist])
    tr_map_pathlist = sorted([str(ds_root / "map" / path) for path in tr_pathlist])
    train_ds = mkds(
        tr_sat_pathlist, 
        tr_map_pathlist, 
        batch_size=config.BATCH_SIZE, 
        eq=args["eq"], 
        aug=args["augment"],
        zoom=args["zoom"],
        flip=args["flip"],
        rotate=args["rotate"],
        cutmix=args["use_cutmix"],
        nbmix=args["nbmix"],
    )

    va_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in va_pathlist])
    va_map_pathlist = sorted([str(ds_root / "map" / path) for path in va_pathlist])
    valid_ds = mkds(va_sat_pathlist, va_map_pathlist, batch_size=config.BATCH_SIZE, test=True, eq=args["eq"])
    # fmt:on

    # * 損失関数を設定
    # * モデルコンパイル
    loss = args["loss"]
    model = compile_model(
        loss=loss,
        xception=args["xception"],
        freeze_enc=args["freeze_enc"],
        freeze_dec=args["freeze_dec"],
    )

    # * 学習済み重みをロード
    if args["pretrained"] is not None:
        if not pathlib.Path(args["pretrained"]).parent.exists():
            _msg = f"{args['pretrained']} not found."
            logging.error(_msg)
            raise FileNotFoundError(_msg)
        ret = model.load_weights(args["pretrained"])
        logging.info(f'{args["pretrained"]}, {ret}')

    # * 訓練ループ
    model_history = model.fit(
        train_ds,
        epochs=args["epochs"],
        validation_data=valid_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=va_steps,
        callbacks=get_callbacks(args["logdir"]),
    )
    logging.info(f"model: {model}")
    logging.info(f"hist : {model_history}")
    return model, model_history


if __name__ == "__main__":
    import argparse

    args = getargs()

    exec_log_dir = config.RES_BASE / "exec_log"
    exec_log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %I:%M:%S",
        filename=f"{str(exec_log_dir/args.logdir)}.log",
    )

    # * NameSpaceをKWArgsにする
    main(**vars(args))
