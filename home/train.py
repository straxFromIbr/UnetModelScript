from typing import List
import functools
import pathlib
import random
import math

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.generic_utils import default

import config
from dataset_utils import preprocess, auguments
from model import losses, unet
from utils import callbacks


def mkds(
    sat_path_list: List[str],
    map_path_list: List[str],
    batch_size: int,
    aug: bool = False,
    zoom: bool = True,
    flip: bool = True,
    rotate: bool = True,
):

    # 衛星画像のデータセット
    image_loader_rgb = functools.partial(preprocess.load_image, channels=3)
    sat_path_list_ds = tf.data.Dataset.list_files(sat_path_list, shuffle=False)
    sat_dataset = sat_path_list_ds.map(
        image_loader_rgb, num_parallel_calls=tf.data.AUTOTUNE
    )

    # 地図のデータセット
    image_loader_gry = functools.partial(preprocess.load_image, channels=1)
    map_path_list_ds = tf.data.Dataset.list_files(map_path_list, shuffle=False)
    map_dataset = map_path_list_ds.map(
        image_loader_gry, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Zipして、batchしてaugmentする。並列処理とprefetch。
    # cacheはメモリ不足に陥るので適用しない
    sat_map = (
        tf.data.Dataset.zip((sat_dataset, map_dataset))
        .shuffle(config.BUFFER_SIZE)
        .repeat()
        .batch(batch_size)
    )
    if aug:
        sat_map = sat_map.map(
            auguments.Augment(zoom=zoom, flip=flip, rotate=rotate),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    sat_map = sat_map.prefetch(tf.data.AUTOTUNE)
    return sat_map


# Define model
def compile_model(loss, xception=False):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE, output_channels=config.OUT_CH
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
    checkpoint_path = str(
        config.CHECKPOINT_PATH / filename / filename
    )  # + "_{epoch:03d}"
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
    pathlist = pathlib.Path(args["datadir"]).glob(f"**/*.{args['suffix']}")
    pathlist = sorted([path.name for path in pathlist])
    random.shuffle(pathlist)
    print(pathlist[:10])

    # * 訓練用と検証用に分割

    nb_tr = int(len(pathlist) * 0.8 * args["dsrate"])
    nb_va = int(len(pathlist) * 0.2 * args["dsrate"])
    tr_pathlist = pathlist[:nb_tr]
    va_pathlist = pathlist[nb_tr : nb_tr + nb_va]

    # * ステップ数を決める
    steps_per_epoch = math.ceil(nb_tr / config.BATCH_SIZE)
    va_steps = math.ceil(nb_va / config.BUFFER_SIZE)

    # * tf.Dataの作成
    ds_root = pathlib.Path(args["datadir"])

    # fmt:off
    tr_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in tr_pathlist])
    tr_map_pathlist = sorted([str(ds_root / "map" / path) for path in tr_pathlist])
    train_ds = mkds(tr_sat_pathlist, tr_map_pathlist, batch_size=config.BATCH_SIZE, aug=args["augment"])

    va_sat_pathlist = sorted([str(ds_root / "sat" / path) for path in va_pathlist])
    va_map_pathlist = sorted([str(ds_root / "map" / path) for path in va_pathlist])
    valid_ds = mkds(va_sat_pathlist, va_map_pathlist, batch_size=config.BATCH_SIZE)
    # fmt:on

    # * 損失関数を設定
    loss = args["loss"]

    # * モデルコンパイル
    model = compile_model(loss=loss, xception=args["xception"])

    # * 学習済み重みをロード
    if args["pretrained"] is not None:
        if not pathlib.Path(args["pretrained"]).parent.exists():
            raise FileNotFoundError(f"{args['pretrained']} not found.")
        ret = model.load_weights(args["pretrained"])

    # * 訓練ループ
    model_history = model.fit(
        train_ds,
        epochs=args["epochs"],
        validation_data=valid_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=va_steps,
        callbacks=get_callbacks(args["logdir"]),
    )
    return model, model_history


if __name__ == "__main__":
    import argparse

    args = getargs()
    # * NameSpaceをKWArgsにする
    main(**vars(args))
