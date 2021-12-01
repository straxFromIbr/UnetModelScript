import os
from typing import List
import pathlib
import random

import tensorflow.keras as keras

import config
from dataset_utils import mk_dataset
from model import losses, unet
from utils import callbacks


# Get Datasets
def make_datasets(
    ds_root: pathlib.Path,
    tr_path: List[str],
    va_path: List[str],
    use_cutmix: bool,
    nbmix: int = 3,
):
    """
    データセット作成。`use_cutmix`でCutmix適用を決める。
    """
    ds_root = pathlib.Path(ds_root)
    train_ds = mk_dataset.mk_base_dataset(tr_path, ds_root / "sat", ds_root / "map")
    if use_cutmix:
        train_ds = mk_dataset.augument_ds(train_ds, nbmix)
    train_ds = mk_dataset.post_process_ds(train_ds)

    valid_ds = mk_dataset.mk_base_dataset(
        va_path,
        ds_root / "sat",
        ds_root / "map"
        # va_path, config.TR_SAT_PATH, config.TR_MAP_PATH
    )
    valid_ds = mk_dataset.post_process_ds(valid_ds)
    return train_ds, valid_ds


# Define model
def compile_model(loss):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE,
        output_channels=config.OUT_CH,
    )
    # Compile the model
    optimizer = keras.optimizers.Adam()
    metric_list = [keras.metrics.MeanIoU(num_classes=2)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    return model


# Define Callbacks
def get_callbacks(filename):
    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename))
    checkpoint_cb = callbacks.get_checkpoint_callback(
        str(config.CHECKPOINT_PATH / filename / filename)
    )
    callback_list = [tboard_cb, checkpoint_cb]
    return callback_list


def getargs():
    parser = argparse.ArgumentParser(description="U-Netによる道路検出。Tversky損失、CutMix")

    parser.add_argument(
        "--logdir",
        help="ログのパス",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--alpha",
        metavar="Alpha",
        help="Tversky損失のalpha",
        default=0.3,
        type=float,
    )

    parser.add_argument(
        "--epochs",
        help="エポック数",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--use_cutmix",
        help="Cutmixを使う？",
        action="store_true",
    )

    parser.add_argument(
        "--nbmix",
        help="CutMix数",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--datadir",
        help="データセットのルートパス",
        type=str,
    )

    parser.add_argument(
        "--pretrained",
        help="事前学習済みのモデル",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    return args


def main(**args):
    random.seed(1)
    pathlist = pathlib.Path(args["datadir"]).glob("*.png")
    pathlist = sorted([path.name for path in pathlist])
    random.shuffle(pathlist)

    nb_tr = int(len(pathlist) * 0.8)
    nb_va = int(len(pathlist) * 0.2)
    tr_pathlist = pathlist[:nb_tr]
    va_pathlist = pathlist[nb_tr : nb_tr + nb_va]

    train_ds, valid_ds = make_datasets(
        ds_root=args["datadir"],
        tr_path=tr_pathlist,
        va_path=va_pathlist,
        use_cutmix=args["use_cutmix"],
        nbmix=args["nbmix"],
    )
    print(args["epochs"])

    loss = losses.TverskyLoss(name="Tversky", alpha=args["alpha"])
    model = compile_model(loss=loss)

    if args["pretrained"] is not None:
        assert pathlib.Path(args["pretrained"]).parent.exists()
        ret = model.load_weights(args["pretrained"])

    checkpointpath = config.CHECKPOINT_PATH / args["logdir"] / args["logdir"]
    if checkpointpath.exists():
        print("load weights from checkpoint")
        model.load_weights(checkpointpath)
        print("done")

    # 訓練
    model_history = model.fit(
        train_ds,
        epochs=args["epochs"],
        validation_data=valid_ds,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=5,
        callbacks=get_callbacks(args["logdir"]),
    )


if __name__ == "__main__":
    import argparse

    args = getargs()
    main(**vars(args))
