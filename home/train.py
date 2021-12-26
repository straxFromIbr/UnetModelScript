import logging
import pathlib

import tensorflow.keras as keras

import config
from dataset_utils import auguments
from dataset_utils.mk_dataset import mk_pathlist, mkds
from model import losses, unet
from utils import callbacks


# Define model
def compile_model(
    loss,
    freeze_enc=False,
    freeze_dec=False,
):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE,
        output_channels=config.OUT_CH,
        freeze_enc=freeze_enc,
        freeze_dec=freeze_dec,
    )

    # Compile the model
    optimizer = keras.optimizers.Adam()
    metric_list = [keras.metrics.MeanIoU(num_classes=2)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    return model


# Define Callbacks
def get_callbacks(filename):
    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename),)
    checkpoint_path = str(config.CHECKPOINT_PATH / filename / filename) + "_{epoch:03d}"

    checkpoint_cb = callbacks.get_checkpoint_callback(checkpoint_path)
    callback_list = [tboard_cb, checkpoint_cb]
    return callback_list


def getargs():
    # fmt:off
    parser = argparse.ArgumentParser(description="U-Netによる道路検出。CutMix")
    parser.add_argument("--logdir", help="ログ/チェックポイントのパス", type=str, required=True)

    # *モデル設定
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
    ds_root = pathlib.Path(args["datadir"])

    (
        ((tr_sat_pathlist, tr_map_pathlist), tr_steps),
        ((va_sat_pathlist, va_map_pathlist), va_steps),
    ) = mk_pathlist(
        ds_root, args["suffix"], batch_size=args["batch_size"], dsrate=args["dsrate"]
    )

    train_ds = mkds(
        tr_sat_pathlist,
        tr_map_pathlist,
        augment=auguments.Augment(),
        batch_size=args["batch_size"],
        cutmix=args["use_cutmix"],
        nbmix=args["nbmix"],
    )

    valid_ds = mkds(
        va_sat_pathlist, va_map_pathlist, batch_size=args["batch_size"], test=True
    )
    # fmt:on

    # * 損失関数を設定
    # * モデルコンパイル
    loss = args["loss"]
    model = compile_model(
        loss=loss,
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
        steps_per_epoch=tr_steps,
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
