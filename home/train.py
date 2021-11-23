import argparse
import pathlib
import sys

import tensorflow.keras as keras

import config
from dataset_utils import mk_dataset
from model import losses, unet
from utils import callbacks


# Get Datasets
def make_datasets(use_cumix: bool, nbmix: int = 3):
    """
    データセット作成。`use_cutmix`でCutmix適用を決める。
    """
    train_ds = mk_dataset.mk_base_dataset(config.TR_SAT_PATH, config.TR_MAP_PATH)
    if use_cumix:
        train_ds = mk_dataset.augument_ds(train_ds, nbmix)
    train_ds = mk_dataset.post_process_ds(train_ds)

    valid_ds = mk_dataset.mk_base_dataset(config.VA_SAT_PATH, config.VA_MAP_PATH)
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

    args = parser.parse_args()

    return args


# # %%
# def train(model: keras.Model, train_ds, valid_ds, NB_EPOCHS):
#     filename = model.loss.name
#     model_history = model.fit(
#         train_ds,
#         epochs=NB_EPOCHS,
#         validation_data=valid_ds,
#         steps_per_epoch=config.STEPS_PER_EPOCH,
#         validation_steps=5,
#         callbacks=get_callbacks(filename),
#     )
#     return model_history


if __name__ == "__main__":
    args = getargs()
    train_ds, valid_ds = make_datasets(use_cumix=args.use_cutmix, nbmix=args.nbmix)
    print(args.epochs)

    loss = losses.TverskyLoss(name="Tversky", alpha=args.alpha)
    model = compile_model(loss=loss)
    ## 訓練
    filename = args.logdir
    model_history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=valid_ds,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=5,
        callbacks=get_callbacks(filename),
    )
