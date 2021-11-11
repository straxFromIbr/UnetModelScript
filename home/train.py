from datetime import datetime

import tensorflow.keras as keras

from model import unet, losses, metrics

import config
from utils import callbacks
from dataset import mk_dataset


def make_datasets():
    train_ds = mk_dataset.mk_dataset(
        SAT_PATH=config.TR_SAT_PATH,
        MAP_PATH=config.TR_MAP_PATH,
    )
    valid_ds = mk_dataset.mk_dataset(
        SAT_PATH=config.VA_SAT_PATH,
        MAP_PATH=config.VA_MAP_PATH,
        batch_size=1,
    )
    return train_ds, valid_ds


def compile_model(loss, optimizer):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE, output_channels=config.OUT_CH
    )
    # Compile the model
    metrics_list = [metrics.iou_coef]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
    return model


def train_model(model, NB_EPOCHS, train_ds, valid_ds):
    filename = (
        datetime.now().strftime("%Y%m%d%H%M_")
        + model.optimizer.name
        + "_"
        + model.loss.name
    )

    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename))
    checkpoint_cb = callbacks.get_checkpoint_callback(
        str(config.CHECKPOINT_PATH / filename)
    )

    model_history = model.fit(
        train_ds,
        epochs=NB_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS,
        validation_data=valid_ds,
        callbacks=[tboard_cb, checkpoint_cb],
    )
    return model_history


def main():
    lossfuncs_list = [
        losses.DICELoss(name="DICE"),
        losses.FocalTverskyLoss(name="FocalTversky"),
        losses.TverskyLoss(name="Tversky"),
    ]
    optimizers_list = [keras.optimizers.Adam(name="Adam")]

    train_ds, valid_ds = make_datasets()
    for optimizer in optimizers_list:
        for loss in lossfuncs_list:
            compiled_model = compile_model(loss=loss, optimizer=optimizer)
            hist = train_model(compiled_model, 300, train_ds, valid_ds)
            print(f"Train done: {loss.name}")


if __name__ == "__main__":
    main()
