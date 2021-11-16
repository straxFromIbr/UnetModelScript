# %%
import sys
from datetime import datetime

import tensorflow.keras as keras

from model import unet, losses, metrics

import config
from utils import callbacks
from dataset_utils import mk_dataset


# %%
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


# %%
# Define Model
def compile_model(loss, optimizer):
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CH)
    model = unet.big_unet_model(
        input_shape=input_shape,
        output_channels=config.OUT_CH,
    )
    # Compile the model
    metric_list = ["accuracy", metrics.iou_coef]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    return model


# %%
# Show Model shape


def get_callbacks(filename):
    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename))
    checkpoint_cb = callbacks.get_checkpoint_callback(
        str(config.CHECKPOINT_PATH / filename)
    )
    callback_list = [tboard_cb, checkpoint_cb]
    return callback_list


# %%
def train(train_ds, valid_ds, NB_EPOCHS, loss, optimizer=keras.optimizers.Adam()):
    model = compile_model(loss=loss, optimizer=optimizer)
    filename = datetime.now().strftime("%y%m%d%H_") + model.loss.name
    model_history = model.fit(
        train_ds,
        epochs=NB_EPOCHS,
        validation_data=valid_ds,
        # steps_per_epoch=config.STEPS_PER_EPOCH,
        # validation_steps=config.VALIDATION_STEPS,
        callbacks=get_callbacks(filename),
    )
    model.save(str(config.MODEL_SAVE_PATH / filename))
    return model_history


# %%
def main():
    lossfunc_dict = {
        "DICE": losses.DICELoss(name="DICE"),
        "FOCAL": losses.FocalTverskyLoss("Focal"),
        "TVERSKY": losses.TverskyLoss("Tversky"),
    }
    train_ds, valid_ds = make_datasets()
    loss = lossfunc_dict[sys.argv[1]]
    hist = train(
        train_ds=train_ds,
        valid_ds=valid_ds,
        NB_EPOCHS=500,
        loss=loss,
    )


# %%
if __name__ == "__main__":
    main()
