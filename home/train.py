import sys

import tensorflow.keras as keras

import config
from model import unet, losses, metrics
from utils import callbacks
from dataset_utils import mk_dataset


# %%
# Get Datasets
def make_datasets():
    train_ds = mk_dataset.mk_base_dataset(config.TR_SAT_PATH, config.TR_MAP_PATH)
    train_ds = mk_dataset.augument_ds(train_ds)
    train_ds = mk_dataset.post_process_ds(train_ds)

    valid_ds = mk_dataset.mk_base_dataset(config.VA_SAT_PATH, config.VA_MAP_PATH)
    valid_ds = mk_dataset.post_process_ds(valid_ds)
    return train_ds, valid_ds


# %%
# Define model
def compile_model(loss):
    model = unet.big_unet_model(
        input_shape=config.INPUT_SIZE,
        output_channels=config.OUT_CH,
    )
    # Compile the model
    optimizer = keras.optimizers.Adam()
    metric_list = [metrics.iou_coef]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    return model


# %%
# Define Callbacks
def get_callbacks(filename):
    tboard_cb = callbacks.get_tboard_callback(str(config.LOG_PATH / filename))
    checkpoint_cb = callbacks.get_checkpoint_callback(
        str(config.CHECKPOINT_PATH / filename / filename)
    )
    callback_list = [tboard_cb, checkpoint_cb]
    return callback_list


# %%
def train(model: keras.Model, train_ds, valid_ds, NB_EPOCHS):
    filename = model.loss.name
    model_history = model.fit(
        train_ds,
        epochs=NB_EPOCHS,
        validation_data=valid_ds,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=5,
        callbacks=get_callbacks(filename),
    )
    model.save(str(config.MODEL_SAVE_PATH / filename))
    model.save(str(config.MODEL_SAVE_PATH / filename) + ".h5")
    return model_history


# %%

if __name__ == "__main__":
    train_ds, valid_ds = make_datasets()
    if sys.argv[1] == "Tversky":
        loss = losses.TverskyLoss(name="Tversky")
    elif sys.argv[1] == "DICE":
        loss = losses.DICELoss(name="DICE")
    elif sys.argv[1] == "Focal":
        loss = losses.FocalTverskyLoss(name="Focal")
    else:
        print("requires one argument: loss")
        sys.exit(1)

    model = compile_model(loss=loss)
    hist = train(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        NB_EPOCHS=30,
    )
