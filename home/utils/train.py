import pathlib
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
from dataset_utils import mk_dataset
from model import losses, residual_unet
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def build_model(pret=None, freeze_enc=False, freeze_dec=False):
    input_shape = (256, 256, 3)
    model = residual_unet.unet(
        input_shape,
        name="unet",
        parallel_dilated=True,
    )
    if pret is not None:
        model = keras.models.load_model(pret)

    for layer in model.layers:
        if "down" in layer.name and freeze_enc:
            layer.trainable = False
        if "up" in layer.name and freeze_dec:
            layer.trainable = False

    optimizer = keras.optimizers.Adam(learning_rate=0.01, name="adam")
    loss = losses.DICELoss(name="dice")
    # loss = losses.TverskyLoss(alpha=0.5, name='Tversky')
    metrics = (
        keras.metrics.MeanIoU(num_classes=2, name='mean_iou'),
        keras.metrics.Precision(name='presision'),
        keras.metrics.Recall(name='recall'),
    )

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def gends(dsrate=1.0, geo_aug=True, ch_aug=True,):
    ds_root = pathlib.Path("/mass_roads9/train")
    suffix = "png"

    batch_size = 32

    augment = None

    aug_param_dic = {}
    if geo_aug:
        aug_param_dic.update({
            "flip_mode": "horizontal_and_vertical",
            "zoom_rate": (-0.3, 0),
            "rotate_rate": 0.2,
        })

    if ch_aug:
        aug_param_dic.update({
            "brightness": 0.2,
            "contrast": 0.2,
        })

    if geo_aug or ch_aug:
        augment = Augment(**aug_param_dic)

    (
        ((tr_sat_pathlist, tr_map_pathlist), tr_steps),
        ((va_sat_pathlist, va_map_pathlist), va_steps),
    ) = mk_dataset.mk_pathlist(
        ds_root, suffix, batch_size=batch_size, dsrate=dsrate
    )

    train_ds = mk_dataset.mkds(
        tr_sat_pathlist,
        tr_map_pathlist,
        augment=augment,
        batch_size=batch_size,
    )

    valid_ds = mk_dataset.mkds(
        va_sat_pathlist,
        va_map_pathlist,
        batch_size=batch_size,
        test=True,
    )
    return (train_ds, tr_steps), (valid_ds, va_steps)
class Augment(layers.Layer):
    def __init__(
        self,
        zoom_rate: Optional[Tuple[float,float]]=None,
        flip_mode: Optional[str]=None,
        rotate_rate: Optional[float]=None,
        trans_rate:  Optional[float]=None,
        hue:         Optional[float]=None,
        brightness:  Optional[float]=None,
        contrast:    Optional[float]=None,

    ):
        super().__init__()

        seed = 99
        interpolation = "nearest"

        self.hue = hue
        self.brightness = brightness

        self.inputs_augment = keras.models.Sequential(name='inputs_augment')
        self.labels_augment = keras.models.Sequential(name='labels_augment')
        # fmt:off
        if zoom_rate is not None:
            self.inputs_augment.add(preprocessing.RandomZoom(zoom_rate, interpolation='bilinear', seed=seed))
            self.labels_augment.add(preprocessing.RandomZoom(zoom_rate, interpolation='bilinear', seed=seed))

        if flip_mode is not None:
            self.inputs_augment.add(preprocessing.RandomFlip(flip_mode, seed=seed))
            self.labels_augment.add(preprocessing.RandomFlip(flip_mode, seed=seed))

        if rotate_rate is not None:
            self.inputs_augment.add(preprocessing.RandomRotation(rotate_rate,  interpolation=interpolation, seed=seed))
            self.labels_augment.add(preprocessing.RandomRotation(rotate_rate,  interpolation=interpolation, seed=seed))

        if trans_rate is not None:
            self.inputs_augment.add(preprocessing.RandomTranslation(trans_rate, trans_rate, interpolation=interpolation, seed=seed))
            self.labels_augment.add(preprocessing.RandomTranslation(trans_rate, trans_rate, interpolation=interpolation, seed=seed))

        if contrast is not None:
            self.inputs_augment.add(preprocessing.RandomContrast(factor=contrast))
       # fmt:on

    def call(self, inputs, labels):
        inputs = self.inputs_augment(inputs)
        labels = self.labels_augment(labels)
        if self.brightness is not None:
            inputs = tf.image.random_brightness(inputs, max_delta=self.brightness)
        if self.hue is not None:
            inputs = tf.image.random_hue(inputs, max_delta=self.hue)


        return inputs, labels
