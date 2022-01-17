import functools
from typing import Optional

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class ResidualBlock(keras.layers.Layer):
    """
    Pre-Activation Residual Block
    WideResNetで提唱
    INPUT -> [BN -> ReLU -> Conv -> BN -> ReLU -> Conv] -> Add -> OUTPUT
    """

    def __init__(
        self,
        filters,
        name,
        kernel=3,
        stride=1,
        apply_dropout=True,
        trainable=True,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
        )
        # * Layer Configuration
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.apply_dropout = apply_dropout

        self.pretrained = False

        # * ShortCut
        self.sc_conv2d = layers.Conv2D(
            filters,
            1,
            strides=stride,
            kernel_initializer="he_normal",
            use_bias=False,
            name=name + "_sc_conv",
        )
        self.sc_bn = layers.BatchNormalization(name=name + "_sc_bn")
        base_conv2d = functools.partial(
            layers.Conv2D,
            filters=filters,
            kernel_size=kernel,
            padding="same",
            strides=stride,
            kernel_initializer="he_normal",
        )

        # * First ConvBlock
        self.bn1 = layers.BatchNormalization(name=name + "_1_bn")
        self.actv1 = layers.Activation(tf.nn.relu, name=name + "_1_actv")
        self.conv2d1 = base_conv2d(name=name + "_1_conv")

        # * Second Conv Block
        self.bn2 = layers.BatchNormalization(name=name + "_2_bn")
        self.actv2 = layers.Activation(tf.nn.relu, name=name + "_2_actv")
        self.dropout2 = layers.Dropout(0.2, name=name + "_2_dropout")
        self.conv2d2 = base_conv2d(name=name + "_2_conv")

        # * shortcut (local skip connection)
        self.add = layers.Add(name=name + "_add")

    def call(self, inputs, *args, **kwargs):
        # * shortcut
        shortcut = self.sc_conv2d(inputs)
        shortcut = self.sc_bn(shortcut)

        # * Pre Activation Residual Block
        # * First ConvBlock
        x = inputs
        x = self.bn1(x)
        x = self.actv1(x)
        x = self.conv2d1(x)

        # * Second ConvBlock
        x = self.bn2(x)
        x = self.actv2(x)
        if self.apply_dropout:
            x = self.dropout2(x)

        x = self.conv2d2(x)

        # * Add Residual Conncetion
        x = self.add([shortcut, x])
        output = x
        return output

    def apply_pretrained_weights(
        self,
        base_conv1: Optional[layers.Layer] = None,
        base_conv2: Optional[layers.Layer] = None,
    ):

        self.pretrained = True
        if base_conv1 is not None:
            self.conv2d1.set_weights(base_conv1.get_weights())
        if base_conv2 is not None:
            self.conv2d2.set_weights(base_conv2.get_weights())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel": self.kernel,
                "stride": self.stride,
                "apply_dropout": self.apply_dropout,
            }
        )
        return config


class ParallelDilatedConvBlock(keras.layers.Layer):
    """
    https://iopscience.iop.org/article/10.1088/1742-6596/1345/5/052066
    から引用
    """

    def __init__(
        self,
        filters,
        name,
        kernel_size=3,
        num_conv=6,
        bottleneck=True,
        trainable=True,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.bottleneck = bottleneck

        self.bottleneck_list = []
        self.conv_list = []
        self.add_list = []
        for d in range(num_conv):
            drate = 2 ** d

            bottleneck_layer = layers.Conv2D(1, 1, name=name + f"_bnconv{d}")
            self.bottleneck_list.append(bottleneck_layer)

            conv_layer = layers.Conv2D(
                filters,
                kernel_size,
                dilation_rate=drate,
                padding="same",
                kernel_initializer="he_normal",
                name=name + f"_conv_dl{drate}",
            )
            self.conv_list.append(conv_layer)

            add_layer = layers.Add(name=name + f"_add{d}")
            self.add_list.append(add_layer)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        first = True
        for bnconv, conv, add in zip(
            self.bottleneck_list, self.conv_list, self.add_list
        ):
            prev_x = x
            if self.bottleneck:
                x = bnconv(x)
            x = conv(x)
            if not first:
                x = add([prev_x, x])
            first = False
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "num_conv": self.num_conv,
                "bottleneck": self.bottleneck,
            }
        )

        return config


def last_stack(inputs, n):
    x = inputs
    residual_block_out = ResidualBlock(n, name="output_res")
    x = residual_block_out(x)
    conv = layers.Conv2D(
        filters=1,
        kernel_size=3,
        activation=tf.nn.sigmoid,
        padding="same",
        name="output_conv",
    )
    x = conv(x)
    return x


def unet(
    input_shape,
    name,
    nb_layer=4,
    kernel_size=3,
    initial_channels=32,
    parallel_dilated=False,
    freeze_enc=False,
    freeze_dec=False,
):
    inputs = layers.Input(input_shape, name="input")
    x = inputs

    # * ========== Encoder ==============
    down_stacks = []
    for l in range(nb_layer):
        name = f"downstack{l}"
        filters = 2 ** l * initial_channels

        residual_block = ResidualBlock(
            filters,
            apply_dropout=False,
            name=name,
            trainable=not freeze_enc,
            kernel=kernel_size,
        )
        x = residual_block(x)
        down_stacks.append(x)

        max_pool = layers.MaxPool2D(name=name + "_pool")
        x = max_pool(x)

    # * ========= Bottom EncDec-Path ==========
    if parallel_dilated:
        # * use parallel dialated module
        print(2 ** nb_layer * initial_channels)
        bottom_layer = ParallelDilatedConvBlock(
            filters=2 ** nb_layer * initial_channels, name="bottom"
        )
    else:
        # * use normal residual block
        bottom_layer = ResidualBlock(
            filters=2 ** nb_layer,
            name="bottom",
            kernel=kernel_size,
        )
    x = bottom_layer(x)

    # * ========== Decoder ==============
    for l, down in zip(range(nb_layer, 0, -1), down_stacks[::-1]):
        name = f"upstack{l}"
        filters = 2 ** l * initial_channels

        conv_trans = layers.Conv2DTranspose(
            filters / 2, 2, strides=2, name=name + "_transconv"
        )
        x = conv_trans(x)

        # * Copy and Concat
        x = layers.Concatenate(name=name + "_concat")((x, down))

        residual_block = ResidualBlock(
            filters,
            name=name + "_res",
            trainable=not freeze_dec,
            kernel=kernel_size,
        )
        x = residual_block(x)


    # * Last block
    x = last_stack(x, initial_channels)

    model = keras.Model(inputs, x, name=name)
    return model


if __name__ == "__main__":
    import numpy as np

    input_shape = (256, 256, 3)
    model = unet(
        (256, 256, 3),
        nb_layer=4,
        initial_channels=32,
        parallel_dilated=True,
        kernel_size=3,
        name="unet",
    )
    model.summary()
    print(model.predict(np.zeros(input_shape)[None]).shape)
