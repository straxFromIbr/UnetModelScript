import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(keras.layers.Layer):
    def __init__(
        self,
        filters,
        name,
        output_fn=tf.nn.relu,
        kernel=3,
        stride=1,
        bottleneck=False,
        trainable=True,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )
        self.bottleneck = bottleneck

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

        # * First ConvBlock
        self.conv2d1 = layers.Conv2D(
            filters,
            kernel,
            padding="same",
            strides=stride,
            kernel_initializer="he_normal",
            name=name + "_1_conv",
        )
        self.bn1 = layers.BatchNormalization(name=name + "_1_bn")
        self.actv1 = layers.Activation(tf.nn.relu, name=name + "_1_relu")

        # * BottleNeck Conv
        self.conv2d_bn = layers.Conv2D(
            1,
            1,
            padding="same",
            strides=stride,
            kernel_initializer="he_normal",
            name=name + "_bn_conv",
        )

        # * Second Conv Block
        self.conv2d2 = layers.Conv2D(
            filters,
            kernel,
            padding="same",
            strides=stride,
            kernel_initializer="he_normal",
            name=name + "_2_conv",
        )
        self.bn2 = layers.BatchNormalization(name=name + "_2_bn")

        # * shortcut (local skip connection)
        self.add = layers.Add(name=name + "_add")

        # * output
        self.actv2 = layers.Activation(output_fn, name=name + "_2_relu")

    def call(self, inputs, *args, **kwargs):
        # * shortcut
        shortcut = self.sc_conv2d(inputs)
        shortcut = self.sc_bn(shortcut)

        # * First ConvBlock
        x = self.conv2d1(inputs)
        x = self.bn1(x)
        x = self.actv1(x)

        if self.bottleneck:
            # * Apply BottleNeck Conv
            x = self.conv2d_bn(x)

        # * Second ConvBlock
        x = self.conv2d2(x)
        x = self.bn2(x)

        # * Add Residual Conncetion
        x = self.add([shortcut, x])
        # * Activation of ConvBlock
        output = self.actv2(x)
        return output


class BottleNeckPath(keras.layers.Layer):
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
        self.bottleneck = bottleneck

        self.bottleneck_list = []
        self.conv_list = []
        self.add_list = []
        for d in range(num_conv):
            drate = 2 ** d

            bottleneck_layer = layers.Conv2D(1, 1, name=f"_bnconv{d}")
            self.bottleneck_list.append(bottleneck_layer)

            conv_layer = layers.Conv2D(
                filters,
                kernel_size,
                dilation_rate=drate,
                padding="same",
                name=f"_conv_dl{drate}",
            )
            self.conv_list.append(conv_layer)

            add_layer = layers.Add(name=f"_add{d}")
            self.add_list.append(add_layer)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for bnconv, conv, add in zip(
            self.bottleneck_list, self.conv_list, self.add_list
        ):
            prev_x = x
            if self.bottleneck:
                x = bnconv(x)
            x = conv(x)
            x = add([prev_x, x])
        return x


# output = layers.Conv2DTranspose(filters / 2, 2, strides=2, name=name + "_trans")(x)
def unet(input_shape, name, nb_layer=4, n=32, freeze_enc=False, freeze_dec=False):
    input = layers.Input(input_shape, name="input")
    x = input
    down_stacks = []
    # * Encoder
    for l in range(nb_layer):
        name = f"downstack{l}"
        filters = 2 ** l * n

        residual_block = ResidualBlock(filters, name=name, trainable=not freeze_enc)
        x = residual_block(x)
        down_stacks.append(x)

        max_pool = layers.MaxPool2D(name=name + "_pool")
        x = max_pool(x)

    # * Decoder
    for l, down in zip(range(nb_layer, 0, -1), down_stacks[::-1]):
        name = f"upstack{l}"
        filters = 2 ** l * n

        residual_block = ResidualBlock(
            filters, name=name + "_res", trainable=not freeze_dec
        )
        x = residual_block(x)

        conv_trans = layers.Conv2DTranspose(
            filters / 2, 2, strides=2, name=name + "_transconv"
        )
        x = conv_trans(x)

        # * Copy and Concat
        x = layers.Concatenate(name=name + "_concat")([x, down])

    residual_block_out = ResidualBlock(1, name="output", output_fn=tf.nn.sigmoid)
    x = residual_block_out(x)

    model = keras.Model(input, x, name=name)
    return model
