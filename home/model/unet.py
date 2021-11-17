import tensorflow as tf
import tensorflow.keras as keras

from . import utils


def big_unet_model(input_shape=(256, 256, 3), output_channels=1):
    inputs = keras.layers.Input(shape=input_shape)

    # initializer = tf.random_normal_initializer(0.0, 0.02)
    x = inputs

    # Downsampling through the model
    down_stack = [
        utils.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        utils.downsample(128, 4),  # (batch_size, 64, 64, 128)
        utils.downsample(256, 4),  # (batch_size, 32, 32, 256)
        utils.downsample(512, 4),  # (batch_size, 16, 16, 512)
        utils.downsample(512, 4),  # (batch_size, 8, 8, 512)
        utils.downsample(512, 4),  # (batch_size, 4, 4, 512)
        utils.downsample(512, 4),  # (batch_size, 2, 2, 512)
        utils.downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    up_stack = [
        utils.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        utils.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        utils.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        utils.upsample(512, 4),  # (batch_size, 16, 16, 1024)
        utils.upsample(256, 4),  # (batch_size, 32, 32, 512)
        utils.upsample(128, 4),  # (batch_size, 64, 64, 256)
        utils.upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat_l = keras.layers.Concatenate()
        x = concat_l([x, skip])

    # # laststacks
    # last_stacks = [
    #     keras.layers.Conv2D(64, (3, 3), padding="same"),
    #     keras.layers.Activation("relu"),
    #     keras.layers.Conv2D(64, (3, 3), padding="same"),
    #     keras.layers.BatchNormalization(axis=3),
    #     keras.layers.Activation("relu"),
    #     keras.layers.Conv2D(output_channels, (1, 1), activation="sigmoid"),
    # ]
    last = keras.layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding="same",
        # kernel_initializer=initializer,
        activation="sigmoid",
    )
    x = last(x)
    # for layer in last_stacks:
    #     x = layer(x)
    return keras.Model(inputs=inputs, outputs=x)


def unet_mobv1_model(input_shape: tuple, output_channels: int):
    inputs = keras.layers.Input(shape=input_shape)

    down_stack = utils.build_down_stack(input_shape)
    up_stack = utils.build_up_stack()

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = keras.layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding="same",
    )  # 64x64 -> 128x128

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    OUTPUT_CLASSES = 2

    input_shape = (128, 128, 3)
    input_shape = (512, 512, 3)

    # model = unet_model(input_shape, output_channels=OUTPUT_CLASSES)
    model = big_unet_model()
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    keras.utils.plot_model(
        model,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
