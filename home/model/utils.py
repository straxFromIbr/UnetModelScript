import tensorflow as tf
import tensorflow.keras as keras


def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    upconv = keras.Sequential()
    upconv.add(
        keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if norm_type.lower() == "batchnorm":
        upconv.add(keras.layers.BatchNormalization())

    if apply_dropout:
        upconv.add(keras.layers.Dropout(0.5))

    upconv.add(keras.layers.ReLU())

    return upconv


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def build_down_stack(input_shape: tuple):
    weights = None
    # weights = "imagenet"
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    # Use the activations of these layers
    base_model_outputs = [
        base_model.get_layer("block_1_expand_relu").output,
        base_model.get_layer("block_3_expand_relu").output,
        base_model.get_layer("block_6_expand_relu").output,
        base_model.get_layer("block_13_expand_relu").output,
        base_model.get_layer("block_16_project").output,
    ]

    # Create the feature extraction model
    down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    # down_stack.trainable = False
    return down_stack


def build_up_stack():
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]
    return up_stack
