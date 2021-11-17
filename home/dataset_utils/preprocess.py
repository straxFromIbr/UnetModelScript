from numpy.lib.shape_base import dsplit
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing

import config


def resize(input_image, height, width):
    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return input_image


def normalize(input_image):
    # Normalizing the images to [0, 1]
    input_image = input_image / 255.0
    return input_image


def load_image(path, channels):
    # ファイルから読み込み
    image_f = tf.io.read_file(path)
    image_jpeg = tf.image.decode_jpeg(image_f, channels=channels)
    # Tensorに変換、正規化([0,1])
    _normer = tf.constant(255.0, dtype=tf.float32)
    image_tensor = tf.cast(image_jpeg, tf.float32) / _normer
    # リサイズ
    image = resize(image_tensor, config.IMG_HEIGHT, config.IMG_WIDTH)
    return image


def load_image_gray(image):
    image_f = tf.io.read_file(image)
    image_jpeg = tf.image.decode_jpeg(image_f, channels=1)
    image_tensor = tf.cast(image_jpeg, tf.float32)
    return image_tensor


# def preprocess_images(*images):
#     processed_images_list = []
#     for image in images:
#         image = resize(image, IMG_HEIGHT, IMG_WIDTH)
#         image = normalize(image)
#         processed_images_list.append(image)
#     return tuple(processed_images_list)


def normalize_map(input_image):
    # [0, 1] to [1,0]
    input_image = (input_image / -255) + 1
    return input_image


def preprocess_image(image):
    image = resize(image, config.IMG_HEIGHT, config.IMG_WIDTH)
    image = normalize(image)
    return image


def preprocess_image_map(image):
    image = resize(image, config.IMG_HEIGHT, config.IMG_WIDTH)
    image = normalize_map(image)
    return image


class Augment(keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf_preprocessing.RandomFlip(
            mode="horizontal_and_vertical", seed=seed
        )
        self.augment_labels = tf_preprocessing.RandomFlip(
            mode="horizontal_and_vertical", seed=seed
        )

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
