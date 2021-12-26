import config
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


class Augment(layers.Layer):
    def __init__(
        self,
        zoom_rate: float = 0.2,
        flip_mode: str = "horizontal_and_vertical",
        rotate_rate: float = 0.2,
        trans_rate: float = 0.2,
    ):
        super().__init__()

        seed = 99
        interpolation = "nearest"

        # fmt:off
        self.i_zoom = preprocessing.RandomZoom(zoom_rate, interpolation='bilinear', seed=seed)
        self.i_flip = preprocessing.RandomFlip(flip_mode, seed=seed)
        self.i_rotate = preprocessing.RandomRotation(rotate_rate, interpolation=interpolation, seed=seed)
        self.i_trans = preprocessing.RandomTranslation(trans_rate, trans_rate, interpolation=interpolation, seed=seed)
        # fmt:on
        
        # fmt:off
        self.t_zoom = preprocessing.RandomZoom(zoom_rate, interpolation='bilinear', seed=seed)
        self.t_flip = preprocessing.RandomFlip(flip_mode, seed=seed)
        self.t_rotate = preprocessing.RandomRotation(rotate_rate,  interpolation=interpolation, seed=seed)
        self.t_trans = preprocessing.RandomTranslation(trans_rate, trans_rate,interpolation=interpolation, seed=seed)
        # fmt:on

    def call(self, inputs, labels):

        inputs = self.i_zoom(inputs)
        inputs = self.i_flip(inputs)
        inputs = self.i_rotate(inputs)
        inputs = self.i_trans(inputs)

        labels = self.t_zoom(labels)
        labels = self.t_flip(labels)
        labels = self.t_rotate(labels)
        labels = self.t_trans(labels)

        return inputs, labels


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# Impplement CUTMIX

# @tf.function
def mixed_img(image1, image2, *bbox):
    # fmt:off
    boundary_x1, boundary_y1, target_h, target_w = bbox
    # image2からパッチを切り出す
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundary_y1, boundary_x1, target_h, target_w)
    # crop2のオフセットでパディング
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundary_y1, boundary_x1, config.IMG_SIZE, config.IMG_SIZE)

    # image1からパッチを切り出す
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundary_y1, boundary_x1, target_h, target_w)
    # crop1のオフセットでパディング
    image1_pad = tf.image.pad_to_bounding_box(
        crop1, boundary_y1, boundary_x1, config.IMG_SIZE, config.IMG_SIZE)

    image1 = image1 - image1_pad
    image = image1 + image2
    return image
    # fmt:on


@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    # rw
    cut_w = config.IMG_SIZE * cut_rat
    cut_w = tf.cast(cut_w, tf.int32)

    # rh
    cut_h = config.IMG_SIZE * cut_rat
    cut_h = tf.cast(cut_h, tf.int32)

    # rx & ry
    cut_x = tf.random.uniform((1,), minval=0, maxval=config.IMG_SIZE, dtype=tf.int32)
    cut_y = tf.random.uniform((1,), minval=0, maxval=config.IMG_SIZE, dtype=tf.int32)

    boundary_x1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, config.IMG_SIZE)
    boundary_y1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, config.IMG_SIZE)
    bb_x2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, config.IMG_SIZE)
    bb_y2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, config.IMG_SIZE)

    target_h = bb_y2 - boundary_y1
    if target_h == 0:
        target_h += 1

    target_w = bb_x2 - boundary_x1
    if target_w == 0:
        target_w += 1
    return boundary_x1, boundary_y1, target_h, target_w


@tf.function
def cutmix(*ds):
    (
        (ip_image1, tar_image1),
        (ip_image2, tar_image2),
        (ip_image2, tar_image2),
        (ip_image2, tar_image2),
    ) = ds
    alpha = [0.25]
    beta = [0.25]

    # ベータ分布から値採取
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Bボックスの高さと幅のオフセット取得
    lambda_value = lambda_value[0][0]
    bbox = get_box(lambda_value)
    ip_image = mixed_img(ip_image1, ip_image2, *bbox)
    tar_image = mixed_img(tar_image1, tar_image2, *bbox)

    return ip_image, tar_image


# @tf.function
def mkmask(bbox, c=3):
    boundary_x1, boundary_y1, target_h, target_w = bbox
    mask = tf.image.pad_to_bounding_box(
        tf.ones((target_h, target_w, c)),
        boundary_y1,
        boundary_x1,
        config.IMG_SIZE,
        config.IMG_SIZE,
    )
    return mask


@tf.function
def cutmix_batch(*ds):
    inp_batch, tar_batch = ds

    alpha = [0.25]
    beta = [0.25]

    inp = inp_batch[0]
    tar = tar_batch[0]
    res_inp_batch = tf.zeros_like(inp_batch)
    res_tar_batch = tf.zeros_like(tar_batch)
    for base_idx in tf.range(len(inp_batch)):
        inp = inp_batch[base_idx]
        tar = tar_batch[base_idx]
        for patch_idx in tf.range(len(inp_batch)):
            # Bボックスの高さと幅のオフセット取得
            lambda_value = sample_beta_distribution(1, alpha, beta)[0][0]
            bbox = get_box(lambda_value)

            # 入力
            inp_patch = inp_batch[patch_idx]
            inp_mask = mkmask(bbox, config.IMG_CH)
            inp = inp * (1 - inp_mask) + inp_patch * inp_mask

            # ターゲット
            tar_patch = tar_batch[patch_idx]
            tar_mask = mkmask(bbox, config.OUT_CH)
            tar = tar * (1 - tar_mask) + tar_patch * tar_mask

        res_inp_batch = tf.concat(
            [
                res_inp_batch[:base_idx],
                tf.expand_dims(inp, axis=0),
                res_inp_batch[base_idx + 1 :],
            ],
            axis=0,
        )
        res_tar_batch = tf.concat(
            [
                res_tar_batch[:base_idx],
                tf.expand_dims(tar, axis=0),
                res_tar_batch[base_idx + 1 :],
            ],
            axis=0,
        )
    return res_inp_batch, res_tar_batch
