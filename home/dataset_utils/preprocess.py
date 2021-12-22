import tensorflow as tf
#import tensorflow_addons as tfa


def resize(input_image, height, width):
    input_image = tf.image.resize(
        input_image, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image


def eqhist(input_image):
    """
    * dtypeがintでなくてはならない
    """
    eq_image = tfa.image.equalize(input_image)
    return eq_image


def load_image(path, channels, height, width, eq=False):
    # ファイルから読み込み
    image_f = tf.io.read_file(path)
    # Tensorに変換
    image_jpeg = tf.image.decode_jpeg(image_f, channels=channels)
    if eq:
        image_jpeg = eqhist(image_jpeg)
    # 正規化([0,1])
    _normer = tf.constant(255.0, dtype=tf.float32)
    image_tensor = tf.cast(image_jpeg, tf.float32) / _normer
    # リサイズ
    image = resize(image_tensor, height=height, width=width)
    return image
