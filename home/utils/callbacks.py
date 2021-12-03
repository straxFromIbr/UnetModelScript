import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from IPython.display import clear_output


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, sample_inp, sample_tar):
        super().__init__()
        self.sample_inp = sample_inp
        self.sample_tar = sample_tar
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # pred_mask = create_mask(self.model.predict(self.sample_inp[tf.newaxis, ...]))
        pred = self.model.predict(self.sample_inp[tf.newaxis, ...])
        plt.imshow(keras.preprocessing.image.array_to_img(self.sample_inp))
        plt.show()
        plt.imshow(keras.preprocessing.image.array_to_img(pred[0]))
        plt.show()
        plt.imshow(keras.preprocessing.image.array_to_img(self.sample_tar))
        plt.show()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


def get_tboard_callback(log_dir):
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )
    return tensorboard_callback


def get_checkpoint_callback(checkpoint_dir):
    # チェックポイントコールバックを作る
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        # save_freq=1,
        save_weights_only=True,
        save_best_only=True,
        # verbose=1,
    )
    return cp_callback
