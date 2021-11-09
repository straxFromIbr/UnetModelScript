import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from .display import create_mask

from IPython.display import clear_output


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, sample_inp, sample_tar):
        super().__init__()
        self.sample_inp = sample_inp
        self.sample_tar = sample_tar
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        pred_mask = create_mask(self.model.predict(self.sample_inp[tf.newaxis, ...]))
        plt.imshow(keras.preprocessing.image.array_to_img(self.sample_inp))
        plt.show()
        plt.imshow(keras.preprocessing.image.array_to_img(pred_mask))
        plt.show()
        plt.imshow(keras.preprocessing.image.array_to_img(self.sample_tar))
        plt.show()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))
