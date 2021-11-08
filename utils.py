import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from IPython.display import clear_output


def generate_images(test_input, tar):
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0]]
    title = ["Input Image", "Ground Truth"]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            generate_images(image[0], create_mask(pred_mask))


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
