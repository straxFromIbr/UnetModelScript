import tensorflow as tf
import tensorflow.keras as keras

"""
参考: https://qiita.com/ppza53893/items/8090322792e1c7f81e57
# 変更点:
    - パラメータ調整を容易にするためにクラスで実装
    - バックエンドとしてTF以外を想定しないため、tf.mathの関数を直接使用
"""


class DICELoss(keras.losses.Loss):
    """
    Tversky損失の`alpha`=0.5であるが継承せずため独立に実装
    """

    def __init__(self, name=None):
        """
        ゼロ除算対策のためのパラメータ設定
        """
        super().__init__(name=name)
        self.smooth = 1e-10

    def call(self, y_true, y_pred):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])
        tp_mul = tf.math.reduce_sum(y_true_pos * y_pred_pos)
        tp_sum = tf.math.reduce_sum(y_true_pos + y_pred_pos)
        dc = 2 * (tp_mul + self.smooth) / (tp_sum + self.smooth)
        return 1.0 - dc


class BCEwithDICELoss(keras.losses.Loss):
    """
    CRESI論文に沿って、
    0.8 * BCE + 0.2 * (1-DICE)
    でやってみる
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.dice = DICELoss(name="DICE")

    def call(self, y_true, y_pred):
        bce_loss = self.bce.call(y_true, y_pred)
        dice_loss = self.dice.call(y_true, y_pred)
        loss_value = tf.add(
            tf.multiply(0.8, bce_loss),
            tf.multiply(0.2, tf.add(1.0, tf.multiply(-1.0, dice_loss))),
        )
        return loss_value


class TverskyLoss(keras.losses.Loss):
    """
    Dice損失の派生
    """

    def __init__(self, name=None, alpha=0.3):
        super().__init__(name=name)
        self.alpha = alpha
        self.smooth = 1.0

    def call(self, y_true, y_pred):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])

        tp = tf.math.reduce_sum(y_true_pos * y_pred_pos)
        fp = tf.math.reduce_sum((1 - y_pred_pos) * y_true_pos)
        fn = tf.math.reduce_sum((y_pred_pos) * (1 - y_true_pos))

        TI = (tp + self.smooth) / (
            tp + self.smooth + self.alpha * fp + (1 - self.alpha) * fn
        )
        Lt = 1.0 - TI

        return Lt


class FocalTverskyLoss(TverskyLoss):
    """
    TverskyLossを継承。新たなハイパーパラメータとして`gamma`=0.75
    """

    def __init__(self, name=None, alpha=0.3, gamma=0.75):
        super().__init__(name=name, alpha=alpha)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        継承元のTverskyLossの結果を利用する
        """
        tversky_loss = super().call(y_true, y_pred)
        ftl = tf.math.pow(tversky_loss, self.gamma)
        return ftl
