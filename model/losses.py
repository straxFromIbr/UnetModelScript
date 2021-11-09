import tensorflow as tf
import tensorflow.keras as keras

"""
参考：https://qiita.com/ppza53893/items/8090322792e1c7f81e57
# 変更点：
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
        self.smooth = 1.0

    def call(self, y_true, y_pred):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])
        tp = tf.math.reduce_sum(y_true_pos * y_pred_pos)
        ptsum = tf.math.reduce_sum(y_true_pos + y_pred_pos)
        dc = (2.0 * tp + self.smooth) / (ptsum + self.smooth)
        return 1.0 - dc


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
            tp + self.smooth + self.alpha * fp + (1 - self.alpha) * fn + self.smooth
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
