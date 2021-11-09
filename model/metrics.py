from tensorflow import math as tfm


def iou_coef(y_true, y_pred):
    smooth = 1
    intersection = tfm.reduce_sum(tfm.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (
        tfm.reduce_sum(y_true, [1, 2, 3])
        + tfm.reduce_sum(y_pred, [1, 2, 3])
        - intersection
    )
    iou = tfm.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
