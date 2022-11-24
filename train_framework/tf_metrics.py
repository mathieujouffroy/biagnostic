import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def precision(y_true, y_pred):
    """


    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
    Returns:

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def sensitivity(y_true, y_pred):
    """


    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
    Returns:

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    """


    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
    Returns:

    """

    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), epsilon=0.0001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coeff (float): computed value of dice coefficient.
    """

    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis)
    dice_coeff = K.mean((2. * intersection + epsilon) / (union + epsilon))

    return dice_coeff


def soft_dice_coefficient(y_true, y_pred, axis=(1, 2, 3), epsilon=0.0001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coeff (float): computed value of dice coefficient.
    """

    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis)
    dice_coeff = K.mean((2. * intersection + epsilon) / (union + epsilon))

    return dice_coeff


def iou(y_true, y_pred, axis=(1, 2, 3), epsilon=0.0001):
    """
    Compute mean Intersection Over Union over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of Intersection Over Union.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        iou (float): computed value of mea IOU.
    """

    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis)
    iou = (intersection + epsilon) / (union + epsilon)
    return iou


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.0001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.
    """

    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis = axis) + K.sum(y_pred, axis = axis)
    dice_coeff = K.mean((2. * intersection + epsilon) / (union + epsilon))
    dice_loss = 1 - dice_coeff

    return dice_loss


def compute_class_sens_spec(pred, label, class_num):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """

    class_pred = pred[class_num]
    class_label = label[class_num]

    tp = np.sum((class_pred == 1) * (class_label == 1))
    tn = np.sum((class_pred == 0) * (class_label == 0))

    fp = np.sum((class_pred == 1) * (class_label == 0))
    fn = np.sum((class_pred == 0) * (class_label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


## add metrics for pytorch
