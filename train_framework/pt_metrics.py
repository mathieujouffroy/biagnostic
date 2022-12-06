import numpy as np
import torch

## add focal loss
# compare specificity, dice_loss, dice_coeff, IOU with torch internal metrics
# check https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d for more metrics & loss implementations

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

    intersection = torch.sum(y_pred*y_true, dim=axis)
    union = torch.sum(y_pred, dim=axis) + torch.sum(y_true, dim=axis)

    dice_coeff = torch.mean((2. * intersection + epsilon) / (union + epsilon))

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

    y_pred = torch.round(y_pred)
    intersection = torch.sum(y_pred*y_true, dim=axis)
    union = torch.sum(y_pred, dim=axis) + torch.sum(y_true, dim=axis)

    dice_coeff = torch.mean((2. * intersection + epsilon) / (union + epsilon))

    return dice_coeff


def iou(y_true, y_pred, axis=(1, 2, 3), epsilon=0.0001):
    """
    Compute Intersection Over Union over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of Intersection Over Union.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        iou (float): computed value of mea IOU.
    """

    y_pred = torch.round(y_pred)
    intersection = torch.sum(y_true * y_pred, axis=axis)
    union = torch.sum(y_true, axis = axis) + torch.sum(y_pred, axis = axis)
    iou =  (intersection + epsilon) / (union + epsilon)
    return iou


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=1.):
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

    intersection = torch.sum(y_pred*y_true, dim=axis)
    union = torch.sum(y_pred, dim=axis) + torch.sum(y_true, dim=axis)

    dice_coeff = torch.mean((2. * intersection + epsilon) / (union + epsilon))
    dice_loss = 1 - dice_coeff

    return dice_loss
