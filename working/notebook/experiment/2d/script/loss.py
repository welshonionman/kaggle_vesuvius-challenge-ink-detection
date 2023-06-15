import segmentation_models_pytorch as smp

def bce_loss(y_pred, y_true):
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    return bce_loss(y_pred, y_true)

def bce_dice_loss(y_pred, y_true):
    dice_loss = smp.losses.DiceLoss("binary")
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    return  dice_loss(y_pred, y_true)+bce_loss(y_pred, y_true)

def dice_loss(y_pred, y_true):
    dice_loss = smp.losses.DiceLoss("binary")
    return  dice_loss(y_pred, y_true)