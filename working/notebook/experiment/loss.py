import segmentation_models_pytorch as smp

def bce_loss(y_pred, y_true):
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    return BCELoss(y_pred, y_true)