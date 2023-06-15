import torch
import torch.nn as nn


class BCEWithDiceLoss(nn.Module):
    def __init__(self, weight_dice=0.5, smooth=1e-7):
        super(BCEWithDiceLoss, self).__init__()
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, outputs, targets):
        bce_loss = nn.BCEWithLogitsLoss()(outputs, targets)

        intersection = torch.sum(torch.sigmoid(outputs) * targets)
        union = torch.sum(torch.sigmoid(outputs)) + torch.sum(targets)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        loss = self.weight_dice * dice_loss + (1 - self.weight_dice) * bce_loss
        return loss
