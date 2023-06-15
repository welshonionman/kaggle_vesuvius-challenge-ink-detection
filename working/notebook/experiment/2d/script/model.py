import torch.nn as nn
import segmentation_models_pytorch as smp


class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        if cfg.model_name == "PSPNet":
            print("PSPNet")
            model = smp.PSPNet
        elif cfg.model_name == "UnetPlusPlus":
            print("UnetPlusPlus")
            model = smp.UnetPlusPlus
        elif cfg.model_name == "Linknet":
            print("Linknet")
            model = smp.Linknet
        elif cfg.model_name == "MAnet":
            print("MAnet")
            model = smp.MAnet
        else:
            model = smp.Unet

        self.encoder = model(
            encoder_name=cfg.backbone,
            encoder_weights=weight,
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        return output


def build_model(cfg, weight="imagenet"):
    model = CustomModel(cfg, weight)

    return model
