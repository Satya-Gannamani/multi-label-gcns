import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 50,
        weights: str = "None",  # can also be None
        progress: bool = True
    ):
        super(ResNet, self).__init__()
        assert num_layers in [34, 50, 101]
        assert weights in ["IMAGENET1K_V1", "IMAGENET1K_V2", None]

        # Convert weights string to the appropriate Enum or None
        if weights is None:
            weight_enum = None
        else:
            if num_layers == 34:
                weight_enum = getattr(ResNet34_Weights, weights)
            elif num_layers == 50:
                weight_enum = getattr(ResNet50_Weights, weights)
            else:
                weight_enum = getattr(ResNet101_Weights, weights)

        params = {"weights": weight_enum, "progress": progress}

        if num_layers == 34:
            self.resnet = models.resnet34(**params)
        elif num_layers == 50:
            self.resnet = models.resnet50(**params)
        else:
            self.resnet = models.resnet101(**params)

    def get_backbone(self):
        return nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)
