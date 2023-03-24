import timm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name: str = "resnet50", n_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=3, drop_path_rate=0.2
        )
        self.backbone.reset_classifier(0, "avg")

        self.linear = nn.Linear(self.backbone.num_features, n_classes)

    def feature(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.linear(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature(images)
        return features
