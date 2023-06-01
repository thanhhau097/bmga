import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    TimmUniversalEncoder, 
    DeepLabV3PlusDecoder,
    UnetPlusPlusDecoder,
    UnetDecoder,
    SegmentationHead,
)



class Model(nn.Module):
    def __init__(self, arch, encoder_name, drop_path, size, pretrained=True):
        super(Model, self).__init__()

        self.encoder = TimmUniversalEncoder(
            encoder_name, in_channels=3, drop_path_rate=drop_path, img_size=size, pretrained=pretrained
        )
        with torch.no_grad():
            dummy_inputs = torch.randn(2, 3, size, size)
            out = self.encoder(dummy_inputs)
            common_stride = size // out[1].shape[2]
        encoder_channels = self.encoder.out_channels
        num_classes = 1

        self.decoder = UnetDecoder(
            n_blocks=5,
            decoder_channels=(256, 128, 64, 32, 16),
            encoder_channels=encoder_channels,
        )

        with torch.no_grad():
            out = self.decoder(*out)

        self.segmentation_head = SegmentationHead(
            in_channels=out.shape[1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=1,
        )

        self._add_hausdorff = False
        # Attention layer
        self.attn_weight = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for dim in encoder_channels
            ]
        )
        self.encoder_2 = TimmUniversalEncoder(
            "tf_efficientnetv2_s_in21ft1k",
            in_channels=1,
            drop_path_rate=drop_path,
            img_size=size,
            pretrained=pretrained,
        )
        with torch.no_grad():
            dummy_inputs = torch.randn(2, 1, size, size)
            out = self.encoder_2(dummy_inputs)
            common_stride = size // out[1].shape[2]
        encoder_channels_2 = self.encoder_2.out_channels

        self.decoder_2 = UnetDecoder(
            n_blocks=5,
            decoder_channels=(256, 128, 64, 32, 16),
            encoder_channels=encoder_channels_2,
        )

        with torch.no_grad():
            out = self.decoder_2(*out)

        self.segmentation_head_2 = SegmentationHead(
            in_channels=out.shape[1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=1,
        )

    def forward(self, images):
        features = self.encoder(images)

        # Apply attention weight for encoder output feature
        for i in range(len(features)):
            e = features[i]
            f = self.attn_weight[i](e)
            w = F.softmax(f, 1)
            e = w * e
            features[i] = features[i].clone() + e

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        features = self.encoder_2(masks)
        decoder_output = self.decoder_2(*features)
        masks_2 = self.segmentation_head_2(decoder_output)
        return (torch.sigmoid(masks), torch.sigmoid(masks_2))

