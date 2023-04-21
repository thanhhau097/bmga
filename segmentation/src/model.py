import torch
import torch.nn as nn

from modules import (
    TimmUniversalEncoder, 
    DeepLabV3Decoder, 
    DeepLabV3PlusDecoderFix, 
    DeepLabV3PlusDecoder,
    UnetPlusPlusDecoder,
    UnetPlusPlusDecoderFix,
    UnetDecoder,
    SegmentationHead,
    SegmentationHeadDouble
)


class Model(nn.Module):
    def __init__(self, arch, encoder_name, drop_path, size, pretrained=True):
        super(Model, self).__init__()

        self.encoder = TimmUniversalEncoder(
            encoder_name,
            in_channels=3,
            drop_path_rate=drop_path,
            img_size=size,
            pretrained=pretrained,
        )
        with torch.no_grad():
            dummy_inputs = torch.randn(2, 3, size, size)
            out = self.encoder(dummy_inputs)
            common_stride = size // out[1].shape[2]
            # if cfg.MODEL.BACKBONE.ENCODER.startswith('tf_efficientnet') or cfg.MODEL.BACKBONE.ENCODER.startswith('ecaresnet'):
            #     common_stride *= 2
        encoder_channels = self.encoder.out_channels
        # conv_dims = 64
        # decoder_channels = [conv_dims * (2 ** i) for i in range(len(encoder_channels))][::-1]
        # attention_type = "scse"
        num_classes = 1

        if arch == "DeepLabV3Plus":
            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=encoder_channels,
            )
        elif arch == "DeepLabV3PlusFix":
            self.decoder = DeepLabV3PlusDecoderFix(
                encoder_channels=encoder_channels,
            )
        elif arch == "UnetPlusPlus":
            self.decoder = UnetPlusPlusDecoder(
                decoder_channels=(256//8, 128//8, 64//8, 32//8, 16//8),
                encoder_channels=encoder_channels,
            )
        elif arch == "UnetPlusPlusFix":
            self.decoder = UnetPlusPlusDecoderFix(
                decoder_channels=(256//8, 128//8, 64//8, 32//8),
                encoder_channels=encoder_channels,
            )
        elif arch == "Unet":
            self.decoder = UnetDecoder(
                n_blocks=5,
                decoder_channels=(256, 128, 64, 32, 16),
                encoder_channels=encoder_channels,
            )

        with torch.no_grad():
            out = self.decoder(*out)
        if arch == "UnetPlusPlus" or arch == "Unet":
            self.segmentation_head = SegmentationHead(
                in_channels=out.shape[1],
                out_channels=num_classes,
                activation=None,
                kernel_size=3,
                upsampling=1,
            )
        elif arch == "UnetPlusPlusFix":
            self.segmentation_head = SegmentationHead(
                in_channels=out.shape[1],
                out_channels=num_classes,
                activation=None,
                kernel_size=3,
                upsampling=common_stride // 2,
            )
        else:
            if encoder_name.startswith(
                "tf_efficientnet"
            ) or encoder_name.startswith("ecaresnet"):
                self.segmentation_head = SegmentationHeadDouble(
                    in_channels=out.shape[1],
                    out_channels=num_classes,
                    activation_func=None,
                    kernel_size=3,
                    upsampling_scale=common_stride,
                )
            else:
                self.segmentation_head = SegmentationHead(
                    in_channels=out.shape[1],
                    out_channels=num_classes,
                    activation=None,
                    kernel_size=3,
                    upsampling=common_stride,
                )
        # self.classification_head = ClassificationHead(
        #     in_channels=self.encoder.out_channels[-1], classes=num_classes
        # )
        self._add_hausdorff = False

    def forward(self, images):
        features = self.encoder(images)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return torch.sigmoid(masks)