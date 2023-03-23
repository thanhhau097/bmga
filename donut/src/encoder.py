import math
import os
from typing import List, Union

import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate

to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)


def prepare_input(
    img: Union[PIL.Image.Image, str],
    input_size: List[int],
    align_long_axis: bool = False,
    random_padding: bool = False,
) -> torch.Tensor:
    """
    Convert PIL Image to tensor according to specified input_size after following steps below:
        - resize
        - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
        - pad
    """
    if isinstance(img, str):
        img = PIL.Image.open(img)
    if isinstance(img, dict):
        img = PIL.Image.open(img["path"])
    img = img.convert("RGB")
    if align_long_axis and (
        (input_size[0] > input_size[1] and img.width > img.height)
        or (input_size[0] < input_size[1] and img.width < img.height)
    ):
        img = rotate(img, angle=-90, expand=True)
    img = resize(img, min(input_size))
    img.thumbnail((input_size[1], input_size[0]))
    delta_width = input_size[1] - img.width
    delta_height = input_size[0] - img.height
    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return to_tensor(ImageOps.expand(img, padding))


class TimmEncoder(nn.Module):
    def __init__(self, input_size: List[int], align_long_axis: bool, backbone_name: str):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis

        self.model = timm.create_model(backbone_name, pretrained=True)
        self.model.reset_classifier(0, "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.flatten(x, 2).permute(0, 2, 1)
        return x

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False):
        return prepare_input(img, self.input_size, self.align_long_axis, random_padding)


class SwinEncoder(nn.Module):
    r"""
    Donut encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations as a Donut Encoder

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )

        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_swin_state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1).reshape(1, new_len**2, -1).squeeze(0)
                    )
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False):
        return prepare_input(img, self.input_size, self.align_long_axis, random_padding)
