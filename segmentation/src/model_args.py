from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """
    resume: Optional[str] = field(
        default=None, metadata={"help": "Path of model checkpoint"}
    )
    encoder_name: str = field(
        default="convnext_xlarge_in22ft1k", metadata={"help": "Pretrained timm model as encoder"}
    )
    drop_path: float = field(
        default=0.2, metadata={"help": "Drop path of encoder"}
    )
    arch: str = field(
        default="Unet", metadata={"help": "Pretrained seg arch"}
    )

