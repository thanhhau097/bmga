from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    max_patches: int = field(
        default=2048,
        metadata={"help": "The maximum number of patches to extract from the image."},
    )
