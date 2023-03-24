from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    train_image_folder: str = field(
        default="./data/train", metadata={"help": "path to image folder"}
    )
    train_jsonl_path: str = field(
        default="./data/train/metadata.jsonl", metadata={"help": "path to train jsonl file"}
    )
    val_image_folder: str = field(
        default="./data/validation", metadata={"help": "path to image folder"}
    )
    val_jsonl_path: str = field(
        default="./data/validation/metadata.jsonl", metadata={"help": "path to val jsonl file"}
    )
    classification_type: str = field(
        default="graph", metadata={"help": "classification type"}
    )