from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """
    train_csv: str = field(
        default='./data/train.csv', metadata={"help": "Path of train data csv"}
    )
    val_csv: str = field(
        default='./data/val.csv', metadata={"help": "Path of val data csv"}
    )
    data_dir: str = field(
        default='./data', metadata={"help": "Path of root data dir"}
    )
    train_dir: str = field(
        default='./data', metadata={"help": "Path of root data dir"}
    )
    size: int = field(
        default=512, metadata={"help": "Input image size"}
    )
