from dataclasses import dataclass, field


@dataclass
class Arguments:
    """
    Extra arguments besides HF's TrainingArguments.
    """

    config: str = field(metadata={"help": "Yaml config path."})
