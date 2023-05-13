import io
import json
import re

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from data.augment import rand_augment_transform

NEW_LINE = "<0x0A>"
SEP_CELL = " | "
BOS_TOKEN = "<|BOS|>"
CHART_TYPES = ["<line>", "<vertical_bar>", "<scatter>", "<dot>", "<horizontal_bar>"]


def create_processor(processor_name):
    processor = Pix2StructProcessor.from_pretrained(processor_name, is_vqa=False)
    # extra_tokens = [BOS_TOKEN] + CHART_TYPES
    # processor.tokenizer.add_tokens(extra_tokens)
    return processor


def display_deplot_output(deplot_output):
    """
    Display the output of DePlot in a nice table format.
    """
    deplot_output = deplot_output.replace("<0x0A>", "\n").replace(" | ", "\t")

    second_a_index = [m.start() for m in re.finditer("\t", deplot_output)][1]
    last_newline_index = deplot_output.rfind("\n", 0, second_a_index)

    title = deplot_output[:last_newline_index]
    table = deplot_output[last_newline_index + 1 :]

    # print(title)

    data = io.StringIO(table)
    try:
        df = pd.read_csv(data, sep="\t")
    except:
        df = pd.DataFrame({"x": [0], "y": [0]})
    return df


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: Pix2StructProcessor,
        max_patches: int = 2048,
        augment: bool = False,
    ):
        super().__init__()
        self.processor = processor
        self.df = df
        self.max_patches = max_patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        image = Image.open(info["image"])

        encoding = self.processor(
            images=image,
            return_tensors="pt",
            add_special_tokens=True,
            max_patches=self.max_patches,
        )
        text = json.load(open(info["text"]))

        if "general_figure_info" in text:
            title = text["general_figure_info"]["title"]["text"]
            y_label = text["general_figure_info"]["y_axis"]["label"]["text"]
            x_label = text["general_figure_info"]["x_axis"]["label"]["text"]
            text["data-series"] = [
                {"x": x, "y": y}
                for x, y in zip(text["models"][0]["x"], text["models"][0]["y"])
            ]
        else:
            title = text["text"][0]["text"]
            y_label = text["text"][1]["text"]
            x_label = text["text"][2]["text"]

        # processed text
        # chart_type = "<" + text["chart-type"] + ">"
        # processed = BOS_TOKEN + chart_type + " "
        processed = "TITLE" + SEP_CELL + title + NEW_LINE
        processed += x_label + SEP_CELL + y_label + NEW_LINE
        for series in text["data-series"]:
            x, y = series["x"], series["y"]
            # round float to 2 decimal places
            if isinstance(x, float):
                x = round(x, 2)
            if isinstance(y, float):
                y = round(y, 2)
            processed += str(x) + SEP_CELL + str(y) + NEW_LINE

        return encoding, processed


def collate_fn(batch, processor: Pix2StructProcessor):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item[1] for item in batch]
    text_inputs = processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
    )
    new_batch["labels"] = text_inputs.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item[0]["flattened_patches"])
        new_batch["attention_mask"].append(item[0]["attention_mask"])
    new_batch["attention_mask"] = torch.cat(new_batch["attention_mask"])
    new_batch["flattened_patches"] = torch.cat(new_batch["flattened_patches"])
    return new_batch


def read_annotation(path: str):
    df = pd.read_csv(path, header=None)
    df["image"] = df[0].apply(lambda x: "../data/benetech/train/images/" + x)
    df["text"] = df[0].apply(
        lambda x: "../data/benetech/train/annotations/" + x.split(".")[0] + ".json"
    )
    df["chart_type"] = df["text"].apply(lambda x: json.load(open(x))["chart-type"])
    return df


if __name__ == "__main__":
    from functools import partial

    processor = create_processor("google/deplot")
    train_df = pd.read_csv("../data/train_list.txt", header=None)
    train_df["image"] = train_df[0].apply(
        lambda x: "../data/benetech/train/images/" + x
    )
    train_df["text"] = train_df[0].apply(
        lambda x: "../data/benetech/train/annotations/" + x.split(".")[0] + ".json"
    )
    dataset = ImageCaptioningDataset(train_df, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=4,
    )
    batch = next(iter(dataloader))

    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    model.to("cuda")

    encoding, text = dataset[0]

    encoding = {k: v.to("cuda") for k, v in encoding.items()}
    output = model.generate(**encoding, max_new_tokens=512)
    decoded = processor.batch_decode(output, skip_special_tokens=True)

    output_table = display_deplot_output(decoded[0])
    gt_table = display_deplot_output(text)
