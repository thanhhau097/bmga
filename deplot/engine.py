import gc
from enum import Enum
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataset import display_deplot_output
from metrics import benetech_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss
from transformers import Pix2StructProcessor, Trainer
from transformers.trainer_pt_utils import nested_detach


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs: Dict, return_outputs=False):
        outputs = model(**inputs)
        if return_outputs:
            return (outputs["loss"], outputs)
        return outputs["loss"]

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # with self.compute_loss_context_manager():
            #     loss, _ = self.compute_loss(model, inputs, return_outputs=True)
            #     loss = loss.mean().detach()

            outputs = model.generate(**inputs, max_new_tokens=256)

        if prediction_loss_only:
            return torch.FloatTensor([0.0]), None, None

        outputs = nested_detach(outputs)
        return torch.FloatTensor([0.0]), outputs, inputs["labels"]


def compute_metrics(eval_preds, val_df: pd.DataFrame, processor: Pix2StructProcessor):
    # assume chart_type is all correct
    chart_types = val_df["chart_type"].values
    ground_truth = []
    predictions = []

    for i, (preds, labels, chart_type) in enumerate(
        zip(eval_preds.predictions, eval_preds.label_ids, chart_types)
    ):
        preds = display_deplot_output(
            processor.decode(preds[preds != -100], skip_special_tokens=True)
        )
        labels = display_deplot_output(processor.decode(labels, skip_special_tokens=True))
        predictions.append(
            pd.DataFrame.from_dict(
                {
                    f"{i}_x": (preds.values[:, 0], chart_type),
                    f"{i}_y": (preds.values[:, 1], chart_type),
                },
                orient="index",
                columns=["data_series", "chart_type"],
            ).rename_axis("id")
        )
        ground_truth.append(
            pd.DataFrame.from_dict(
                {
                    f"{i}_x": (labels.values[:, 0], chart_type),
                    f"{i}_y": (labels.values[:, 1], chart_type),
                },
                orient="index",
                columns=["data_series", "chart_type"],
            ).rename_axis("id")
        )

    predictions = pd.concat(predictions)
    ground_truth = pd.concat(ground_truth)
    score_by_chart = {}
    for chart_type in val_df["chart_type"].unique():
        score_by_chart[chart_type] = benetech_score(
            ground_truth[ground_truth["chart_type"] == chart_type],
            predictions[predictions["chart_type"] == chart_type],
        )
    score_by_chart["overall"] = benetech_score(ground_truth, predictions)
    return score_by_chart
