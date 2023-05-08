import gc
from typing import Dict

import numpy as np
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import Model

from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
import segmentation_models_pytorch as smp

DiceLoss = smp.losses.DiceLoss(mode="multilabel")

from monai.metrics.utils import get_mask_edges, get_surface_distance
def compute_hausdorff_monai(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist

def compute_hausdorff_loss(pred, gt):
    y = gt.float().numpy()
    y_pred = pred.float().numpy()

    # hausdorff distance score
    batch_size, n_class = y_pred.shape[:2]
    spatial_size = y_pred.shape[2:]
    max_dist = np.sqrt(np.sum([ss**2 for ss in spatial_size]))
    hd_loss = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        hd_loss[b, c] = compute_hausdorff_monai(y_pred[b, c], y[b, c], max_dist)

    return torch.from_numpy(hd_loss)

@torch.no_grad()
def hausdorff_loss(predictions, gts):
    # non_empty_idxs = torch.stack([label.sum() > 0 for label in gts])
    hausdorff_loss = compute_hausdorff_loss(
        torch.where(torch.sigmoid(predictions).detach().cpu() >= 0.5, 1, 0),
        gts.detach().cpu(),
    )
    # hausdorff_loss[:, 2] *= 4
    return hausdorff_loss.mean().float().to(predictions.device)

def seg_criterion(y_pred, y_true, add_hausdorff=False):
    losses = 0
    losses += DiceLoss(y_pred, y_true)
    if add_hausdorff:
        losses += hausdorff_loss(y_pred, y_true)
    return losses

class CustomTrainer(Trainer):
    def __init__(self, pos_neg_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.pos_neg_ratio = pos_neg_ratio

    def compute_loss(self, model: Model, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = model(inputs["images"].to(device))
        labels = inputs.get("labels")
        loss = seg_criterion(outputs, labels)
        if return_outputs:
            return (loss, outputs)
        return loss

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
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if type(outputs) == tuple:
            outputs = outputs[0]  # return only classification outputs
        outputs = outputs.float()
        outputs = nested_detach(outputs)
        gc.collect()
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    tps, fps, fns, tns = smp.metrics.get_stats(
        torch.from_numpy(predictions), torch.from_numpy(labels).long().unsqueeze(1), mode="multilabel", threshold=0.5
    )
    f1_score = smp.metrics.f1_score(tps, fps, fns, tns).mean(dim=0).cpu()
    mean_f1_score = f1_score.mean().item()
    iou_score = smp.metrics.iou_score(tps, fps, fns, tns).mean(dim=0).cpu()
    mean_iou_score = iou_score.mean().item()
    recall = smp.metrics.recall(tps, fps, fns, tns).mean(dim=0).cpu()
    mean_recall = recall.mean().item()
    prec = smp.metrics.precision(tps, fps, fns, tns).mean(dim=0).cpu()
    mean_prec = prec.mean().item()
    # mean_h_dists = h_dists.mean().
    return {"iou_score": mean_iou_score, "f1_score": mean_f1_score, "recall": mean_recall, "prec": mean_prec}
