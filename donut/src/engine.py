import math
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from Levenshtein import distance as edit_distance
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import speed_metrics

from .model import DonutModel
from .util import JSONParseEvaluator


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model: DonutModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs=False,
    ):
        outputs = model(**inputs)
        if return_outputs:
            return outputs.loss, outputs.logits
        return outputs.loss

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
            decoder_prompts = pad_sequence(
                [
                    input_id[: end_idx + 1]
                    for input_id, end_idx in zip(
                        inputs["decoder_input_ids"], inputs["prompt_end_idxs"]
                    )
                ],
                batch_first=True,
            )
            if "encoder_input_ids" not in inputs:
                sequences = model.inference(
                    image_tensors=inputs["image_tensors"],
                    prompt_tensors=decoder_prompts,
                    return_json=False,
                    return_attentions=False,
                    return_decoded_sequences=False,
                    # device and type is controlled by Trainer
                    fp16=False,
                )["sequences"]
            else:
                sequences = model.inference(
                    image_tensors=inputs["image_tensors"],
                    prompt_tensors=decoder_prompts,
                    encoder_input_ids=inputs["encoder_input_ids"],
                    return_json=False,
                    return_attentions=False,
                    return_decoded_sequences=False,
                    # device and type is controlled by Trainer
                    # fp16=False
                )["sequences"]
        loss = torch.zeros((1,))

        if prediction_loss_only:
            return (loss, None, None)

        return loss, nested_detach(sequences), nested_detach(inputs["decoder_labels"])

    def _score_prediction(self, pred, answer, replace_tokens=[]):
        for token in replace_tokens:
            pred = pred.replace(token, "")
            answer = answer.replace(token, "")

        pred = re.sub(r"<.*?>", "", pred, count=1).strip()
        pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)

        answer = re.sub(r"(?:(?<=>) | (?=</s_))", "", answer)
        score = 0 # edit_distance(pred, answer) / max(len(pred), len(answer))

        return score, pred, answer

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Predict",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        num_samples = len(eval_dataset)

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        # calculate metric
        scores = []
        tokenizer = self.model.decoder.tokenizer

        # replace `ignore_token` (`-100`) before decode
        output.predictions[output.predictions == -100] = tokenizer.pad_token_id
        output.label_ids[output.label_ids == -100] = tokenizer.pad_token_id

        predictions = tokenizer.batch_decode(output.predictions)
        answers = tokenizer.batch_decode(output.label_ids)

        json_preds = []
        json_answers = []
        accs = []

        evaluator = JSONParseEvaluator()
        for pred, answer in zip(predictions, answers):
            score, pred, answer = self._score_prediction(
                pred, answer, [tokenizer.eos_token, tokenizer.pad_token]
            )
            scores.append(score)

            json_pred = self.model.token2json(pred)
            json_answer = self.model.token2json(answer)
            json_preds.append(json_pred)
            json_answers.append(json_answer)
            acc = 0 # evaluator.cal_acc(json_pred, json_answer)
            accs.append(acc)

            # TODO: add option for verbosity here
            print(f"Prediction: {pred}")
            print(f"    Answer: {answer}")
            print(f" Normed ED: {score}\n")
            print(f" Accuracy : {acc}\n")

        print("\n\n")

        class_acc = calculate_acc("class", json_preds, json_answers)
        x_type_acc = calculate_acc("x_type", json_preds, json_answers)
        y_type_acc = calculate_acc("y_type", json_preds, json_answers)

        x_labels_score, x_labels_similarity_score = calculate_labels_acc("x_labels", json_preds, json_answers)
        y_labels_score, y_labels_similarity_score = calculate_labels_acc("y_labels", json_preds, json_answers)

        output.metrics.update({f"{metric_key_prefix}_class_acc": class_acc})
        output.metrics.update({f"{metric_key_prefix}_x_type_acc": x_type_acc})
        output.metrics.update({f"{metric_key_prefix}_y_type_acc": y_type_acc})

        output.metrics.update({f"{metric_key_prefix}_x_labels_acc": x_labels_score})
        output.metrics.update({f"{metric_key_prefix}_x_labels_similarity_acc": x_labels_similarity_score})
        output.metrics.update({f"{metric_key_prefix}_y_labels_acc": y_labels_score})
        output.metrics.update({f"{metric_key_prefix}_y_labels_similarity_acc": y_labels_similarity_score})

        f1_score = 0 # evaluator.cal_f1(json_preds, json_answers)
        output.metrics.update({f"{metric_key_prefix}_f1": f1_score})
        output.metrics.update({f"{metric_key_prefix}_acc": np.mean(accs)})

        output.metrics.update({f"{metric_key_prefix}_ED": np.mean(scores)})

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


def calculate_acc(key, preds, answers):
    pred_classes = []
    for pred in preds:
        if isinstance(pred, list):
            pred = pred[0]
        pred_classes.append(pred.get(key, "1"))


    answer_classes = []
    for answer in answers:
        if isinstance(answer, list):
            answer = answer[0]
        answer_classes.append(answer.get(key, "2"))

    from sklearn.metrics import accuracy_score

    return accuracy_score(answer_classes, pred_classes)


def calculate_labels_acc(key, preds, answers):
    pred_labels = []
    for pred in preds:
        if isinstance(pred, list):
            pred = pred[0]
        pred_labels.append(pred.get(key, ["1"]))

    answer_labels = []
    for answer in answers:
        if isinstance(answer, list):
            answer = answer[0]
        answer_labels.append(answer.get(key, ["2"]))

    score = 0
    similarity_score = 0

    for i in range(len(pred_labels)):
        if len(pred_labels[i]) == len(answer_labels[i]):
            match = 0
            similarity = 0
            for j in range(len(pred_labels[i])):
                if pred_labels[i][j] == answer_labels[i][j]:
                    match += 1
                # 1 - edit distance
                similarity += 1 - edit_distance(pred_labels[i][j], answer_labels[i][j]) / max(len(pred_labels[i][j]), len(answer_labels[i][j]))

            similarity /= len(pred_labels[i])
            similarity_score += similarity
            
            if match == len(pred_labels[i]):
                score += 1

    return score / len(pred_labels), similarity_score / len(pred_labels)
