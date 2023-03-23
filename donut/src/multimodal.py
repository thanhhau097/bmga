import os
import re
from typing import List, Optional, Union

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from .attention import AttentionFusionModule, ResidualAttentionBlock
from .model import DonutModel


class MultimodalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "multimodal"

    def __init__(
        self,
        input_size: List[int] = [2560, 1920],
        align_long_axis: bool = False,
        window_size: int = 10,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 1536,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        backbone_name: str = "",
        text_encoder_name: str = "xlm-roberta-base",
        fusion: str = "attention_cat",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.backbone_name = backbone_name
        self.text_encoder_name = text_encoder_name
        self.fusion = fusion


class MultimodalModel(DonutModel, PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = MultimodalConfig
    base_model_prefix = "multimodal"

    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        self.text_encoder_tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        self.n_heads = 8
        if config.fusion == "attention_cat":
            self.fusion_module = AttentionFusionModule(
                {
                    "vision": self.encoder.model.num_features,
                    "text": self.text_encoder.config.hidden_size,
                }
            )
        elif config.fusion == "attention_clip":
            self.text_proj = nn.Linear(
                self.text_encoder.config.hidden_size, self.encoder.model.num_features
            )
            d_model = self.encoder.model.num_features
            self.fusion_module = ResidualAttentionBlock(
                d_model, n_head=self.n_heads, is_cross_attention=True
            )

    def encode(self, image_tensors: torch.Tensor, encoder_input_ids: torch.Tensor):
        encoder_outputs = self.encoder(image_tensors)
        text_encoder_outputs = self.text_encoder(encoder_input_ids).last_hidden_state
        if self.config.fusion == "attention_cat":
            fused = self.fusion_module({"vision": encoder_outputs, "text": text_encoder_outputs})
        elif self.config.fusion == "attention_clip":
            text_encoder_outputs = self.text_proj(text_encoder_outputs)
            mask = encoder_input_ids == self.text_encoder_tokenizer.pad_token_id
            mask = torch.stack([mask] * encoder_outputs.shape[1], 1)
            mask = torch.stack([mask] * self.n_heads, 1).flatten(0, 1)
            # fused = self.fusion_module(text_encoder_outputs, encoder_outputs, encoder_outputs) # text as query
            fused = self.fusion_module(
                encoder_outputs, text_encoder_outputs, text_encoder_outputs, mask
            )  # vision as query
        return fused

    def forward(
        self,
        image_tensors: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encode(image_tensors, encoder_input_ids)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
        )
        return decoder_outputs

    @torch.no_grad()
    def inference(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        last_hidden_state: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
        return_decoded_sequences: bool = True,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            # image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        if last_hidden_state is None:
            last_hidden_state = self.encode(image_tensors, encoder_input_ids)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )
        if not return_decoded_sequences:
            return decoder_output

        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(
                self.decoder.tokenizer.pad_token, ""
            )
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output
