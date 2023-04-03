import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .src.strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import PARSeq

class TextRecognitionModel:
    def __init__(self, weights_path='baudm/parseq', model_name="parseq", config_path=None) -> None:
        # Load model and image transforms
        # self.parseq = torch.hub.load(weights_path, model_name, pretrained=True).eval()
        with open(config_path) as f:
            config = json.load(f)
        self.parseq = PARSeq(**config)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.parseq.load_state_dict(state_dict, strict=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parseq.to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def predict(self, image_paths):
        labels, confidences = [], []
        for path in image_paths:
            if isinstance(path, str):
                img = Image.open(path).convert('RGB')
            elif isinstance(path, Image.Image):
                img = path
            elif isinstance(path, np.ndarray):
                img = Image.fromarray(path).convert('RGB')
            # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
            img = self.img_transform(img).unsqueeze(0)
            img = img.to(self.device)
            logits = self.parseq(img).detach().cpu()
            # logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = self.parseq.tokenizer.decode(pred)
            labels.append(label)
            confidences.append(confidence)

        return labels, confidences

    def postprocess(self, predictions, image_paths):
        return predictions

    def __call__(self, image_paths):
        predictions = self.predict(image_paths)
        return self.postprocess(predictions, image_paths)
