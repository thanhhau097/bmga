import torch
from PIL import Image
from src.strhub.data.module import SceneTextDataModule


class TextRecognitionModel:
    def __init__(self, weights_path='baudm/parseq', model_name="parseq") -> None:
        # Load model and image transforms
        self.parseq = torch.hub.load(weights_path, model_name, pretrained=True).eval()
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def predict(self, image_paths):
        labels, confidences = [], []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
            img = self.img_transform(img).unsqueeze(0)

            logits = self.parseq(img)
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
