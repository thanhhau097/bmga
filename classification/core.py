import cv2
import numpy as np
import torch
from tqdm import tqdm

from .src.model import Model


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, size):
        self.image_paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = image.transpose(2, 0, 1)
        image = image.astype("float32")
        image /= 255.0
        return torch.tensor(image).float()


def collate_fn(batch):
    images = torch.stack(batch)
    return {"images": images}


class ClassificationModel:
    def __init__(self, model_name, n_classes, weights_path):
        self.model = Model(model_name, n_classes, pretrained=False)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict(self, image_paths, size=(640, 320), batch_size=16, num_workers=0):
        dataset = InferenceDataset(image_paths, size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        predictions = []
        for batch in tqdm(dataloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(batch["images"])

            # softmax
            output = torch.exp(output) / torch.exp(output).sum(dim=1, keepdim=True)
            predictions.append(output.cpu().numpy())

        return np.concatenate(predictions)

    def postprocess(self, predictions, image_paths):
        return predictions

    def __call__(self, image_paths, size=(640, 320), batch_size=16, num_workers=16):
        predictions = self.predict(image_paths, size, batch_size, num_workers)
        return self.postprocess(predictions, image_paths)
