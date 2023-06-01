import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from .src.model import Model
from .src.old_model import OldModel


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, size):
        self.image_paths = image_paths
        self.transform = A.Compose(
            [
                A.Resize(width=size, height=size, interpolation=1), 
                ToTensorV2()
            ]
        )
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        transformed = self.transform(image=img)
        return transformed['image'] / 255.0


def collate_fn(batch):
    images = torch.stack(batch)
    return {"images": images}


class SegmentationModel:
    def __init__(self, arch, encoder_name, drop_path, size, weights_path, version="new"):
        self.version = version
        if version == "new":
            self.model = Model(arch, encoder_name, drop_path, size, pretrained=False)
        elif version == "old":
            self.model = OldModel(arch, encoder_name, drop_path, size, pretrained=False)
        else:
            raise ValueError(f"Unknown version: {version}")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        self.size = size

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict(self, image_paths, batch_size=16, num_workers=0):
        dataset = InferenceDataset(image_paths, self.size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        masks = []
        for batch in tqdm(dataloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                mask = self.model(batch["images"])
            
            if self.version == "old":
                masks.append(mask.cpu().numpy())
            else:
                masks.append(mask[1].cpu().numpy())

        return np.concatenate(masks)

    def postprocess(self, predictions, image_paths):
        return predictions

    def __call__(self, image_paths, batch_size=16, num_workers=16):
        predictions = self.predict(image_paths, self.size, batch_size, num_workers)
        return self.postprocess(predictions, image_paths)
