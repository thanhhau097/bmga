import torch

from .src.tools.demo import Predictor, get_image_list, image_demo, get_exp


class ObjectDetectionModel:
    def __init__(
        self, name, experiment_path, weights_path, classes, conf_thre=0.15, nms_thre=0.25, test_size=(640, 640)
    ):
        self.exp = get_exp(experiment_path, name)
        self.exp.test_conf = conf_thre
        self.exp.nmsthre = nms_thre
        self.exp.test_size = test_size
        self.model = self.exp.get_model()

        ckpt = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = Predictor(
            self.model, self.exp, classes, None, None, device, False, False
        )

    def predict(self, image_paths, size=(640, 640), batch_size=16, num_workers=16):
        all_outputs, all_img_info = [], []
        for path in image_paths:
            outputs, img_info = self.predictor.inference(path)
            all_outputs.append(outputs)
            all_img_info.append(img_info)

        return all_outputs, all_img_info
        