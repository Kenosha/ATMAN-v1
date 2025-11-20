import torch
import torch.nn.functional as F
import torchreid
import numpy as np
from PIL import Image

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class TorchreidEncoder:
    """
    Wrapper around torchreid models to provide embeddings
    compatible with Ultralytics BoT-SORT.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        height: int = 256,
        width: int = 128,
        device: str | torch.device | None = None,
    ):
        if device is None or str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True,
        ).to(self.device).eval()

        train_tfms, test_tfms = torchreid.data.transforms.build_transforms(height=height, width=width)
        self.transform = test_tfms

    def __call__(self, img: np.ndarray, dets: np.ndarray):
        """
        Extract normalized embeddings for every detection.
        The detections array is expected to be [x_center, y_center, width, height, ...]
        as provided by Ultralytics.
        """
        if len(dets) == 0:
            return []

        boxes = xywh2xyxy(torch.from_numpy(dets[:, :4]))
        crops = []
        for box in boxes:
            crop = save_one_box(box, img, BGR=True, save=False)  # Input from pipeline is BGR
            crop = Image.fromarray(crop)
            tensor = self.transform(crop).unsqueeze(0)
            crops.append(tensor)

        batch = torch.cat(crops).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        feats = F.normalize(feats, dim=1)
        return [feat.cpu().numpy() for feat in feats]
