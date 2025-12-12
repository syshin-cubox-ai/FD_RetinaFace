from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import onnxslim
import torch
import torch.nn as nn
from torch import Tensor

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import priorbox
from models.retinaface import RetinaFace
from utils.box_utils import batched_decode_landm, batched_decode_xywh


class RetinaFaceWrapper(nn.Module):
    def __init__(self, cfg: dict, img_size: int):
        super().__init__()
        self.cfg = cfg
        match self.cfg["name"]:
            case "Resnet50":
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://github.com/syshin-cubox-ai/FD_RetinaFace/releases/"
                    "download/v0.0.1-weights/Resnet50_Final.pth"
                )
            case "mobilenet0.25":
                state_dict = torch.load("weights/mobilenet0.25_Final.pth")
            case _:
                raise ValueError("Wrong cfg")

        self.model = RetinaFace(self.cfg, phase="test")
        self.model.load_state_dict(state_dict)
        self.priors = priorbox(
            min_sizes=self.cfg["min_sizes"],
            steps=self.cfg["steps"],
            clip=False,
            image_size=(img_size, img_size),
        )
        self.scale = img_size
        self.conf_thres = 0.9
        self.iou_thres = 0.5

    def forward(self, x: Tensor) -> Tensor:
        loc, conf, landm = self.model(x)

        # Decode
        boxes = batched_decode_xywh(loc, self.priors, self.cfg["variance"])
        boxes = boxes * self.scale
        scores = conf[..., 1:2]
        landmarks = batched_decode_landm(landm, self.priors, self.cfg["variance"])
        landmarks = landmarks * self.scale

        out = torch.cat((boxes, scores, landmarks), 2)
        return out


def convert_onnx(
    model: nn.Module,
    img: Tensor,
    output_path: str | Path,
    dynamic=False,
):
    model.eval()
    output_path = Path(output_path)

    # Define input and output names
    input_names = ["image"]
    output_names = ["pred"]

    # Define dynamic_axes
    if dynamic:
        dynamic_axes = {input_names[0]: {0: "N"}, output_names[0]: {0: "N"}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (img,),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=False,
        external_data=False,
        dynamic_axes=dynamic_axes,
    )

    # Simplify ONNX model
    onnxslim.slim(str(output_path), output_model=str(output_path))

    # Compare output with torch model and ONNX model
    torch_out = model(img).detach().numpy()
    session = onnxruntime.InferenceSession(
        output_path, providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-4)
    except AssertionError as e:
        print(e)
    print(f"Successfully export ONNX model: {output_path}")


if __name__ == "__main__":
    img = cv2.imread("curve/debug.jpg")
    assert img is not None
    img = cv2.dnn.blobFromImage(img, 1, img.shape[:2][::-1], (104, 117, 123))
    img = torch.from_numpy(img)

    model = RetinaFaceWrapper(cfg_re50, img.shape[2])
    convert_onnx(model, img, "onnx_files/retinaface_resnet50.onnx")
    model = RetinaFaceWrapper(cfg_mnet, img.shape[2])
    convert_onnx(model, img, "onnx_files/retinaface_mobilenet025.onnx")
