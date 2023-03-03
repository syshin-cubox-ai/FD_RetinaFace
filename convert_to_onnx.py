import os

import cv2
import numpy as np
import onnx
import onnx.shape_inference
import onnxruntime.tools.symbolic_shape_infer
import onnxsim
import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

from data import cfg_re50, cfg_mnet
from layers.functions.prior_box import priorbox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm


class RetinaFaceWrapper(nn.Module):
    def __init__(self, cfg: dict, img_size: int):
        super().__init__()
        self.cfg = cfg
        match self.cfg['name']:
            case 'Resnet50':
                state_dict = torch.utils.model_zoo.load_url(
                    'https://github.com/syshin-cubox-ai/FD_RetinaFace/releases/'
                    'download/v0.0.1-weights/Resnet50_Final.pth'
                )
            case 'mobilenet0.25':
                state_dict = torch.load('weights/mobilenet0.25_Final.pth')
            case _:
                raise ValueError('Wrong cfg')

        self.model = RetinaFace(self.cfg, phase='test')
        self.model.load_state_dict(state_dict)
        self.priors = priorbox(
            min_sizes=self.cfg['min_sizes'],
            steps=self.cfg['steps'],
            clip=False,
            image_size=(img_size, img_size),
        )
        self.scale = img_size
        self.conf_thres = 0.9
        self.iou_thres = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc, conf, landm = self.model(x)
        loc, conf, landm = loc.squeeze(0), conf.squeeze(0), landm.squeeze(0)

        # Decode
        boxes = decode(loc, self.priors, self.cfg['variance'])
        boxes = boxes * self.scale
        scores = conf[:, 1]
        landmarks = decode_landm(landm, self.priors, self.cfg['variance'])
        landmarks = landmarks * self.scale

        # Ignore low scores
        keep = torch.nonzero(scores > self.conf_thres).squeeze(1)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # NMS
        keep = torchvision.ops.nms(boxes, scores, self.iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        out = torch.cat((boxes, scores.unsqueeze(1), landmarks), 1)
        return out


def convert_onnx(model, img, output_path, opset=17, dynamic=False, simplify=True):
    assert isinstance(model, torch.nn.Module)

    model.eval()
    print("[in progress] torch.onnx.export...")

    # Define input and output names
    input_names = ['image']
    output_names = ['pred']

    # Define dynamic_axes
    if dynamic:
        dynamic_axes = {input_names[0]: {0: 'N'},
                        output_names[0]: {0: 'N'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    try:
        onnx_model = onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
    except Exception as e:
        print(f'ERROR: {e}, skip symbolic shape inference.')
    onnx.shape_inference.infer_shapes_path(output_path, output_path, check_type=True, strict_mode=True, data_prop=True)

    # Compare output with torch model and ONNX model
    torch_out = model(img).detach().numpy()
    session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-4)
    except AssertionError as e:
        print(e)
        stdin = input('Do you want to ignore the error and proceed with the export ([y]/n)? ')
        if stdin == 'n':
            os.remove(output_path)
            exit(1)

    # Simplify ONNX model
    if simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')


if __name__ == '__main__':
    img = cv2.imread('curve/debug.jpg').astype(np.float32)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    model = RetinaFaceWrapper(cfg_re50, img.shape[2])
    # model = RetinaFaceWrapper(cfg_mnet, img.shape[2])

    convert_onnx(model, img, 'onnx_files/retinaface.onnx')
