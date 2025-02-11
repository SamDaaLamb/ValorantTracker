# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', 'path/to/yolov5s.onnx')  # file from branch
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates or loads a YOLOv5 model

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.yolo import Model
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)

        if pretrained and channels == 3 and classes == 80:
            model = DetectMultiBackend(path, device=device, fuse=autoshape)  # download/load FP32 model
            # model = models.experimental.attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-nano model https://github.com/ultralytics/yolov5
    return _create('yolov5n', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create('yolov5m', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create('yolov5l', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create('yolov5x', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-nano-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5n6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-small-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5s6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-medium-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5m6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-large-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5l6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-xlarge-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5x6', pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == '__main__':
    model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Verify inference
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2

    imgs = [
        'data/images/zidane.jpg',  # filename
        Path('data/images/zidane.jpg'),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy
    
    results = model(imgs, size=320)  # batched inference
    results.print()
    results.save()

dependencies = ['torch']


# from demo.ASPP import SRDetectModel
# import torch
import torch.nn as nn

# === Basic Convolution Block ===
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# === Bottleneck CSP Block ===
class BottleneckCSP(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv3 = Conv(hidden_channels, hidden_channels, 3, 1)
        self.conv4 = Conv(hidden_channels * 2, out_channels, 1, 1)
        self.n = n

    def forward(self, x):
        y1 = self.conv1(x)
        for _ in range(self.n):
            y1 = self.conv3(y1)
        y2 = self.conv2(x)
        return self.conv4(torch.cat((y1, y2), dim=1))

# === Spatial Pyramid Pooling (SPP) ===
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv2 = Conv(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1))

# === Backbone (Feature Extractor) ===
class Backbone(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.focus = Conv(3, 64, 3)
        self.conv1 = Conv(64, 128, 3, 2)
        self.bottleneck1 = BottleneckCSP(128, 128, 3)
        self.conv2 = Conv(128, 256, 3, 2)
        self.bottleneck2 = BottleneckCSP(256, 256, 9)
        self.conv3 = Conv(256, 512, 3, 2)
        self.bottleneck3 = BottleneckCSP(512, 512, 9)
        self.conv4 = Conv(512, 1024, 3, 2)
        self.spp = SPP(1024, 1024)
        self.bottleneck4 = BottleneckCSP(1024, 1024, 3)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.conv2(x)
        x = self.bottleneck2(x)
        x = self.conv3(x)
        x = self.bottleneck3(x)
        x = self.conv4(x)
        x = self.spp(x)
        x = self.bottleneck4(x)
        return x

# === Head (Upsampling & Detection) ===
class Head(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = Conv(1024, 512, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv(512, 256, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.detect = nn.Conv2d(256, nc * (5 + nc), 1, 1)  # Detection layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.detect(x)
        return x

# === YOLOv5 Model (Backbone + Head) ===
class YOLOv5(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.backbone = Backbone(nc)
        self.head = Head(nc)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# === Test the Model ===
if __name__ == "__main__":
    nc = 80  # Number of classes (change this for your dataset)
    model = YOLOv5(nc)
    x = torch.randn(1, 3, 640, 640)  # Example input tensor
    y = model(x)
    print("Output shape:", y.shape)  # Expected output shape for YOLOv5

def srdetect():
    # model = YOLOv5(1)
    # checkpoint = torch.hub.load_state_dict_from_url('https://github.com/SamDaaLamb/ValorantTracker/blob/main/runs/train/weights/best.pt?raw=true', map_location="cpu")
    # state_dict = {key.replace("net.", ""): value for key, value in checkpoint["state_dict"].items()}
    # model.load_state_dict(state_dict)
    # return model
    model = YOLOv5(1)
    
    # Load the checkpoint file
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://github.com/SamDaaLamb/ValorantTracker/blob/main/runs/train/weights/best.pt?raw=true', 
        map_location="cpu"
    )
    
    # Print the keys to check the file structure
    print("Checkpoint Keys:", checkpoint.keys())

    return model

