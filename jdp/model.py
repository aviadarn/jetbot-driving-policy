"""The 2024 policy: ImageNet ResNet18 with fc = Linear(512, 2) regressing the click (x, y).

Input is what the car's camera gave it: a BGR image in [0, 1]. The 2024 notebooks applied
the ImageNet RGB mean/std to BGR-ordered channels in both training and live use; that quirk
is kept (it is consistent, so harmless) and moved inside the model so the exported ONNX
takes raw BGR/255 and needs no preprocessing on the Nano.
"""
import torch
import torchvision


class SteeringNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet18(weights=weights)
        self.net.fc = torch.nn.Linear(512, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, bgr01):
        return self.net((bgr01 - self.mean) / self.std)


class TorchPolicy:
    """Wraps a trained SteeringNet as a sim policy: RGB uint8 frame -> (x, y)."""

    def __init__(self, model, device):
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img_rgb, pose=None):
        t = torch.from_numpy(img_rgb[..., ::-1].copy()).to(self.device)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return self.model(t)[0].float().cpu().numpy()
