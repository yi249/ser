from torchvision import transforms as torch_transforms
import torch
from torch import Tensor

class RandomRandomRotation(torch_transforms.RandomRotation):
    def __init__(
        self, *args, p=1.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p=p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        F = torch_transforms.functional
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        if torch.rand(1) < self.p:
            angle = self.get_params(self.degrees)
        else:
            angle = 0
        return F.rotate(img, angle, self.resample, self.expand, self.center, fill)

def transforms(*stages):
    return torch_transforms.Compose(
        [
            torch_transforms.ToTensor(),
            *stages,
        ]
    )


def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def flip(p=1):
    """
    Flip a tensor both vertically and horizontally
    """
    return torch_transforms.Compose(
        [
            RandomRandomRotation(degrees=180,p=p)
        ]
    )
