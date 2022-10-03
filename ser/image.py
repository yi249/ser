from ser.data import test_dataloader
from ser.transforms import transforms, normalize

def load_image(label):
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return(images)