
from ser.transforms import transforms, normalize, flip

import  numpy as np
from PIL import Image

def transform():
    return(transforms(*[normalize,flip]))


def test_both():
    x = np.asarray([[[0,0,1],[0,0,0],[0,0,0]]])
    img = Image.fromarray((x).astype(np.uint8))
    y = np.asarray([[[0,0,0],[0,0,0],[0,0,1]]])
    expected = Image.fromarray((y).astype(np.uint8))
    assert transform()(img) == expected 


def test_flip():
    x = np.asarray([[[0,0,1],[1,0,0],[0,1,0]]])
    img = Image.fromarray((x).astype(np.uint8))
    y = np.asarray([[[0,1,0],[1,0,0],[0,0,1]]])
    expected = Image.fromarray((y).astype(np.uint8))
    output = flip()(img)
    assert expected == output