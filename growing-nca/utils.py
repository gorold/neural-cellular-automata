import os, io, base64, zipfile, json, requests, glob
import PIL.Image, PIL.ImageDraw
import numpy as np
import matplotlib.pylab as pl

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def load_image(url, max_size=40):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
    img = transforms.ToTensor()(img)
    img[:3, ...] *= img[3:, ...] # premultiply RGB by Alpha
    return img

def load_emoji(emoji):
    """
    Loads the given emoji as a tensor.
    Usage example:
        emoji = load_emoji('🦎')
    Emojis to choose from:
        🦎😀💥👁🐠🦋🐞🕸🥨🎄
    
    Parameters
    ----------
    emoji: str
        Chosen emoji to load.

    Returns
    -------
    tensor
        (4, h, w)
    """
    emojis = ['🦎', '😀', '💥', '👁', '🐠', '🦋', '🐞', '🕸', '🥨', '🎄']
    assert emoji < len(emojis)
    emoji = emojis[emoji]
    code = hex(ord(emoji))[2:].lower()
    url = f'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u{code}.png'
    return load_image(url)

def to_rgba(x):
    """
    Transforms the state representation of the cellular automata into RGBA format.
    Primarily used in performing loss calculations with target image (which is in RGBA format).

    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
        State representation tensor.

    Returns
    -------
    tensor
        Tensor of shape (batch, 4, h, w).
    """
    return x[:, :4, ...]

def to_alpha(x):
    """
    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
        Tensor where 4th channel represents alpha.

    Returns
    -------
    tensor
        Tensor of size (batch, 1, h, w).
    """
    return x[:, 3:4, ...].clamp(min=0, max=1)

def to_rgb(x):
    """
    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
        Tensor with first 3 channels representing RGB channels, premultiplied by alpha.

    Returns
    -------
    tensor
        Tensor of size (batch, 3, h, w).
    """
    rgb, a = x[:, :3, ...], to_alpha(x)
    return 1.0 - a + rgb

def pad_target(target, p=16):
    """
    Pads target with p pixels in top-bottom and left-right dimensions.
    """
    return F.pad(target, [p//2, p//2, p//2, p//2, 0, 0])

class SamplePool:
    """
    Creates a pool of samples to sample from.
    The keyword argument **slots make up the attributes which will be sampled.
    """

    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = v.size(0)
            assert self._size == v.size(0)
            setattr(self, k, torch.as_tensor(v))

    def sample(self, n):
        """
        For each attribute (from self._slot_names), sample n and create a new SamplePool object to hold them.
        Attributes must be indexable.
        """
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def replace(self, **new_slots):
        """
        Replace current samples with new values.
        """
        for k, v in new_slots.items():
            assert k in self._slot_names
            getattr(self, k)[:] = v

    def commit(self):
        """
        Commit the parent's indexed attribute with the child's attribute.
        """
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def make_seed(size, num_seeds, channel_n):
    """
    Create a seed state, where all cells are 0 except for the middle which is 1.0.
    First three channels (representing RGB) are kept as 0.

    Parameters
    ----------
    h: int
        Height of the state
    w: int
        Width of the state
    pool_size: int
        Number of seeds to create.
    channel_n: int
        Number of channels to represent state.

    Returns
    -------
    tensor
        Tensor of size (pool_size, channel_n, h, w).
    """
    h, w = size
    seed = torch.zeros((num_seeds, channel_n, h, w), dtype=torch.float32)
    seed[:, 3:, h//2, w//2] = 1.0
    return seed

def make_circle_masks(n, h, w):
    """
    Create circle mask on a grid of size (h, w).
    Center and radius chosen randomly from [-0.5, 0.5] and [0.1, 0.4] respectively (for grid scale of [-1, 1]).

    Parameters
    ----------
    n: int
        Number of channels to create mask for.
    h: int
        Height of image.
    w: int
        Width of image.

    Returns
    -------
    tensor
        (n, h, w)
    """
    x = torch.linspace(-1, 1, steps=w).view(1, 1, -1) # x.shape = (1, 1, w)
    y = torch.linspace(-1, 1, steps=h).view(1, -1, 1) # y.shape = (1, h, 1)
    center = torch.rand((2, n, 1, 1)) - 0.5 # center \in [-0.5, 0.5], center.shape = (2, n, 1, 1)
    r = torch.rand((n, 1, 1)) * 0.3 + 0.1 # r \in [0.1, 0.4] # r.shape = (n, 1, 1)
    x = (x - center[0]) / r # x.shape = (n, 1, w)
    y = (y - center[1]) / r # y.shape = (n, h, 1)
    mask = (x * x + y * y < 1.0).float() # mask.shape = (n, h, w)
    return mask
