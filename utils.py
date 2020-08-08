import os, io, requests, gc
from errno import EEXIST
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def load_image(img_src, max_size=40):
    """
    Loads an image from img_src and returns a tensor of shape (4, h, w) in premultiplied RGBA format.
    """
    img = PIL.Image.open(img_src)
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
    r = requests.get(url)
    return load_image(io.BytesIO(r.content))

def load_emoji_dict(path):
    """
    Loads a list of emojis from a given file directory as a dict of tensors.
    Returns
    -------
    dict[str->tensor]
        Dictionary of class names to tensor.
    """
    return {f: load_image(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}

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
    return x[:, 3:4, ...].clamp(min=0.0, max=1.0)

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
    return (1.0 - a + rgb).clamp(min=0.0, max=1.0)

def pad_target(target, p=16):
    """
    Pads target with p pixels in top-bottom and left-right dimensions.
    """
    return F.pad(target, [p//2, p//2, p//2, p//2, 0, 0])

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

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """
    try:
        os.makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else: raise
    return mypath

def viz_batch(x0, x, save_path, start_e, end_e):
    """
    Save visualizations from a single update sequence.

    Parameters
    ----------
    x0: tensor
        Initial input cell state.
    x: tensor
        Cell state output.
    """
    vis0 = to_rgb(x0).transpose(1,2).transpose(2,3)
    vis1 = to_rgb(x).transpose(1,2).transpose(2,3)

    plt.figure(figsize=[15, 5])
    for i in range(x0.size(0)):
        plt.subplot(2, x0.size(0), i+1)
        plt.imshow(vis0[i])
        plt.axis('off')
    for i in range(x.size(0)):
        plt.subplot(2, x0.size(0), i+1+x0.size(0))
        plt.imshow(vis1[i])
        plt.axis('off')

    save_path = mkdir_p(os.path.join(save_path, 'batch_viz'))
    save_path = os.path.join(save_path, f'epoch_{start_e}_{end_e}.png')
    plt.savefig(save_path)
    plt.close()

def viz_loss(losses, save_path, start_e, end_e):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(torch.log10(torch.as_tensor(losses)), 'b-', alpha=0.5)
    
    save_path = mkdir_p(os.path.join(save_path, 'losses'))
    save_path = os.path.join(save_path, f'epoch_{start_e}_{end_e}.png')
    plt.savefig(save_path)
    plt.close()

class GCDebug:

    def __init__(self):
        self.count = 0

    def __call__(self):
        print(f'Debugger position: {self.count}')
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), f'cuda: {obj.is_cuda}')
            except:
                pass
        self.count += 1
