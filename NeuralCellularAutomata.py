from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

class GrowingNCA(nn.Module):
    """
    Attributes
    ----------
    channel_n: int
        Number of channels for the Cellular Automata representation. 
        Channels 1, 2, 3 represent the RGB channels respectively.
        Channel 4 represents the alpha channel which demarcates living cells for alpha > 0.1
    fire_rate: float \in (0, 1]
        Probability to update a particular cell in the stochastic cell update.
    """

    def __init__(self, device, channel_n=16, fire_rate=0.5, hidden_size=128):
        super(GrowingNCA, self).__init__()
        assert fire_rate > 0 and fire_rate <= 1

        self.update_rules = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channel_n*3, hidden_size, 1)), # inputs -> 3 filters x channel_n
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(hidden_size, channel_n, 1)), # outputs -> channel_n (interpreted as update rules)
        ]))
        nn.init.zeros_(self.update_rules.conv2.weight)
        nn.init.zeros_(self.update_rules.conv2.bias)
        self.to(device)

        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.device = device

    def alive(self, x):
        """
        Returns a map indicating which cells are 'alive'.
        A cell is 'alive' if minimally one cell in its 3x3 neighborhood (or itself) has alpha > 0.1.
        Recall that alpha is the 4th channel.

        Parameters
        ----------
        x: tensor, (batch, channel_n, height, width)
            The input tensor to check.
        
        Returns
        -------
        tensor
            A tensor of shape (batch, 1, height, width), whose values are 1 if the cell is alive and 0 if dead.
        """
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def _perceive(self, x, w):
        """
        Applies the 

        F.conv2d(input, weight, ...):
            input.shape = [batch, in_channels, iH, iW]
            weight.shape = [out_channels, in_channels/groups, kH, kW]

        Parameters
        ----------
        x: tensor, (batch, channel_n, height, width)
            The input tensor to perform convolution on.
        w: tensor, (3, 3)
            A weight vector to use to perform convolution on the input vector, x.
        Returns
        -------
        tensor
            A tensor of shape (batch, channel_n, height, width)
        """
        w = w.to(self.device)
        w = w.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1) # results in w.shape = [self.channel_n, 1, 3, 3]
        return F.conv2d(x, w, padding=1, groups=self.channel_n)

    def perceive(self, x, angle):
        """
        Returns a feature map processed by fixed filters, 'Cell Identity', 'Sobel_x', 'Sobel_y'.

        Parameters
        ----------
        x: tensor, (batch, channel_n, height, width)
            The input tensor to perform "perception" on.

        Returns
        -------
        tensor
            A tensor of shape (batch, channel_n*3, height, width).
        """
        identify = torch.tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
             dtype=torch.float32,
             requires_grad=False
        )
        dx = torch.tensor(
            [[-1,  0,  1],
             [-2,  0,  2],
             [-1,  0,  1]],
             dtype=torch.float32,
             requires_grad=False
        ) / 8.0
        dy = dx.T 
        c = torch.cos(torch.tensor(angle))
        s = torch.sin(torch.tensor(angle))

        w1 = c * dx - s * dy
        w2 = s * dx + c * dy
        
        y1 = self._perceive(x, w1)
        y2 = self._perceive(x, w2)
        y = torch.cat((x, y1, y2), dim=1)

        return y

    def update(self, x, fire_rate, angle, step_size):
        """

        Parameters
        ----------
        x: tensor, (batch, channel_n, height, width)

        Returns
        -------
        tensor
            (batch, channel_n, height, width)
        """
        # Hmmm don't think they mentioned pre_life_mask/post_life_mask in the paper
        pre_life_mask = self.alive(x)

        y = self.perceive(x, angle)
        dx = self.update_rules(y) * step_size

        stochastic_mask = torch.rand([dx.size(0), 1, dx.size(2), dx.size(3)]) > fire_rate
        stochastic_mask = stochastic_mask.float().to(self.device)
        x =  x + (dx * stochastic_mask)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float() 
        x = x * life_mask

        return x


    def forward(self, x, steps=1, fire_rate=None, angle=0.0, step_size=1.0):
        if fire_rate is None:
            fire_rate = self.fire_rate
        for step in range(steps):
            x = self.update(x, fire_rate, angle, step_size)
        return x

class NewNCA(nn.Module):
    
    def __init__(self, ):
        super(NewNCA, self).__init__()

    def forward(self, x):
        pass
    