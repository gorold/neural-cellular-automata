from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from utils import *

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
            ('conv1', nn.Conv2d(channel_n*3, hidden_size, 1)), # inputs -> 3 filters * channel_n
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


class NCAEncoder(nn.Module):
    def __init__(self, latent_dims = 2):
        super(NCAEncoder, self).__init__()
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 8, 7, stride=3, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(8, 16, 3, stride=1, padding=0)),
            ('relu2', nn.ReLU()),
            ('adaptive_mp', nn.AdaptiveMaxPool2d(4)), # 4 x 4 x 16 = 256
            ('flatten', nn.Flatten()),
        ]))

        self.fc_mu = nn.Linear(in_features = 256, out_features = latent_dims)
        self.fc_logvar = nn.Linear(in_features = 256, out_features = latent_dims)
            
    def forward(self, x):
        '''
        Parameters
        ----------
            x torch.tensor: tensor of shape [batch_size, 4, h, w]. h, w = 56 by default
        
        Returns
        -------
            x_mu torch.tensor: tensor of shape [batch_size, latent_dim]
            x_logvar torch.tensor: tensor of shape [batch_size, latent_dim]
        '''
        x = self.body(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class NCADecoder(nn.Module):
    def __init__(self, latent_dims = 2, height = 56, width = 56, output_channel = 16):
        super(NCADecoder, self).__init__()

        assert height % 4 == 0 and width % 4 == 0, 'Height and Width need to be divisible by 4'

        self.h = height//4
        self.w = width//4

        self.fc1 = nn.Linear(latent_dims, out_features= 256*self.h*self.w)
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, output_channel, 4, stride=2, padding=1)

    def forward(self, x):
        '''
        Parameters
        ----------
            x torch.tensor: tensor of shape [batch_size, latent_dim].
        
        Returns
        -------
            x torch.tensor: tensor of shape [batch_size, 16, 56, 56]
        '''
        x = self.fc1(x)
        x = x.view(x.size(0), 256, self.h, self.w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class NCAVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims = 2, output_width = 56, output_height = 56, output_channel = 16):
        super(NCAVariationalAutoencoder, self).__init__()
        self.encoder = NCAEncoder(latent_dims = latent_dims)
        self.decoder = NCADecoder(latent_dims = latent_dims, height = output_height, width = output_width, output_channel = output_channel)
    
    def forward(self, x):
        '''
        Parameters
        ----------
            x torch.tensor: tensor of shape [batch_size, 4, h, w]. h, w = 56 by default
        
        Returns
        -------
            x torch.tensor: tensor of shape [batch_size, output_channel, output_width, output_height]
            latent_mu torch.tensor: tensor of shape [batch_size, latent_dim]
            latent_logvar torch.tensor: tensor of shape [batch_size, latent_dim]
        '''
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        '''
        Function to sample from the latent distribution. Returns 0.5*e^log(var)+mu
        Parameters
        ----------
            mu torch.tensor: tensor of shape [batch_size, latent_dim]
            logvar torch.tensor: tensor of shape [batch_size, latent_dim].
        
        Returns
        -------
            sampled_tensor torch.tensor: tensor of shape [batch_size, latent_dim]
        '''
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_() #define normal distribution
            return eps.mul(std).add_(mu) #sample from normal distribution
        else:
            return mu

class ConditionalNCA(nn.Module):
    def __init__(self, device, channel_n=16, fire_rate=0.5, hidden_size=128, latent_dims = 2):
        super(ConditionalNCA, self).__init__()
        assert fire_rate > 0 and fire_rate <= 1

        # Modify the encoder to more of a VAE kind of architecture
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(4, 8, 7, stride=3, padding=1)),
        #     ('relu1', nn.ReLU()),
        #     ('conv2', nn.Conv2d(8, 16, 3, stride=1, padding=0)),
        #     ('relu2', nn.ReLU()),
        #     ('adaptive_mp', nn.AdaptiveMaxPool2d(4)), # 4 x 4 x 16 = 256
        #     ('flatten', nn.Flatten()),
        #     ('fc1', nn.Linear(256, 128)),
        #     ('relu3', nn.ReLU()),
        #     ('fc2', nn.Linear(128, 16)),
        # ]))
        self.encoder = NCAVariationalAutoencoder(latent_dims = latent_dims)

        self.update_rules = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channel_n*3 + 16, hidden_size, 1)), # inputs -> 3 filters * channel_n
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

    def update(self, x, encoding, fire_rate, angle, step_size):
        """
        Performs single update step to input x.

        Parameters
        ----------
        x: tensor, (batch, channel_n, height, width)
        encoding: tensor, (batch, , )

        Returns
        -------
        tensor
            (batch, channel_n, height, width)
        """
        pre_life_mask = self.alive(x)

        y = self.perceive(x, angle)
        # encoding = encoding.expand(-1, -1, y.size(2), y.size(3))
        y = torch.cat((y, encoding), dim=1)
        dx = self.update_rules(y) * step_size

        stochastic_mask = torch.rand([dx.size(0), 1, dx.size(2), dx.size(3)]) > fire_rate
        stochastic_mask = stochastic_mask.float().to(self.device)
        x =  x + (dx * stochastic_mask)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask

        return x

    def get_encoding(self, target):
        """
        Parameters
        ----------
        target: tensor, (batch, 4, h, w)

        Returns
        -------
        tensor:
            Shape (batch, ?, 1, 1)
        """
        # return self.encoder(target).view(target.size(0), -1, 1, 1)
        encoding, latent_mu, latent_logvar = self.encoder(target)
        return encoding, latent_mu, latent_logvar

    def forward(self, x, target, encoding=None, steps=1, fire_rate=None, angle=0.0, step_size=1.0):
        """
        Parameters
        ----------
        x: tensor, (batch, channel_n, h, w)
        target: tensor, (batch, 4, h, w)

        Returns
        -------
        x torch.tensor:
            Shape (batch, channel_n, h, w)
        mu torch.tensor:
            Shape (batch, latent_dims)
        logvar torch.tensor:
            Shape (batch, latent_dims)
        """
        if fire_rate is None:
            fire_rate = self.fire_rate

        if encoding is None:
            encoding, latent_mu, latent_logvar = self.get_encoding(target)

        for step in range(steps):
            x = self.update(x, encoding, fire_rate, angle, step_size)

        if self.training:
            return x, latent_mu, latent_logvar
        else:
            return x
    