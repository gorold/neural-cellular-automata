from math import ceil

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils import *
from dataloaders import *
from NeuralCellularAutomata import *

@torch.no_grad()
def rank_losses(x, target):
    """
    Returns the indices of losses from descending order (highest to lowest).

    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
    target: tensor, (4, h, w)

    Returns
    -------
    tensor
        Tensor representing the indices of losses in descending order.
    """

    loss = torch.mean(torch.pow(to_rgba(x) - target, 2), dim=[1, 2, 3]).detach().cpu()
    return torch.argsort(loss, descending=True)

def train_step(nca, x0, target, steps, optimizer, scheduler, split=8):
    nca.train()
    xs = []
    total_loss = 0
    for x, t in zip(torch.split(x0, split), torch.split(target, split)):
        if isinstance(nca, GrowingNCA):
            x = nca(x, steps=steps)
        elif isinstance(nca, ConditionalNCA):
            x = nca(x, t, steps=steps)
        
        loss = F.mse_loss(to_rgba(x), t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        xs.append(x)
        total_loss += loss.detach().cpu()
    
    x = torch.cat(xs, dim=0)
    total_loss /= x0.size(0)

    return x, float(loss)

def train_step_vae(nca, x0, target, t_aug, steps, optimizer, scheduler, writer, epoch, save_epoch, split=8):
    nca.train()
    xs = []
    x_recons = []
    total_loss = 0
    for x, t, t_o in zip(torch.split(x0, split), torch.split(target, split), torch.split(t_aug, split)):
        x, x_recon, mu, logvar = nca(x, t_o, steps=steps)
        mse_loss = F.mse_loss(to_rgba(x), t, reduction='sum')
        vae_mse_loss = F.mse_loss(x_recon, t_o, reduction='sum')
        kld_loss = 1*(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        loss = (mse_loss + kld_loss + vae_mse_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        xs.append(x)
        if enable_vae:
            x_recons.append(x_recon)
        total_loss += loss.detach().cpu()
    
    x = torch.cat(xs, dim=0)
    x_recon = torch.cat(x_recons, dim=0)

    if epoch % save_epoch == 0:
        writer.add_image('VAE_Progress', make_grid(x_recon), global_step = epoch)
    total_loss /= x0.size(0)
    writer.add_scalar('MSE loss', mse_loss.detach().item(), epoch)
    writer.add_scalar('VAE MSE loss', vae_mse_loss.detach().item(), epoch)
    writer.add_scalar('KLD loss', kld_loss.detach().item(), epoch)

    return x, float(loss)

def pool_train(nca, target, optimizer, scheduler, epochs, device, steps_low, steps_high, pool_size, batch_size, damage_n, fig_dir, model_path, save_epoch=100):
    """
    Training procedure using the 'sample pool' training strategy.

    Parameters
    ----------
    target: tensor, (4, h, w)
        Tensor representing the target image. Assumed to be premultiplied RGBA.
    """
    target = pad_target(target).to(device) # (4, h+16, w+16). Store (single) target on cuda
    h, w = target.size(1), target.size(2)
    seed = make_seed((h, w), pool_size, nca.channel_n) # do not store seeds on cuda
    pool = SamplePool(x=seed)

    losses = list()

    start_epoch = 1 # used for saving figures
    for epoch in range(1, epochs + 1):
        steps = int(torch.randint(low=steps_low, high=steps_high+1, size=()))
        batch = pool.sample(batch_size)
        x0 = batch.x.to(device)
        ranked_losses = rank_losses(x0, target)
        x0 = batch.x[ranked_losses].to(device)
        x0[:1] = make_seed((h, w), 1, nca.channel_n).to(device) # change first sample to a seed
        
        if damage_n:
            damage = 1 - make_circle_masks(damage_n, h, w).unsqueeze(1).to(device)
            x0[-damage_n:] *= damage # mask last damage_n tensors in the first dim

        x, loss = train_step(nca, x0, target, steps, optimizer, scheduler)

        batch.replace(x=x.detach().cpu())
        batch.commit()

        losses.append(loss)

        print(f'Loss (epoch {epoch}): {loss}')
        if epoch % save_epoch == 0 or epoch == epochs:

            # Save visualizations
            viz_batch(x0.detach().cpu(), x.detach().cpu(), fig_dir, start_epoch, epoch)
            viz_loss(losses, fig_dir, 1, epoch) # entire loss history
            viz_loss(losses[start_epoch-1:epoch-1], fig_dir, start_epoch, epoch) # from previous save point

            # Save model
            torch.save(nca.state_dict(), model_path)

            start_epoch += save_epoch

def conditional_pool_train(nca, targets, optimizer, scheduler, epochs, device, steps_low, steps_high, pool_size, batch_size, damage_n, fig_dir, model_path, enable_vae, save_epoch=100):
    """
    targets: dict[str->tensor]
        Dict mapping target class name to tensor of size (4, h, w).
    pool_size: int
        The number of seeds in the SamplePool per class.
    batch_size: int
        The number of samples per class for SamplePool. 
        Each batch will have total (num_emojis * batch_size) samples.
    damage_n: int
        Number of images to damage per class.
    """
    writer = SummaryWriter(fig_dir)
    targets = {k: pad_target(v) for k, v in targets.items()} # do not store all targets on cuda    
    seeds = {c: make_seed((targets[c].size(1), targets[c].size(2)), pool_size, nca.channel_n) for c in targets}
    pool = ConditionalSamplePool(targets=targets, **seeds) # do not store seeds on cuda

    losses = list()
    graph_model = False

    start_epoch = 1 # used for saving figures
    for epoch in range(1, epochs+1):
        steps = int(torch.randint(low=steps_low, high=steps_high+1, size=()))
        batch = pool.sample(batch_size)
        for k in batch._slot_names:
            h, w = getattr(batch, k).size(2), getattr(batch, k).size(3)
            x0 = getattr(batch, k).to(device) # send x0 to cuda
            t = ConditionalSamplePool.t_container[k].to(device) # send t to cuda
            ranked_losses = rank_losses(x0, t)
            setattr(batch, k, getattr(batch, k).to(device)) # send batch seeds to cuda
            getattr(batch, k)[ranked_losses[0]:ranked_losses[0]+1] = make_seed((h, w), 1, nca.channel_n).to(device)

            if damage_n:
                damage = 1 - make_circle_masks(damage_n, h, w).unsqueeze(1).to(device)
                getattr(batch, k)[ranked_losses[-damage_n:]] *= damage

        x0 = batch.x_tensor # already on cuda
        t = batch.targets_tensor.to(device)
        t_aug = batch.targets_augmented.to(device)
        if not graph_model:
            writer.add_graph(nca, (torch.rand_like(x0), torch.rand_like(t_aug)))
            graph_model = True

        if enable_vae:
            x, loss = train_step(nca, x0, t, t_aug, steps, optimizer, scheduler, writer, epoch, save_epoch)
        else:
            x, loss = train_step(nca, x0, t, steps, optimizer, scheduler)

        batch.replace(x.detach().cpu())
        batch.commit()

        losses.append(loss)
        writer.add_scalar('training loss', losses[-1], epoch)
    
        print(f'Loss (epoch {epoch}): {loss}')
        if epoch % save_epoch == 0 or epoch == epochs:

            # Save visualizations
            viz_batch(x0.detach().cpu(), x.detach().cpu(), fig_dir, start_epoch, epoch)
            viz_loss(losses, fig_dir, 1, epoch) # entire loss history
            viz_loss(losses[start_epoch-1:epoch-1], fig_dir, start_epoch, epoch) # from previous save point

            writer.add_image('CA_Progress', make_grid(to_rgb(x)), global_step = epoch)
            writer.add_image('VAE_Targets', make_grid(t_aug), global_step = epoch)
            writer.add_image('CA_Targets', make_grid(t), global_step = epoch)

            # Save model
            torch.save(nca.state_dict(), model_path)

            start_epoch += save_epoch
