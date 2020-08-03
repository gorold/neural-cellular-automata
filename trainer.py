import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *

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

def train_step(nca, x, target, steps, optimizer, scheduler):
    nca.train()
    x = nca(x, steps=steps)
    loss = F.mse_loss(to_rgba(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def rank_losses(x, target):
    """
    Returns the indices of losses from descending order (highest to lowest), per sample in the batch.

    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
    target: tensor, (4, h, w)
    """
    loss = torch.mean(torch.pow(to_rgba(x) - target, 2), dim=[1, 2, 3])
    return torch.argsort(loss, descending=True)

def pool_train(nca, target, optimizer, scheduler, epochs, device, pool_size, batch_size, damage_n, fig_dir, model_path, save_epoch=100):
    """
    Training procedure using the 'sample pool' training strategy.

    Parameters
    ----------
    nca
        Neural Cellular Automata model
    target: tensor, (4, h, w)
        Tensor representing the target image. Assumed to be premultiplied RGBA.
    """
    target = pad_target(target).to(device) # (4, h+16, w+16)
    h, w = target.size(1), target.size(2)
    seed = make_seed((h, w), pool_size, nca.channel_n).to(device)
    pool = SamplePool(x=seed)
    batch = pool.sample(pool_size).x

    losses = list()

    for epoch in range(1, epochs + 1):
        steps = int(torch.randint(low=64, high=96+1, size=()))
        batch = pool.sample(batch_size)
        x0 = batch.x
        ranked_losses = rank_losses(x0, target)
        x0 = batch.x[ranked_losses]
        x0[:1] = make_seed((h, w), 1, nca.channel_n).to(device) # change first sample to a seed
        
        if damage_n:
            damage = 1 - make_circle_masks(damage_n, h, w).unsqueeze(1)
            x0[-damage_n:] *= damage.to(device) # mask last damage_n tensors in the first dim

        x, loss = train_step(nca, x0, target, steps, optimizer, scheduler)

        batch.replace(x=x.detach())
        batch.commit()

        losses.append(loss.detach().cpu())

        print(f'Loss (epoch {epoch}): {loss.item()}')
        start_epoch = 1
        if epoch % save_epoch == 0 or epoch == epochs:

            # Save visualizations
            viz_batch(x0.detach().cpu(), x.detach().cpu(), fig_dir, start_epoch, epoch)
            viz_loss(losses, fig_dir, start_epoch, epoch)

            # Save model
            torch.save(nca.state_dict(), model_path)

            start_epoch += save_epoch
