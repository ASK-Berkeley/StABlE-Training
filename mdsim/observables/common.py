import torch
from copy import deepcopy


class ObservableMSELoss(torch.nn.Module):
    def __init__(self, target_obs):
        super().__init__()
        self.target_obs = target_obs

    def forward(self, pred_obs):
        return (pred_obs - self.target_obs).pow(2).mean()


class ObservableMAELoss(torch.nn.Module):
    def __init__(self, target_obs):
        super().__init__()
        self.target_obs = target_obs

    def forward(self, pred_obs):
        return (pred_obs - self.target_obs).abs().mean()


class IMDHingeLoss(torch.nn.Module):
    def __init__(self, lower_bound):
        super().__init__()
        assert (
            lower_bound.shape[0] == 0 and len(lower_bound.shape) == 1,
            "IMD target observable bound must be a scalar",
        )
        self.lower_bound = lower_bound

    def forward(self, pred_obs):
        return torch.exp(torch.clamp(self.lower_bound - pred_obs, min=0)) - 1


class BondLengthDeviation(torch.nn.Module):
    def __init__(self, name, bonds, mean_bond_lens, cell, device):
        super(BondLengthDeviation, self).__init__()
        self.name = name
        self.bonds = bonds
        self.mean_bond_lens = mean_bond_lens
        self.cell = cell
        self.device = device

    def forward(self, stacked_radii):
        if self.name == "lips":  # account for non-cubic cell
            dists = torch.stack(
                [
                    compute_distance_matrix_batch(self.cell, radii)
                    for radii in stacked_radii
                ]
            )
            bond_lens = dists[:, :, self.bonds[:, 0], self.bonds[:, 1]]
        else:
            if self.name == "water":
                # only keep local bonds
                n_atoms_local = stacked_radii.shape[-2]
                bonds = self.bonds[: int(n_atoms_local * 2 / 3)]
            else:
                bonds = self.bonds
            bond_lens = distance_pbc(
                stacked_radii[:, :, bonds[:, 0]],
                stacked_radii[:, :, bonds[:, 1]],
                torch.diag(self.cell),
            ).to(self.device)
        max_bond_dev_per_replica = (
            (bond_lens - self.mean_bond_lens[: bonds.shape[0]])
            .abs()
            .max(dim=-1)[0]
            .max(dim=0)[0]
            .detach()
        )
        return bond_lens, max_bond_dev_per_replica


def radii_to_dists(radii, params):
    # Get rij matrix
    r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
    # get rid of diagonal 0 entries of r matrix (for gradient stability)
    r = r[:, ~torch.eye(r.shape[1], dtype=bool)].reshape(r.shape[0], r.shape[1], -1, 3)
    try:
        r.requires_grad = True
    except RuntimeError:
        pass

    # compute distance matrix:
    return torch.sqrt(torch.sum(r**2, axis=-1)).unsqueeze(-1)


"""
Below functions were taken from 
https://github.com/kyonofx/MDsim/blob/main/observable.ipynb
"""


def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1, 1, 3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta**2).sum(dim=-1))


# different from previous functions, now needs to deal with non-cubic cells.
def compute_distance_matrix_batch(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    # NxNxCells distance array
    shifted = cart_coords.unsqueeze(2) + shifts.unsqueeze(0).unsqueeze(0)
    dist = cart_coords.unsqueeze(2).unsqueeze(2) - shifted.unsqueeze(1)
    dist = dist.pow(2).sum(dim=-1).sqrt()
    # But we want only min
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix


def get_smoothed_diffusivity(xyz):
    seq_len = xyz.shape[0] - 1
    diff = torch.zeros(seq_len)
    msd = deepcopy(diff)
    for i in range(seq_len):
        _diff, _msd = get_diffusivity_traj(xyz[i:].transpose(0, 1).unsqueeze(0))
        diff[: seq_len - i] += _diff.flatten()
        msd[: seq_len - i] += _msd.flatten()
    diff = diff / torch.flip(torch.arange(seq_len) + 1, dims=[0])
    msd = msd / torch.flip(torch.arange(seq_len) + 1, dims=[0])
    return diff, msd


def get_diffusivity_traj(pos_seq, dilation=1):
    """
    Input: B x N x T x 3
    Output: B x T
    """
    # substract CoM
    bsize, time_steps = pos_seq.shape[0], pos_seq.shape[2]
    pos_seq = pos_seq - pos_seq.mean(1, keepdims=True)
    msd = (
        (pos_seq[:, :, 1:] - pos_seq[:, :, 0].unsqueeze(2))
        .pow(2)
        .sum(dim=-1)
        .mean(dim=1)
    )
    diff = msd / (torch.arange(1, time_steps) * dilation) / 6
    return diff.view(bsize, time_steps - 1), msd.view(bsize, time_steps - 1)
