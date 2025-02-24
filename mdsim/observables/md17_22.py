import torch
import numpy as np
import os
from torchmd.observable import DifferentiableADF
from mdsim.common.utils import data_to_atoms
from mdsim.datasets.lmdb_dataset import LmdbDataset
from ase.neighborlist import natural_cutoffs, NeighborList
from ase import units


# MD17/MD22 utils
def get_hr(traj, bins):
    """
    compute h(r) (the RDF) for MD17 and MD22 simulations.
    traj: T x N_atoms x 3
    Adapted from https://github.com/kyonofx/MDsim/blob/main/observable.ipynb
    """
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    hist[0] = 0  # sometimes there's a weird nonzero element at the beginning
    return hist


def get_hr_reweighted(traj, energies, bins, original_temp, new_temp):
    energies -= energies.mean()
    weights = np.exp(-energies / units.kB * (1 / new_temp - 1 / original_temp))
    weights /= weights.sum()
    N_eff = (-weights * np.log(weights)).sum().exp()  # effective sample size
    # compute hrs weighted by weights
    hrs = np.stack([get_hr(t.unsqueeze(0), bins) for t in traj])
    hr_reweighted = (weights * hrs).sum(0)
    return hr_reweighted, N_eff


def find_hr_adf_from_file(
    base_path: str, name: str, molecule: str, size: str, params, device
):
    # RDF plotting parameters
    xlim = params.max_rdf_dist
    n_bins = int(xlim / params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1)  # for computing h(r)
    # load ground truth data
    DATAPATH = os.path.join(base_path, name, molecule, size, "train/nequip_npz.npz")
    gt_data = np.load(DATAPATH)
    gt_traj = torch.FloatTensor(gt_data.f.R)
    gt_energies = torch.FloatTensor(gt_data.f.E)
    hist_gt, N_eff = get_hr_reweighted(
        gt_traj, gt_energies, bins, 500, params.temperature
    )
    print(
        f"Effective sample size for RDF computation reduced from {gt_energies.shape[0]} to {int(N_eff)} due to reweighting"
    )
    hist_gt = 100 * hist_gt / hist_gt.sum()
    # ADF
    temp_data = LmdbDataset(
        {"src": os.path.join(base_path, name, molecule, size, "train")}
    )
    init_data = temp_data.__getitem__(0)
    n_atoms = init_data["pos"].shape[0]
    atoms = data_to_atoms(init_data)
    # extract bond and atom type information
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T
    raw_atoms = data_to_atoms(temp_data.__getitem__(0))
    cell = torch.Tensor(raw_atoms.cell).to(device)
    diff_adf = DifferentiableADF(n_atoms, bonds, cell, params, device)
    keep = 50 if name == "md17" else 10  # save memory for md22
    hist_adf = diff_adf(gt_traj[:keep].to(device))
    return torch.Tensor(hist_gt).to(device), torch.Tensor(hist_adf).to(device)
