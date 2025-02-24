import torch
import numpy as np
import math
import os
from torchmd.observable import DifferentiableADF
from mdsim.common.utils import data_to_atoms
from mdsim.observables.common import distance_pbc
from mdsim.datasets.lmdb_dataset import LmdbDataset
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.observables.common import get_smoothed_diffusivity


# Water utils
class WaterRDFMAE(torch.nn.Module):
    """
    Mean Absolute Error in Water RDF.
    """

    def __init__(self, base_path, gt_rdfs, n_atoms, params, device):
        super(WaterRDFMAE, self).__init__()
        self.gt_rdfs = gt_rdfs
        self.n_atoms = n_atoms
        self.device = device
        self.params = params
        self.xlim = params.max_rdf_dist
        n_bins = int(self.xlim / params.dr)
        self.bins = np.linspace(1e-6, self.xlim, n_bins + 1)  # for computing RDF
        # get ground truth data
        DATAPATH = os.path.join(base_path, "water", "1k", "val/nequip_npz.npz")
        gt_data = np.load(DATAPATH, allow_pickle=True)
        self.ptypes = torch.tensor(gt_data.f.atom_types)
        self.lattices = torch.tensor(gt_data.f.lengths[0]).float()

    def forward(self, stacked_radii):
        # Expected shape of stacked_radii is [Timesteps, N_replicas, N_atoms, 3]
        max_maes = []
        rdf_list = []
        for i in range(
            stacked_radii.shape[1]
        ):  # explicit loop since vmap makes some numpy things weird
            rdfs = get_water_rdfs(
                stacked_radii[:, i],
                self.ptypes[: stacked_radii.shape[-2]],
                self.lattices,
                self.bins,
                self.device,
            )
            # compute MAEs of all element-conditioned RDFs
            max_maes.append(
                torch.max(
                    torch.cat(
                        [
                            self.xlim * torch.abs(rdf - gt_rdf).mean().unsqueeze(-1)
                            for rdf, gt_rdf in zip(rdfs.values(), self.gt_rdfs.values())
                        ]
                    )
                )
            )
            rdf_list.append(torch.cat([rdf.flatten() for rdf in rdfs.values()]))
        return torch.stack(rdf_list).to(self.device), torch.stack(max_maes).to(
            self.device
        )


class MinimumIntermolecularDistance(torch.nn.Module):
    """
    Minimum distance between two atoms on different water molecules.
    """

    def __init__(self, bonds, cell, device, element_mask=None):
        super(MinimumIntermolecularDistance, self).__init__()
        self.cell = cell
        self.device = device
        # construct a tensor containing all the intermolecular bonds
        self.element_mask = element_mask

    def forward(self, stacked_radii):
        num_atoms = stacked_radii.shape[-2]
        missing_edges = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if not (
                    (i % 3 == 0 and (j == i + 1 or j == i + 2))
                    or (j % 3 == 0 and (i == j + 1 or i == j + 2))
                ):  # i and j are not on the same atom
                    if self.element_mask == "O":
                        if i % 3 == 0 and j % 3 == 0:  # both oxygen
                            missing_edges.append([i, j])
                    elif self.element_mask == "H":
                        if i % 3 != 0 and j % 3 != 0:  # both hydrogen
                            missing_edges.append([i, j])
                    elif self.element_mask is None:
                        missing_edges.append([i, j])
        self.intermolecular_edges = torch.Tensor(missing_edges).to(torch.long)
        stacked_radii = ((stacked_radii / torch.diag(self.cell)) % 1) * torch.diag(
            self.cell
        )  # wrap coords
        intermolecular_distances = distance_pbc(
            stacked_radii[:, :, self.intermolecular_edges[:, 0]],
            stacked_radii[:, :, self.intermolecular_edges[:, 1]],
            torch.diag(self.cell),
        ).to(
            self.device
        )  # compute distances under minimum image convention
        return intermolecular_distances.min(dim=-1)[0].min(dim=0)[0].detach()


def n_closest_molecules(_xyz, center_idx, n, lattices):
    """
    Returns the coordinates of the n-closest molecules to the center atom (and the center atom itself)
    """
    # wrap coordinates
    xyz = ((_xyz / torch.diag(lattices)) % 1) * torch.diag(lattices)  # wrap coords
    distances = distance_pbc(xyz[:, center_idx].unsqueeze(1), xyz, torch.diag(lattices))
    sorted_distances, idxs = torch.sort(distances, dim=-1)
    molecules = torch.floor(idxs / 3)
    unique_molecules = torch.stack(
        [first_unique_elements(mol, n + 1) for mol in molecules]
    )
    atom_idxs = torch.stack(
        [expand_molecules_to_atoms(mol) for mol in unique_molecules]
    ).long()
    # return original (unwrapped) coordinates and corresponding indices
    final_coords = torch.stack(
        [_xyz[i, atom_idx.long()] for i, atom_idx in enumerate(atom_idxs)]
    )
    return final_coords, atom_idxs


def expand_molecules_to_atoms(molecule_idxs):
    atom_idxs = []
    for idx in molecule_idxs:
        atom_idxs.append(3 * idx)
        atom_idxs.append(3 * idx + 1)
        atom_idxs.append(3 * idx + 2)
    return torch.Tensor(atom_idxs)


def first_unique_elements(tensor, num_elements):
    # Create an empty list to store unique elements
    unique_elements = []
    # Iterate through the tensor
    for element in tensor:
        # Check if the element is not already in the unique_elements list
        if element.item() not in unique_elements:
            # Add the element to the unique_elements list
            unique_elements.append(element.item())
            # Break the loop if we have reached the required number of unique elements
            if len(unique_elements) == num_elements:
                break

    return torch.Tensor(unique_elements)


def find_water_rdfs_diffusivity_from_file(base_path: str, size: str, params, device):
    n_closest = params.n_closest_molecules
    xlim = params.max_rdf_dist
    n_bins = int(xlim / params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1)  # for computing RDF

    # get ground truth data
    DATAPATH = os.path.join(base_path, "water", size, "test/nequip_npz.npz")
    gt_data = np.load(DATAPATH, allow_pickle=True)
    atom_types = torch.tensor(gt_data.f.atom_types)
    oxygen_atoms_mask = atom_types == 8
    lattices = torch.tensor(gt_data.f.lengths[0]).float()
    gt_traj = torch.tensor(gt_data.f.unwrapped_coords)
    gt_data_continuous = np.load(
        os.path.join(base_path, "contiguous-water", "1k", "test/nequip_npz.npz")
    )
    gt_traj_continuous = torch.tensor(gt_data_continuous.f.unwrapped_coords)
    gt_diffusivity, gt_msd = get_smoothed_diffusivity(
        gt_traj_continuous[0::100, atom_types == 8]
    )  # track diffusivity of oxygen atoms, unit is A^2/ps
    gt_diffusivity = gt_diffusivity[:100].to(device)
    gt_msd = gt_msd[:100].to(device)
    # recording frequency of underlying data is 10 fs.
    # Want to match frequency of our data collection which is params.n_dump*params.integrator_config["dt"]
    keep_freq = math.ceil(params.n_dump * params.timestep / 10)
    gt_rdfs = get_water_rdfs(gt_traj[::keep_freq], atom_types, lattices, bins, device)
    # local rdfs
    gt_local_neighborhoods = torch.cat(
        [
            n_closest_molecules(
                gt_traj[::keep_freq], i, n_closest, torch.diag(lattices)
            )[0]
            for i in range(64)
        ]
    )
    gt_rdfs_local = get_water_rdfs(
        gt_local_neighborhoods[::keep_freq],
        atom_types[: gt_local_neighborhoods.shape[-2]],
        lattices,
        bins,
        device,
    )

    # ADF
    temp_data = LmdbDataset({"src": os.path.join(base_path, "water", size, "train")})
    init_data = temp_data.__getitem__(0)
    atoms = data_to_atoms(init_data)
    # extract bond and atom type information
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T
    gt_adf = DifferentiableADF(
        gt_traj.shape[-2], bonds, torch.diag(lattices).to(device), params, device
    )(gt_traj[0:200][::keep_freq].to(torch.float).to(device))
    # TODO: O-O conditioned RDF using oxygen_atoms_mask
    return gt_rdfs, gt_rdfs_local, gt_diffusivity, gt_msd, gt_adf, oxygen_atoms_mask


"""
Below functions were taken from https://github.com/kyonofx/MDsim/blob/main/observable.ipynb
"""


def distance_pbc_select(x, lattices, indices0, indices1):
    x0 = x[:, indices0]
    x1 = x[:, indices1]
    x0_size = x0.shape[1]
    x1_size = x1.shape[1]
    x0 = x0.repeat([1, x1_size, 1])
    x1 = x1.repeat_interleave(x0_size, dim=1)
    delta = torch.abs(x0 - x1)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta**2).sum(axis=-1))


def get_water_rdfs(data_seq, ptypes, lattices, bins, device="cpu"):
    """
    get atom-type conditioned water RDF curves.
    """
    data_seq = data_seq.to(device).float()
    lattices = lattices.to(device).float()
    type2indices = {"H": ptypes == 1, "O": ptypes == 8}
    pairs = [("O", "O"), ("H", "H"), ("H", "O")]

    data_seq = ((data_seq / lattices) % 1) * lattices  # coords are wrapped
    all_rdfs = {}
    n_rdfs = 3
    for idx in range(n_rdfs):
        type1, type2 = pairs[idx]
        indices0 = type2indices[type1].to(device)
        indices1 = type2indices[type2].to(device)

        data_pdist = distance_pbc_select(data_seq, lattices, indices0, indices1)
        data_pdist = data_pdist.cpu().numpy()
        data_shape = data_pdist.shape[1]
        data_hists = np.stack([np.histogram(dist, bins)[0] for dist in data_pdist])
        rho_data = data_shape / torch.prod(lattices, dim=-1)
        rho_data = rho_data.cpu().numpy()
        Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        data_rdfs = data_hists / Z_data
        data_rdf_mean = data_rdfs.mean(0)
        all_rdfs[type1 + type2] = torch.Tensor(np.array(([data_rdf_mean]))).to(device)
    return all_rdfs
