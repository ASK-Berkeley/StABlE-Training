from mdsim.common.utils import cleanup_atoms_batch
import torch
from nequip.data import AtomicDataDict
from torch_geometric.nn import radius_graph


class ForceCalculator:
    """
    Class (similar to ASE Calculator) to compute energies and forces for a
    given set of atomic positions using a machine learning force field.
    Has a few StABlE-specific modifications, including
        1. Batched input (simulating multiple replicas)
        2. Option to output individual atomic energies (needed for localized Boltzmann estimator)
    """

    def __init__(
        self,
        model,
        model_type,
        model_config,
        r_max_key,
        n_atoms,
        atomic_numbers,
        atoms_batch,
        pbc,
        cell,
        device,
    ):
        self.model = model
        self.model_type = model_type
        self.model_config = model_config
        self.r_max_key = r_max_key
        self.n_atoms = n_atoms
        self.atoms_batch = atoms_batch
        self.atomic_numbers = atomic_numbers
        self.pbc = pbc
        self.cell = cell
        self.device = device

    def calculate_energy_force(
        self, radii, retain_grad=False, output_individual_energies=False
    ):
        """
        Calculate energies and forces of a set of atomic positions.
        Args:
            radii (torch.Tensor): Atom positions (Shape: (N_replicas, N_atoms, 3))
            retain_grad (bool): Whether to store the computational graph
                                of the force calculation (default: False)
            output_individual_energies (bool): whether to output individual atomic energies
                                                in addition to global energy (default: False)
        Returns:
            Global potential energy, individual atomic energies (optional), forces
        """

        batch_size = radii.shape[0]
        batch = torch.arange(batch_size).repeat_interleave(self.n_atoms).to(self.device)
        with torch.enable_grad():
            if not radii.requires_grad:
                radii.requires_grad = True

            # Wrap positions: this is crucial for stable simulation of systems with PBC
            # see the (now fixed) bug in the FairChem repo: https://github.com/FAIR-Chem/fairchem/pull/783
            if self.pbc:
                diag = torch.diag(self.cell)
                radii = ((radii / diag) % 1) * diag - diag / 2

            # assign radii and batch
            self.atoms_batch["pos"] = radii.reshape(-1, 3)
            self.atoms_batch["batch"] = batch
            # make these match the number of replicas (different from n_replicas when doing bottom-up stuff)
            self.atoms_batch["cell"] = (
                self.atoms_batch["cell"][0].unsqueeze(0).repeat(batch_size, 1, 1)
            )
            self.atoms_batch["pbc"] = (
                self.atoms_batch["pbc"][0].unsqueeze(0).repeat(batch_size, 1)
            )
            all_energies = None
            if self.model_type == "nequip":
                self.atoms_batch["atom_types"] = self.atoms_batch["atom_types"][
                    0 : self.n_atoms
                ].repeat(batch_size, 1)
                # recompute neighbor list
                self.atoms_batch["edge_index"] = radius_graph(
                    radii.reshape(-1, 3),
                    r=self.model_config["model"][self.r_max_key],
                    batch=batch,
                    max_num_neighbors=32,
                )
                # Assumes cubic cell
                self.atoms_batch["edge_cell_shift"] = torch.zeros(
                    (self.atoms_batch["edge_index"].shape[1], 3)
                ).to(self.device)
                ptr = torch.cat(
                    [
                        torch.tensor([0]).to(self.device),
                        torch.cumsum(self.atoms_batch["natoms"], 0),
                    ]
                )
                self.atoms_batch["ptr"] = ptr.long()
                atoms_updated = self.model(self.atoms_batch)
                energy = atoms_updated[AtomicDataDict.TOTAL_ENERGY_KEY]
                forces = atoms_updated[AtomicDataDict.FORCE_KEY].reshape(
                    -1, self.n_atoms, 3
                )
            else:
                self.atoms_batch = cleanup_atoms_batch(self.atoms_batch)
                self.atoms_batch["natoms"] = (
                    torch.Tensor([self.n_atoms]).repeat(batch_size).to(self.device)
                )
                self.atoms_batch["atomic_numbers"] = self.atomic_numbers.repeat(
                    batch_size
                )
                if output_individual_energies:
                    if self.model_type == "gemnet_t":
                        energy, all_energies, forces = self.model(
                            self.atoms_batch, output_individual_energies=True
                        )
                    else:
                        raise RuntimeError(
                            f"Outputting individual energies is only supported for gemnet_t, not {self.model_type}"
                        )
                else:
                    energy, forces = self.model(self.atoms_batch)
                forces = forces.reshape(-1, self.n_atoms, 3)
            assert not torch.any(torch.isnan(forces))
            energy = energy if retain_grad else energy.detach()
            forces = forces if retain_grad else forces.detach()
            if all_energies is None:
                return energy, forces
            else:
                return energy, all_energies, forces
