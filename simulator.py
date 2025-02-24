import numpy as np
import warnings
import gsd.hoomd
import torch
import math
import os
from tqdm import tqdm
from torch_geometric.nn import radius_graph
from torch.utils.data import DataLoader
import ase
from ase import units
import nequip.scripts.deploy
from nequip.train.loss import Loss
from nequip.data import AtomicData
from nequip.utils.torch_geometric import Batch
from nequip.utils import load_file
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.md.ase_utils import OCPCalculator
from mdsim.md.calculator import ForceCalculator
from mdsim.md.integrator import NoseHoover, Langevin
from copy import deepcopy
from mdsim.common.registry import registry
from mdsim.common.utils import data_to_atoms, initialize_velocities, dump_params_to_yml
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.observables.common import (
    distance_pbc,
    BondLengthDeviation,
    compute_distance_matrix_batch,
)
from mdsim.observables.water import WaterRDFMAE, MinimumIntermolecularDistance
from simulator_utils import create_frame, thermo_log


class Simulator:
    """
    Simulator class which performs forward MD simulations with a machine learning force field (MLFF).
    This class stores all attributes about the simulations, including the current states of all the replicas,
    and whether or not they have become unstable.
    """

    def __init__(self, config, params, model, model_path, model_config, gt_rdf):
        super(Simulator, self).__init__()
        print("Initializing MD simulation environment")
        self.params = params
        # GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        # Set config params
        self.name = config["name"]
        self.pbc = self.name == "water"
        self.molecule = config["molecule"] if "molecule" in config else ""
        self.size = config["size"]
        self.model_type = config["model"]
        self.l_max = config["l_max"]
        self.all_unstable = False
        self.first_simulation = True
        self.config = config
        self.data_dir = config["src"]
        self.train = params.train
        self.n_replicas = config["n_replicas"]
        self.minibatch_size = config["minibatch_size"]
        self.gradient_clipping = config["gradient_clipping"]
        self.shuffle = config["shuffle"]
        self.reset_probability = config["reset_probability"]
        self.vacf_window = config["vacf_window"]
        self.optimizer = config["optimizer"]
        self.adjoint = config["adjoint"]
        self.stability_tol = config["imd_tol"] if self.pbc else config["bond_dev_tol"]
        self.max_frac_unstable_threshold = config["max_frac_unstable_threshold"]
        self.min_frac_unstable_threshold = config["min_frac_unstable_threshold"]
        self.results_dir = os.path.join(config["log_dir"], config["results_dir"])
        self.eval_model = config["eval_model"]
        self.n_dump = config["n_dump"]
        if self.name == "water":
            self.n_local_neighborhoods = config["n_local_neighborhoods"]
            self.n_closest_molecules = config["n_closest_molecules"]
            self.n_atoms_local = 3 * (self.n_closest_molecules + 1)
        self.n_dump_vacf = config["n_dump_vacf"]

        # Initialize model
        self.curr_model_path = model_path
        self.model = model
        if self.model_type == "nequip":
            self.rescale_layers = []
            outer_layer = self.model
            while hasattr(outer_layer, "unscale"):
                self.rescale_layers.append(outer_layer)
                outer_layer = getattr(outer_layer, "model", None)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model_config = model_config

        # initialize datasets
        self.DATAPATH_TRAIN = os.path.join(
            self.data_dir, self.name, self.molecule, self.size, "train"
        )
        self.DATAPATH_TEST = os.path.join(
            self.data_dir, self.name, self.molecule, self.size, "test"
        )

        self.train_dataset = LmdbDataset({"src": self.DATAPATH_TRAIN})
        self.test_dataset = LmdbDataset({"src": self.DATAPATH_TEST})
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=data_list_collater,
            batch_size=self.minibatch_size,
        )

        # initialize system properties

        # get random initial condition from dataset
        init_data = self.train_dataset.__getitem__(10)
        self.n_atoms = init_data["pos"].shape[0]
        self.atoms = data_to_atoms(init_data)

        # extract bond and atom type information
        NL = NeighborList(natural_cutoffs(self.atoms), self_interaction=False)
        NL.update(self.atoms)
        self.bonds = (
            torch.tensor(NL.get_connectivity_matrix().todense().nonzero())
            .to(self.device)
            .T
        )
        # filter out extra edges (don't know why they're there)
        if self.name == "water":
            mask = torch.abs(self.bonds[:, 0] - self.bonds[:, 1]) <= 2
            self.bonds = self.bonds[mask]

        self.atom_types = self.atoms.get_chemical_symbols()
        # atom type mapping
        if self.model_type == "nequip":
            type_names = self.model_config["model"][
                nequip.scripts.deploy.TYPE_NAMES_KEY
            ]
            species_to_type_name = {s: s for s in ase.data.chemical_symbols}
            type_name_to_index = {n: i for i, n in enumerate(type_names)}
            chemical_symbol_to_type = {
                sym: type_name_to_index[species_to_type_name[sym]]
                for sym in ase.data.chemical_symbols
                if sym in type_name_to_index
            }
            self.typeid = np.zeros(self.n_atoms, dtype=int)
            for i, _type in enumerate(self.atom_types):
                self.typeid[i] = chemical_symbol_to_type[_type]
            self.final_atom_types = (
                torch.Tensor(self.typeid)
                .repeat(self.n_replicas)
                .to(self.device)
                .to(torch.long)
                .unsqueeze(-1)
            )

        gt_data_train = np.load(os.path.join(self.DATAPATH_TRAIN, "nequip_npz.npz"))
        if self.name == "water":
            pos_field_train = gt_data_train.f.wrapped_coords
            self.gt_data_spacing_fs = 10
        else:
            pos_field_train = gt_data_train.f.R
            if self.name == "md17":
                self.gt_data_spacing_fs = 0.5
            else:
                self.gt_data_spacing_fs = 1

        self.gt_traj_train = torch.FloatTensor(pos_field_train).to(self.device)

        # set initial stability metric values
        self.instability_per_replica = (
            100 * torch.ones((self.n_replicas,)).to(self.device)
            if params.stability_criterion == "imd"
            else torch.zeros((self.n_replicas,)).to(self.device)
        )

        # amount of time each replica has been stable
        self.stable_time = torch.zeros((self.n_replicas,)).to(self.device)

        self.masses = torch.Tensor(self.atoms.get_masses().reshape(1, -1, 1)).to(
            self.device
        )
        self.r_max_key = "r_max" if self.model_type == "nequip" else "cutoff"

        self.nsteps = params.steps
        self.eq_steps = params.eq_steps
        # ensure that the number of logged steps is a multiple of the vacf window (for chopping up the trajectory)
        self.nsteps -= (self.nsteps - self.eq_steps) % self.vacf_window
        if (self.nsteps - self.eq_steps) < self.vacf_window:
            self.nsteps = self.eq_steps + 2 * self.vacf_window  # at least two windows
        while (
            self.nsteps < params.steps
        ):  # nsteps should be at least as long as what was requested
            self.nsteps += self.vacf_window
        self.ps_per_epoch = self.nsteps * self.config["timestep"] // 1000.0

        self.temp = self.config["temperature"] * units.kB
        self.atomic_numbers = (
            torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        )
        self.batch = (
            torch.arange(self.n_replicas)
            .repeat_interleave(self.n_atoms)
            .to(self.device)
        )
        self.ic_stddev = params.ic_stddev

        dataset = self.train_dataset if self.train else self.test_dataset
        # pick the first n_replicas ICs if doing simulation (so we can directly compare replicas' stabilities across models)
        samples = (
            np.random.choice(
                np.arange(dataset.__len__()), self.n_replicas, replace=False
            )
            if self.train
            else np.arange(self.n_replicas)
        )

        self.raw_atoms = [data_to_atoms(dataset.__getitem__(i)) for i in samples]
        self.cell = torch.Tensor(self.raw_atoms[0].cell).to(self.device)

        bond_lens = torch.stack(
            [
                distance_pbc(
                    gt_traj_train.unsqueeze(0)[:, self.bonds[:, 0]],
                    gt_traj_train.unsqueeze(0)[:, self.bonds[:, 1]],
                    torch.diag(self.cell).to(self.device),
                ).mean(dim=0)
                for gt_traj_train in self.gt_traj_train
            ]
        )
        self.mean_bond_lens = bond_lens.mean(0)
        self.bond_lens_var = bond_lens.var(0)

        self.gt_rdf = gt_rdf
        # choose the appropriate stability criterion based on the type of system
        if self.name == "water":
            if params.stability_criterion == "imd":
                self.stability_criterion = MinimumIntermolecularDistance(
                    self.bonds, self.cell, self.device
                )
            else:
                self.stability_criterion = WaterRDFMAE(
                    self.data_dir, self.gt_rdf, self.n_atoms, self.params, self.device
                )
            self.rdf_mae = WaterRDFMAE(
                self.data_dir, self.gt_rdf, self.n_atoms, self.params, self.device
            )
            self.min_imd = MinimumIntermolecularDistance(
                self.bonds, self.cell, self.device
            )
            self.bond_length_dev = BondLengthDeviation(
                self.name, self.bonds, self.mean_bond_lens, self.cell, self.device
            )

        else:
            self.stability_criterion = BondLengthDeviation(
                self.name, self.bonds, self.mean_bond_lens, self.cell, self.device
            )

        # Set initial positions and velocities
        radii = torch.stack(
            [torch.Tensor(atoms.get_positions()) for atoms in self.raw_atoms]
        )
        self.radii = (radii + torch.normal(torch.zeros_like(radii), self.ic_stddev)).to(
            self.device
        )
        self.velocities = torch.Tensor(
            initialize_velocities(self.n_atoms, self.masses, self.temp, self.n_replicas)
        ).to(self.device)
        # assign velocities to atoms
        for i in range(len(self.raw_atoms)):
            self.raw_atoms[i].set_velocities(self.velocities[i].cpu().numpy())
        # initialize checkpoint states for resetting
        self.checkpoint_radii = []
        self.checkpoint_radii.append(self.radii)
        self.checkpoint_velocities = []
        self.checkpoint_velocities.append(self.velocities)
        self.checkpoint_zetas = []
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
        self.checkpoint_zetas.append(self.zeta)
        self.original_radii = self.radii.clone()
        self.original_velocities = self.velocities.clone()
        self.original_zeta = self.zeta.clone()
        self.all_radii = []

        # create batch of atoms to be operated on
        self.atoms_batch = [
            AtomicData.from_ase(
                atoms=a, r_max=self.model_config["model"][self.r_max_key]
            )
            for a in self.raw_atoms
        ]
        self.atoms_batch = Batch.from_data_list(
            self.atoms_batch
        )  # DataBatch for non-Nequip models
        self.atoms_batch["natoms"] = (
            torch.Tensor([self.n_atoms]).repeat(self.n_replicas).to(self.device)
        )
        self.atoms_batch["cell"] = self.atoms_batch["cell"].to(self.device)
        self.atoms_batch["atomic_numbers"] = (
            self.atoms_batch["atomic_numbers"].squeeze().to(torch.long).to(self.device)
        )
        # convert to dict for Nequip
        if self.model_type == "nequip":
            self.atoms_batch = AtomicData.to_AtomicDataDict(self.atoms_batch)
            self.atoms_batch = {
                k: v.to(self.device) for k, v in self.atoms_batch.items()
            }
            self.atoms_batch["atom_types"] = self.final_atom_types
            del self.atoms_batch["ptr"]
            del self.atoms_batch["atomic_numbers"]

        # Initialize calculator
        self.calculator = ForceCalculator(
            self.model,
            self.model_type,
            self.model_config,
            self.r_max_key,
            self.n_atoms,
            self.atomic_numbers,
            self.atoms_batch,
            self.pbc,
            self.cell,
            self.device,
        )

        # Initialize integrator
        self.dt = config["timestep"]
        self.integrator_type = self.config["integrator"]
        self.integrator = registry.get_integrator_class(self.integrator_type)(
            self.calculator,
            self.masses,
            self.n_replicas,
            self.n_atoms,
            self.config,
            self.device,
        )

        self.diameter_viz = params.diameter_viz
        self.exp_name = params.exp_name
        self.training_observable = params.training_observable
        self.obs_loss_weight = params.obs_loss_weight
        self.energy_force_loss_weight = params.energy_force_loss_weight

        # limit CPU usage
        torch.set_num_threads(10)

        molecule_for_name = self.name if self.name == "water" else self.molecule
        name = f"{self.model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}"
        self.save_dir = (
            os.path.join(self.results_dir, name)
            if self.train
            else os.path.join(self.results_dir, name, "inference", self.eval_model)
        )
        os.makedirs(self.save_dir, exist_ok=True)
        dump_params_to_yml(self.params, self.save_dir)
        # File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")

        # initialize trainer to calculate energy/force gradients
        if self.train:
            if self.model_type == "nequip":
                self.train_dict = load_file(
                    supported_formats=dict(
                        torch=["pth", "pt"], yaml=["yaml"], json=["json"]
                    ),
                    filename=os.path.join(self.save_dir, "trainer.pth"),
                    enforced_format=None,
                )
                self.nequip_loss = Loss(coeffs=self.train_dict["loss_coeffs"])
            else:
                self.trainer = OCPCalculator(
                    config_yml=self.model_config,
                    checkpoint=self.curr_model_path,
                    test_data_src=self.DATAPATH_TEST,
                    energy_units_to_eV=1.0,
                ).trainer
                self.trainer.model = self.model

    def stability_per_replica(self):
        """
        Check whether each replica has become unstable based on the specified threshold
        Returns:
            stability (torch.Tensor): Boolean indicating whether each replica is stable (Shape: (N_replicas,))
        """
        stability = (
            (self.instability_per_replica > self.stability_tol)
            if self.params.stability_criterion == "imd"
            else (self.instability_per_replica < self.stability_tol)
        )
        return stability

    def set_starting_states(self):
        """
        Sets the starting states (positions, velocities, thermostat zeta parameter)
        for the next round of MD simulation based on whether we are in a simulation or learning phase

        Returns:
            Fraction of replicas which are unstable (float between 0 and 1)
        """

        # find replicas which violate the stability criterion
        reset_replicas = ~self.stability_per_replica()
        num_unstable_replicas = reset_replicas.count_nonzero().item()
        if num_unstable_replicas / self.n_replicas >= self.max_frac_unstable_threshold:
            if not self.all_unstable:
                # threshold of unstable replicas reached, transition from simulation to learning phase
                print(
                    "Threshold of unstable replicas has been reached... Start Learning"
                )
            self.all_unstable = True

        if (
            self.all_unstable
        ):  # we are in a learning phase already, so reset all replicas to the initial values
            self.radii = self.original_radii
            self.velocities = self.original_velocities
            self.zeta = self.original_zeta

        else:  # we are in a simulation phase
            if (
                self.first_simulation
            ):  # in the first epoch of every simulation phase, randomly reset some fraction of replicas to an earlier time
                # pick replicas to reset (only pick from stable replicas)
                stable_replicas = (~reset_replicas).nonzero()
                random_reset_idxs = torch.randperm(stable_replicas.shape[0])[
                    0 : math.ceil(self.reset_probability * stable_replicas.shape[0])
                ].to(self.device)
                # uniformly sample times to reset each replica to
                reset_times = torch.randint(
                    0, len(self.checkpoint_radii), (len(random_reset_idxs),)
                ).to(self.device)

                for idx, time in zip(random_reset_idxs, reset_times):
                    self.radii[idx] = self.checkpoint_radii[time][idx]
                    self.velocities[idx] = self.checkpoint_velocities[time][idx]
                    self.zeta[idx] = self.checkpoint_zetas[time][idx]

            # reset the replicas which are unstable, and continue simulating the rest
            exp_reset_replicas = (
                (reset_replicas).unsqueeze(-1).unsqueeze(-1).expand_as(self.radii)
            )
            self.radii = torch.where(
                exp_reset_replicas,
                self.original_radii.detach().clone(),
                self.radii.detach().clone(),
            ).requires_grad_(True)
            self.velocities = torch.where(
                exp_reset_replicas,
                self.original_velocities.detach().clone(),
                self.velocities.detach().clone(),
            ).requires_grad_(True)
            self.zeta = torch.where(
                reset_replicas.unsqueeze(-1).unsqueeze(-1),
                self.original_zeta.detach().clone(),
                self.zeta.detach().clone(),
            ).requires_grad_(True)

            # update stability times for each replica
            if not self.train:
                self.stable_time = torch.where(
                    reset_replicas,
                    self.stable_time,
                    self.stable_time + self.ps_per_epoch,
                )
        self.first_simulation = False
        return num_unstable_replicas / self.n_replicas

    def solve(self):
        """
        Performs a forward Molecular Dynamics simulation with the Simulator's MLFF calculator, saving the trajectory for
        the subsequent Boltzmann estimator computation, and tracking and logging various metrics along the way.
        At the end of the simulation, checks whether each replica has become unstable.
        """

        self.mode = (
            "learning" if self.optimizer.param_groups[0]["lr"] > 0 else "simulation"
        )
        self.running_vels = []
        self.running_accs = []
        self.running_radii = []
        self.original_radii = self.radii.clone()
        self.original_velocities = self.velocities.clone()
        self.original_zeta = self.zeta.clone()

        # log checkpoint states for resetting
        if not self.all_unstable:
            self.checkpoint_radii.append(self.original_radii)
            self.checkpoint_velocities.append(self.original_velocities)
            self.checkpoint_zetas.append(self.original_zeta)
        # File dump
        if self.train or self.epoch == 0:  # create one long simulation for inference
            try:
                self.t = gsd.hoomd.open(
                    name=f"{self.save_dir}/sim_epoch{self.epoch+1}_{self.mode}.gsd",
                    mode="w",
                )
            except:
                warnings.warn("Unable to save simulation trajectories")

        with torch.no_grad():
            self.step = -1
            # Initialize forces/potential of starting configuration
            _, self.forces = self.integrator.calculator.calculate_energy_force(
                self.radii
            )

            ######### Begin MD ###########

            print("Start MD trajectory", file=self.f)
            for step in tqdm(range(self.nsteps)):
                self.step = step
                # MD Step
                radii, velocities, energy, forces, zeta = self.integrator.step(
                    self.radii,
                    self.velocities,
                    self.forces,
                    self.zeta,
                    retain_grad=False,
                )
                # dump frames
                if self.step % self.n_dump == 0:
                    print(
                        self.step,
                        thermo_log(
                            self.radii,
                            self.velocities,
                            energy,
                            self.masses,
                            self.n_atoms,
                            self.stability_criterion,
                            self.bond_length_dev if self.pbc else None,
                            self.rdf_mae if self.pbc else None,
                            self.pbc,
                        ),
                        file=self.f,
                    )
                    step = (
                        self.step if self.train else (self.epoch + 1) * self.step
                    )  # don't overwrite previous epochs at inference time

                    if hasattr(self, "t"):
                        self.t.append(
                            create_frame(
                                self.radii,
                                self.velocities,
                                self.cell,
                                self.bonds,
                                self.pbc,
                                self.diameter_viz,
                                self.n_atoms,
                                self.dt,
                                self.name,
                                frame=step / self.n_dump,
                            )
                        )
                # save trajectory for gradient calculation
                if step >= self.eq_steps:
                    self.running_radii.append(radii.detach().clone())
                    self.running_vels.append(velocities.detach().clone())
                    self.running_accs.append((forces / self.masses).detach().clone())

                    if step % self.n_dump == 0 and not self.train:
                        self.all_radii.append(
                            radii.detach().cpu()
                        )  # save whole trajectory without resetting at inference time

                # update state
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.forces.copy_(forces)
                if self.integrator_type == "NoseHoover":
                    self.zeta.copy_(zeta)

            ######### End MD ###########

            # combine radii
            self.stacked_radii = torch.stack(self.running_radii[:: self.n_dump])

            # compute instability metric for all replicas (either bond length deviation, min intermolecular distance, or RDF MAE)
            self.instability_per_replica = self.stability_criterion(self.stacked_radii)

            if isinstance(self.instability_per_replica, tuple):
                self.instability_per_replica = self.instability_per_replica[-1]
            self.mean_instability = self.instability_per_replica.mean()
            if self.pbc:
                self.mean_bond_length_dev = self.bond_length_dev(self.stacked_radii)[
                    1
                ].mean()
                self.mean_rdf_mae = self.rdf_mae(self.stacked_radii)[-1].mean()
            self.stacked_vels = torch.cat(self.running_vels)

        if self.train:
            try:
                self.t.close()
            except:
                pass
        return self
