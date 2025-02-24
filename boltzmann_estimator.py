import numpy as np
import torch
import math
from nff.utils.scatter import compute_grad
import os
from tqdm import tqdm
from torchmd.observable import DifferentiableRDF, DifferentiableADF, DifferentiableVACF
from functorch import vmap
from ase import units
from mdsim.common.utils import process_gradient
from simulator_utils import energy_force_gradient
from mdsim.observables.common import radii_to_dists, ObservableMSELoss, IMDHingeLoss
from mdsim.observables.water import n_closest_molecules


class BoltzmannEstimator:
    def __init__(
        self,
        gt_rdf_package,
        mean_bond_lens,
        gt_vacf,
        gt_adf,
        params,
        device,
    ):
        super(BoltzmannEstimator, self).__init__()
        self.params = params
        self.device = device

        # Water system
        if self.params.name == "water":
            gt_rdf, gt_rdf_local = gt_rdf_package
            # Training to match RDF
            if params.training_observable == "rdf":
                self.gt_obs = torch.cat(
                    [rdf.flatten() for rdf in gt_rdf.values()]
                )  # combine RDFs together
                self.gt_obs_local = torch.cat(
                    [rdf.flatten() for rdf in gt_rdf_local.values()]
                )  # combine RDFs together
                self.obs_loss = ObservableMSELoss(self.gt_obs)
            elif params.training_observable == "imd":
                # Training to match minimum intermolecular distance
                self.gt_obs = torch.Tensor([1.44]).to(
                    self.device
                )  # ground truth min IMD
                self.gt_obs_local = self.gt_obs
                self.obs_loss = IMDHingeLoss(self.gt_obs)
            elif params.training_observable == "bond_length_dev":
                self.gt_obs = mean_bond_lens[:2].to(
                    self.device
                )  # ground truth bond length dev
                self.gt_obs_local = self.gt_obs
                self.obs_loss = ObservableMSELoss(self.gt_obs)

        else:  # MD17 or MD22 system
            self.gt_obs = gt_rdf_package[0]
            self.gt_obs_local = self.gt_obs
            self.obs_loss = ObservableMSELoss(self.gt_obs)

        # Initialize remaining observables and losses
        self.gt_vacf = gt_vacf
        self.gt_adf = gt_adf

        self.adf_loss = ObservableMSELoss(self.gt_adf)
        self.vacf_loss = ObservableMSELoss(self.gt_vacf)

    def estimator(self, g, df_dtheta, mse_gradient):
        """
        Computes the Boltzmann gradient estimator described in Eqn. 6 of
        the paper for a minibatch of simulation states (either global or
        local states depending on the system). Note: Technically, we should scale
        the estimator by k_B*T, but we skip this step as it is essentially
        like scaling the learning rate.
        Args:
            g (torch.Tensor): Simulation observable estimates (Shape: (minibatch_size, d_observable))
            df_dtheta (torch.Tensor): Potential energy gradients (Shape: (minibatch_size, num_model_parameters))
            mse_gradient (torch.Tensor): Gradient of the upstream observable loss function,
                                        averaged over the minibatch (Shape: (1, d_observable))
        Returns:
            estimator (torch.Tensor): Gradient of the observable component of the StABlE loss
                                        w.r.t model parameters (Shape: (num_model_parameters))
        """

        jacobian = (
            df_dtheta.mean(0).unsqueeze(0) * g.mean(0).unsqueeze(-1)
            - df_dtheta.unsqueeze(1) * g.unsqueeze(-1)
        ).mean(dim=0)
        # compute Vector-Jacobian product to obtain the final gradient
        estimator = torch.mm(
            mse_gradient.to(torch.float32), jacobian.to(torch.float32)
        )[0]
        return estimator.detach()

    def compute(self, simulator):
        self.simulator = simulator

        # initialize observable computation functions
        diff_rdf = DifferentiableRDF(self.params, self.device)
        diff_adf = DifferentiableADF(
            self.simulator.n_atoms,
            self.simulator.bonds,
            self.simulator.cell,
            self.params,
            self.device,
        )
        diff_vacf = DifferentiableVACF(self.params, self.device)

        model = simulator.model
        # find which replicas are unstable
        stable_replicas = self.simulator.stability_per_replica()
        # if focusing on accuracy, always only keep stable replicas for gradient calculation
        if not self.simulator.all_unstable:
            mask = stable_replicas
        # if focusing on stability, option to only keep unstable replicas for gradient calculation
        else:
            mask = (
                ~stable_replicas
                if (self.params.only_train_on_unstable_replicas)
                else torch.ones((self.simulator.n_replicas), dtype=torch.bool).to(
                    self.device
                )
            )
        # store original shapes of model parameters
        original_numel = [param.data.numel() for param in model.parameters()]
        original_shapes = [param.data.shape for param in model.parameters()]

        # get continuous trajectories (permute to make replica dimension come first)
        radii_traj = torch.stack(self.simulator.running_radii)

        # take i.i.d samples for RDF loss
        stacked_radii = radii_traj[:: self.simulator.n_dump]
        if self.simulator.mode == "learning":
            # save replicas
            np.save(
                os.path.join(
                    self.simulator.save_dir,
                    f"stacked_radii_epoch{self.simulator.epoch}.npy",
                ),
                stacked_radii.cpu(),
            )
            np.save(
                os.path.join(
                    self.simulator.save_dir,
                    f"stable_replicas_epoch{self.simulator.epoch}.npy",
                ),
                stable_replicas.cpu(),
            )

        velocity_subsample_ratio = math.ceil(
            self.simulator.gt_data_spacing_fs / (self.simulator.dt / units.fs)
        )  # make the vacf spacing the same as the underlying GT data
        velocities_traj = torch.stack(simulator.running_vels).permute(1, 0, 2, 3)[
            :, ::velocity_subsample_ratio
        ]
        vacfs_per_replica = vmap(diff_vacf)(velocities_traj)
        vacfs_per_replica[~stable_replicas] = torch.zeros(1, 100).to(
            self.device
        )  # zero out the unstable replica vacfs
        # split into sub-trajectories of length = vacf_window
        velocities_traj = velocities_traj[
            :,
            : math.floor(velocities_traj.shape[1] / self.simulator.vacf_window)
            * self.simulator.vacf_window,
        ]
        velocities_traj = velocities_traj.reshape(
            velocities_traj.shape[0],
            -1,
            self.simulator.vacf_window,
            self.simulator.n_atoms,
            3,
        )

        # sample i.i.d paths
        velocities_traj = velocities_traj[:, :: self.simulator.n_dump_vacf]

        vacfs = vmap(vmap(diff_vacf))(velocities_traj)
        mean_vacf = vacfs.mean(dim=(0, 1))
        mean_vacf_loss = self.vacf_loss(mean_vacf)

        # energy/force loss
        if (
            self.simulator.energy_force_loss_weight != 0
            and self.simulator.train
            and simulator.optimizer.param_groups[0]["lr"] > 0
        ):
            energy_force_package = (energy_force_gradient(self.simulator),)
        else:
            energy_force_package = None

        # VACF stuff
        vacf_package = (
            vacfs_per_replica,
            mean_vacf_loss.to(self.device),
        )

        ###Main Observable Stuff ###
        if self.simulator.pbc:  # Water
            if self.simulator.training_observable == "imd":
                obs = torch.cat(
                    [self.simulator.min_imd(s.unsqueeze(0)) for s in stacked_radii]
                )
            elif self.simulator.training_observable == "rdf":
                obs = torch.cat(
                    [self.simulator.rdf_mae(s.unsqueeze(0))[0] for s in stacked_radii]
                )
            elif self.simulator.training_observable == "bond_length_dev":
                obs = torch.cat(
                    [
                        self.simulator.bond_length_dev(s.unsqueeze(0))[0]
                        for s in stacked_radii
                    ]
                )

            obs = obs.reshape(-1, simulator.n_replicas, self.gt_obs_local.shape[-1])
            adfs = torch.stack(
                [
                    diff_adf(rad)
                    for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)
                ]
            ).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1])

        else:  # observable is always RDF
            r2d = lambda r: radii_to_dists(r, self.simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(
                -1, self.simulator.n_atoms, self.simulator.n_atoms - 1, 1
            )

            obs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(
                -1, self.simulator.n_replicas, self.gt_obs.shape[-1]
            )  # this way of calculating uses less memory

            adfs = torch.stack(
                [
                    diff_adf(rad)
                    for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)
                ]
            ).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1])

        np.save(
            os.path.join(
                self.simulator.save_dir, f"obs_epoch{self.simulator.epoch}.npy"
            ),
            obs.cpu(),
        )
        mean_obs = obs.mean(dim=(0, 1))
        mean_adf = adfs.mean(dim=(0, 1))
        mean_obs_loss = self.obs_loss(mean_obs)
        mean_adf_loss = self.adf_loss(mean_adf)

        # If we aren't in the learning phase, return without computing the Boltzmann estimator
        if (
            self.params.obs_loss_weight == 0
            or not self.simulator.train
            or simulator.optimizer.param_groups[0]["lr"] == 0
        ):
            obs_gradient_estimators = None
            obs_package = (
                obs_gradient_estimators,
                mean_obs,
                self.obs_loss(mean_obs).to(self.device),
                mean_adf,
                self.adf_loss(mean_adf).to(self.device),
            )
            return (obs_package, vacf_package, energy_force_package)

        #### Simulation phase computations #####

        stacked_radii = stacked_radii[:, mask]
        stacked_radii = stacked_radii.reshape(-1, self.simulator.n_atoms, 3)

        if self.simulator.name == "water":
            # Water subsampling
            (
                stacked_radii,
                local_stacked_radii,
                atomic_indices,
                bond_lens,
                imds,
            ) = self.process_local_neighborhoods(stacked_radii)

        else:
            obs = obs[:, mask].reshape(-1, obs.shape[-1])
            adfs = adfs[:, mask].reshape(-1, adfs.shape[-1])

        obs.requires_grad = True
        adfs.requires_grad = True

        # shuffle the radii, rdfs, and losses
        if self.simulator.shuffle:
            shuffle_idx = torch.randperm(stacked_radii.shape[0])
            stacked_radii = stacked_radii[shuffle_idx]
            if self.simulator.name == "water":
                atomic_indices = atomic_indices[shuffle_idx]
                bond_lens = bond_lens[shuffle_idx]
            else:
                obs = obs[shuffle_idx]
                adfs = adfs[shuffle_idx]

        num_blocks = math.ceil(stacked_radii.shape[0] / self.simulator.minibatch_size)
        obs_gradient_estimators = []
        raw_grads = []
        pe_grads_flattened = []
        if self.simulator.name == "water":
            print(
                f"Computing {self.simulator.training_observable} gradients from {stacked_radii.shape[0]} local environments of {local_stacked_radii.shape[-2]} atoms in minibatches of size {self.simulator.minibatch_size}"
            )
        else:
            print(
                f"Computing {self.simulator.training_observable} gradients from {stacked_radii.shape[0]} structures in minibatches of size {self.simulator.minibatch_size}"
            )
        num_params = len(list(model.parameters()))

        # first compute gradients of potential energy (in batches of size n_replicas)
        for i in tqdm(range(num_blocks)):
            start = self.simulator.minibatch_size * i
            end = self.simulator.minibatch_size * (i + 1)
            with torch.enable_grad():
                radii_in = stacked_radii[start:end]
                if self.simulator.name == "water":
                    atomic_indices_in = atomic_indices[start:end]
                radii_in.requires_grad = True
                energy_force_output = self.simulator.calculator.calculate_energy_force(
                    radii_in,
                    retain_grad=True,
                    output_individual_energies=self.simulator.name == "water",
                )
                if len(energy_force_output) == 2:
                    # global energy
                    energy = energy_force_output[0]
                elif len(energy_force_output) == 3:
                    # individual atomic energies
                    energy = energy_force_output[1].reshape(radii_in.shape[0], -1, 1)
                    # sum atomic energies within local neighborhoods
                    energy = torch.stack(
                        [
                            energy[i, atomic_index].sum()
                            for i, atomic_index in enumerate(atomic_indices_in)
                        ]
                    )
                    energy = energy.reshape(-1, 1)

            # need to do another loop here over batches of local neighborhoods
            num_local_blocks = math.ceil(
                energy.shape[0] / self.simulator.minibatch_size
            )
            for i in range(num_local_blocks):
                start_inner = self.simulator.minibatch_size * i
                end_inner = self.simulator.minibatch_size * (i + 1)
                local_energy = energy[start_inner:end_inner]

                def get_vjp(v):
                    return compute_grad(
                        inputs=list(model.parameters()),
                        output=local_energy,
                        grad_outputs=v,
                        allow_unused=True,
                        create_graph=False,
                    )

                vectorized_vjp = vmap(get_vjp)
                I_N = torch.eye(local_energy.shape[0]).unsqueeze(-1).to(self.device)
                num_samples = local_energy.shape[0]

                grads_vectorized = vectorized_vjp(I_N)
                # flatten the gradients for vectorization
                grads_flattened = torch.stack(
                    [
                        torch.cat(
                            [
                                grads_vectorized[i][j].flatten().detach()
                                for i in range(num_params)
                            ]
                        )
                        for j in range(num_samples)
                    ]
                )

                pe_grads_flattened.append(grads_flattened)

        pe_grads_flattened = torch.cat(pe_grads_flattened)
        # Now compute final gradient estimators in batches of size self.simulator.minibatch_size
        num_blocks = math.ceil(stacked_radii.shape[0] / (self.simulator.minibatch_size))

        for i in tqdm(range(num_blocks)):
            start = self.simulator.minibatch_size * i
            end = self.simulator.minibatch_size * (i + 1)
            if self.simulator.training_observable == "imd":
                obs = imds
            elif self.simulator.training_observable == "bond_length_dev":
                obs = bond_lens

            obs = obs.reshape(pe_grads_flattened.shape[0], -1)
            obs_batch = [obs[start:end]]
            pe_grads_batch = pe_grads_flattened[start:end]
            grad_outputs_obs = [
                2 * (ob.mean(0) - gt_ob).unsqueeze(0)
                for ob, gt_ob in zip(
                    obs_batch,
                    self.gt_obs_local.chunk(len(obs_batch)),
                )
            ]
            final_vjp = [
                self.estimator(obs, pe_grads_batch, grad_output_obs)
                for obs, grad_output_obs in zip(obs_batch, grad_outputs_obs)
            ]
            final_vjp = torch.stack(final_vjp).mean(0)

            raw_grads.append(final_vjp)

        # compute mean across minibatches
        mean_vjps = torch.stack(raw_grads).mean(dim=0)
        # re-assemble flattened gradients into correct shape
        mean_vjps = tuple(
            [
                g.reshape(shape)
                for g, shape in zip(mean_vjps.split(original_numel), original_shapes)
            ]
        )
        obs_gradient_estimators.append(mean_vjps)

        # return final quantities
        obs_package = (
            obs_gradient_estimators,
            mean_obs,
            mean_obs_loss.to(self.device),
            mean_adf,
            mean_adf_loss.to(self.device),
        )

        return obs_package, vacf_package, energy_force_package

    def process_local_neighborhoods(self, stacked_radii):
        """
        Extracts local "shells" of water molecules with which to compute the
        localized Boltzmann estimator, and computes ground truth observables.

        To provide informative samples for optimization, we assign the shells
        to one of 100 bins based on their deviation from the equilibrium O-H bond length.
        Then, we choose a fixed number (5) of shells from each bin.

        Args:
            stacked_radii (torch.Tensor):
        Returns:
            stacked_radii (torch.Tensor):
            local_stacked_radii (torch.Tensor):
            atomic_indices (torch.Tensor):
            bond_lens (torch.Tensor):
            imds (torch.Tensor):
        """

        # extract all local neighborhoods of size n_molecules centered around each atom

        # pick centers of each local neighborhood (always an Oxygen atom)
        center_atoms = 3 * np.random.choice(
            np.arange(int(self.simulator.n_atoms / 3)),
            self.simulator.n_local_neighborhoods,
            replace=False,
        )
        local_neighborhoods = [
            n_closest_molecules(
                stacked_radii,
                center_atom,
                self.simulator.n_closest_molecules,
                self.simulator.cell,
            )
            for center_atom in center_atoms
        ]

        local_stacked_radii = torch.stack(
            [local_neighborhood[0] for local_neighborhood in local_neighborhoods],
            dim=1,
        )
        atomic_indices = torch.stack(
            [local_neighborhood[1] for local_neighborhood in local_neighborhoods],
            dim=1,
        )

        imds = torch.cat(
            [
                self.simulator.min_imd(s.unsqueeze(0).unsqueeze(0))
                for s in local_stacked_radii.reshape(
                    -1, self.simulator.n_atoms_local, 3
                )
            ]
        )
        imds = imds.reshape(
            local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1
        )
        bond_lens = torch.cat(
            [
                self.simulator.bond_length_dev(s.unsqueeze(0).unsqueeze(0))[0]
                for s in local_stacked_radii.reshape(
                    -1, self.simulator.n_atoms_local, 3
                )
            ]
        )
        bond_len_devs = torch.cat(
            [
                self.simulator.bond_length_dev(s.unsqueeze(0).unsqueeze(0))[1]
                for s in local_stacked_radii.reshape(
                    -1, self.simulator.n_atoms_local, 3
                )
            ]
        )
        bond_lens = bond_lens.reshape(
            local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1
        )
        bond_len_devs = bond_len_devs.reshape(
            local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1
        )

        # scheme to subsample local neighborhoods
        bond_len_devs_temp = bond_len_devs.reshape(-1, 1).cpu()
        bond_lens_temp = bond_lens.reshape(-1, bond_lens.shape[-1]).cpu()
        _, bins = np.histogram(bond_len_devs_temp, bins=100)
        bin_assignments = np.digitize(bond_len_devs_temp, bins)
        samples_per_bin = 5 * self.simulator.n_replicas

        full_list = []
        for bin_index in range(1, 101):
            # Filter elements belonging to the current bin
            bin_elements = bond_lens_temp[(bin_assignments == bin_index).squeeze(-1)]

            # Randomly sample elements from the bin, if available
            if len(bin_elements) >= samples_per_bin:
                shuffle_idx = torch.randperm(bin_elements.shape[0])
                sampled_elements = bin_elements[shuffle_idx][:samples_per_bin]
            else:
                # If less than 5 elements, take all available
                sampled_elements = torch.Tensor(bin_elements)

            full_list = full_list + [sampled_elements]
        full_list = torch.cat(full_list).to(self.simulator.device)
        # subsample the relevant frames
        indices = [
            torch.cat(
                [
                    x[0].unsqueeze(0).to(self.simulator.device)
                    for x in torch.where(bond_lens == value)[0:2]
                ]
            )
            for value in full_list
        ]
        local_stacked_radii = torch.stack(
            [local_stacked_radii[idx[0], idx[1]] for idx in indices]
        )
        stacked_radii = torch.stack([stacked_radii[idx[0]] for idx in indices])
        atomic_indices = torch.stack(
            [atomic_indices[idx[0], idx[1]] for idx in indices]
        )
        bond_lens = full_list

        # Save relevant quantities
        np.save(
            os.path.join(
                self.simulator.save_dir,
                f"local_stacked_radii_epoch{self.simulator.epoch}.npy",
            ),
            local_stacked_radii.cpu(),
        )

        np.save(
            os.path.join(
                self.simulator.save_dir,
                f"local_imds_epoch{self.simulator.epoch}.npy",
            ),
            imds.cpu(),
        )
        np.save(
            os.path.join(
                self.simulator.save_dir,
                f"local_bond_len_devs_epoch{self.simulator.epoch}.npy",
            ),
            bond_len_devs.cpu(),
        )

        return stacked_radii, local_stacked_radii, atomic_indices, bond_lens, imds
