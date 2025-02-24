import shutil
import os
import torch
import gsd
import numpy as np
from ase import units
from tqdm import tqdm
from torch_cluster import radius_graph
from collections import OrderedDict
from nequip.utils import atomic_write
from nequip.utils.torch_geometric import Batch
from nequip.data import AtomicData
from mdsim.md.ase_utils import OCPCalculator
from mdsim.common.utils import process_gradient
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.observables.md17_22 import get_hr
from mdsim.observables.water import get_water_rdfs
from mdsim.observables.common import get_smoothed_diffusivity
import json
from torchmd.observable import DifferentiableADF
from torch.utils.tensorboard.summary import hparams


def calculate_final_metrics(
    simulator,
    params,
    device,
    results_dir,
    energy_maes,
    force_maes,
    gt_rdf,
    gt_adf=None,
    gt_vacf=None,
    gt_diffusivity=None,
    oxygen_atoms_mask=None,
    all_vacfs_per_replica=None,
):
    """
    Compute and save all final metrics during inference with a StABlE-trained model on held-out initial conditions.
    These include energy/force losses, stability across replicas, and accuracy of various observables computed only
    over the stable portion of each replica trajectory. The per-replica observables are also saved. The quantities computed
    by this function are used to produce the final figures.

    This function is only run at inference time.
    """
    np.save(
        os.path.join(results_dir, f"replicas_stable_time.npy"),
        simulator.stable_time.cpu().numpy(),
    )
    full_traj = torch.stack(simulator.all_radii)
    np.save(os.path.join(results_dir, "full_traj.npy"), full_traj)
    hyperparameters = {
        "lr": params.lr,
        "ef_loss_weight": params.energy_force_loss_weight,
    }
    steps_per_epoch = int(simulator.nsteps / simulator.n_dump)
    stable_steps = simulator.stable_time * steps_per_epoch
    stable_trajs = [
        full_traj[: int(upper_step_limit), i]
        for i, upper_step_limit in enumerate(stable_steps)
    ]
    xlim = params.max_rdf_dist
    n_bins = int(xlim / params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1)  # for computing h(r)
    if params.name == "md17" or params.name == "md22":
        final_rdfs = torch.stack(
            [torch.Tensor(get_hr(traj, bins)).to(device) for traj in stable_trajs]
        )
        gt_rdf = (
            final_rdfs[0].sum() * gt_rdf / gt_rdf.sum()
        )  # normalize to be on the same scale
        final_rdf_maes = xlim * torch.abs(gt_rdf.unsqueeze(0) - final_rdfs).mean(-1)
        adf = DifferentiableADF(
            simulator.n_atoms, simulator.bonds, simulator.cell, params, device
        )
        final_adfs = torch.stack([adf(traj[::5].to(device)) for traj in stable_trajs])
        final_adf_maes = torch.abs(gt_adf.unsqueeze(0) - final_adfs).mean(-1)
        count_per_replica = (
            torch.stack(all_vacfs_per_replica).sum(0)[:, 0].unsqueeze(-1) + 1e-8
        )
        final_vacfs = (
            torch.stack(all_vacfs_per_replica).sum(0) / count_per_replica
        ).to(device)
        final_vacf_maes = torch.abs(gt_vacf.unsqueeze(0) - final_vacfs).mean(-1)
        final_metrics = {
            "Energy MAE (kcal/mol)": energy_maes[-1],
            "Force MAE (kcal/mol-A)": force_maes[-1],
            "Mean Stability (ps)": simulator.stable_time.mean().item(),
            "Std Dev Stability (ps)": simulator.stable_time.std().item(),
            "Mean RDF MAE": final_rdf_maes.mean().item(),
            "Mean ADF MAE": final_adf_maes.mean().item(),
            "Mean VACF MAE": final_vacf_maes.mean().item(),
            "Std Dev RDF MAE": final_rdf_maes.std().item(),
            "Std Dev ADF MAE": final_adf_maes.std().item(),
            "Std Dev VACF MAE": final_vacf_maes.std().item(),
        }
        # save rdfs, adfs, and vacfs at the end of the trajectory
        np.save(os.path.join(results_dir, "final_rdfs.npy"), final_rdfs.cpu())
        np.save(os.path.join(results_dir, "final_adfs.npy"), final_adfs.cpu())
        np.save(os.path.join(results_dir, "final_vacfs.npy"), final_vacfs.cpu())
        np.save(os.path.join(results_dir, "final_rdf_maes.npy"), final_rdf_maes.cpu())
        np.save(os.path.join(results_dir, "final_adf_maes.npy"), final_adf_maes.cpu())
        np.save(os.path.join(results_dir, "final_vacf_maes.npy"), final_vacf_maes.cpu())

    elif params.name == "water":
        final_rdfs = [
            get_water_rdfs(
                traj,
                simulator.rdf_mae.ptypes,
                simulator.rdf_mae.lattices,
                simulator.rdf_mae.bins,
                device,
            )
            for traj in stable_trajs
        ]
        final_rdfs_by_key = {
            k: torch.stack([final_rdf[k] for final_rdf in final_rdfs])
            for k in gt_rdf.keys()
        }
        final_rdf_maes = {
            k: xlim * torch.abs(gt_rdf[k] - final_rdfs_by_key[k]).mean(-1).squeeze(-1)
            for k in gt_rdf.keys()
        }
        # Recording frequency is 1 ps for diffusion coefficient
        all_diffusivities = [
            get_smoothed_diffusivity(
                traj[:: int(1000 / params.n_dump), oxygen_atoms_mask]
            )[0]
            for traj in stable_trajs
        ]
        all_msds = [
            get_smoothed_diffusivity(
                traj[:: int(1000 / params.n_dump), oxygen_atoms_mask]
            )[1]
            for traj in stable_trajs
        ]
        last_diffusivities = torch.cat(
            [
                diff[-1].unsqueeze(-1) if len(diff) > 0 else torch.Tensor([0.0])
                for diff in all_diffusivities
            ]
        )
        diffusivity_maes = (
            10 * (gt_diffusivity[-1].to(device) - last_diffusivities.to(device)).abs()
        )

        # save full diffusivity trajectory
        all_diffusivities = [diff.cpu() for diff in all_diffusivities]
        np.save(
            os.path.join(results_dir, "all_diffusivities.npy"),
            np.array(all_diffusivities, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(results_dir, "all_msds.npy"),
            np.array(all_msds, dtype=object),
            allow_pickle=True,
        )

        final_metrics = {
            "Energy MAE": energy_maes[-1],
            "Force MAE": force_maes[-1],
            "Mean Stability (ps)": simulator.stable_time.median().item(),
            "Std Dev Stability (ps)": simulator.stable_time.std().item(),
            "Mean OO RDF MAE": final_rdf_maes["OO"].mean().item(),
            "Mean HO RDF MAE": final_rdf_maes["HO"].mean().item(),
            "Mean HH RDF MAE": final_rdf_maes["HH"].mean().item(),
            "Std Dev OO RDF MAE": final_rdf_maes["OO"].std().item(),
            "Std Dev HO RDF MAE": final_rdf_maes["HO"].std().item(),
            "Std Dev HH RDF MAE": final_rdf_maes["HH"].std().item(),
            "Mean Diffusivity MAE (10^-9 m^2/s)": diffusivity_maes.mean().item(),
            "Std Dev Diffusivity MAE (10^-9 m^2/s)": diffusivity_maes.std().item(),
        }
        # save rdf, adf, and diffusivity at the end of the traj
        for key, final_rdfs in final_rdfs_by_key.items():
            np.save(
                os.path.join(results_dir, f"final_{key}_rdfs.npy"),
                final_rdfs.squeeze(1).cpu(),
            )
            np.save(
                os.path.join(results_dir, f"final_{key}_rdf_maes.npy"),
                final_rdf_maes[key].cpu(),
            )

        np.save(
            os.path.join(results_dir, "final_diffusivities.npy"),
            last_diffusivities.cpu().detach().numpy(),
        )

    # save final metrics to JSON
    with open(os.path.join(results_dir, "final_metrics.json"), "w") as fp:
        json.dump(final_metrics, fp, indent=4, separators=(",", ": "))
    return hparams(hyperparameters, final_metrics)


def energy_force_gradient(simulator):
    """
    Computes a single gradient (averaged across the training dataset batches) of the original
    energy and forces (QM) training objective.

    Returns a tuple of gradients with the same shape as simulator.model.parameters()
    """
    # store original shapes of model parameters
    original_numel = [param.data.numel() for param in simulator.model.parameters()]
    original_shapes = [param.data.shape for param in simulator.model.parameters()]
    print(
        f"Computing gradients of bottom-up (energy-force) objective on {simulator.train_dataset.__len__()} samples"
    )
    gradients = []
    losses = []
    if simulator.model_type == "nequip":
        with torch.enable_grad():
            for data in tqdm(simulator.train_dataloader):
                # Do any target rescaling
                data = data.to(simulator.device)
                data = Batch(
                    batch=data.batch,
                    ptr=data.ptr,
                    pos=data.pos,
                    cell=data.cell,
                    atomic_numbers=data.atomic_numbers,
                    force=data.force,
                    y=data.y,
                )
                data = AtomicData.to_AtomicDataDict(data)
                actual_batch_size = int(data["pos"].shape[0] / simulator.n_atoms)
                data["cell"] = (
                    data["cell"][0].unsqueeze(0).repeat(actual_batch_size, 1, 1)
                )
                data["pbc"] = (
                    simulator.atoms_batch["pbc"][0]
                    .unsqueeze(0)
                    .repeat(actual_batch_size, 1)
                )
                data["atom_types"] = simulator.atoms_batch["atom_types"][
                    0 : simulator.n_atoms
                ].repeat(actual_batch_size, 1)

                data_unscaled = data
                for layer in simulator.rescale_layers:
                    # normalizes the targets
                    data_unscaled = layer.unscale(data_unscaled)
                # Run model
                data_unscaled["edge_index"] = radius_graph(
                    data_unscaled["pos"].reshape(-1, 3),
                    r=simulator.model_config["model"][simulator.r_max_key],
                    batch=data_unscaled["batch"],
                    max_num_neighbors=32,
                )
                data_unscaled["edge_cell_shift"] = torch.zeros(
                    (data_unscaled["edge_index"].shape[1], 3)
                ).to(simulator.device)
                out = simulator.model(data_unscaled)
                data_unscaled["forces"] = data_unscaled["force"]
                data_unscaled["total_energy"] = data_unscaled["y"].unsqueeze(-1)
                loss, _ = simulator.nequip_loss(pred=out, ref=data_unscaled)
                grads = torch.autograd.grad(
                    loss, simulator.model.parameters(), allow_unused=True
                )
                gradients.append(
                    process_gradient(
                        simulator.model.parameters(), grads, simulator.device
                    )
                )
                losses.append(loss.detach())
    else:
        with torch.enable_grad():
            for batch in tqdm(simulator.train_dataloader):
                with torch.cuda.amp.autocast(
                    enabled=simulator.trainer.scaler is not None
                ):
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(simulator.device)
                    out = simulator.trainer._forward(batch)
                    loss = simulator.trainer._compute_loss(out, [batch])
                    loss = (
                        simulator.trainer.scaler.scale(loss)
                        if simulator.trainer.scaler
                        else loss
                    )
                    grads = torch.autograd.grad(
                        loss, simulator.model.parameters(), allow_unused=True
                    )
                    gradients.append(
                        process_gradient(
                            simulator.model.parameters(), grads, simulator.device
                        )
                    )
                    losses.append(loss.detach())
    grads_flattened = torch.stack(
        [torch.cat([param.flatten().detach() for param in grad]) for grad in gradients]
    )
    mean_grads = grads_flattened.mean(0)
    final_grads = tuple(
        [
            g.reshape(shape)
            for g, shape in zip(mean_grads.split(original_numel), original_shapes)
        ]
    )
    return final_grads


def energy_force_error(simulator):
    """
    Compute the energy/force errors of the simulator's MLFF model on a test set of structures
    Returns a dictionary containing the error metrics.
    """
    if simulator.model_type == "nequip":
        data_config = f"configs/stable_training/{simulator.name}/nequip_data_cfg/{simulator.molecule}.yml"
        # call nequip evaluation script
        os.system(
            f"nequip-evaluate --train-dir {os.path.dirname(simulator.curr_model_path)} \
                    --model {simulator.curr_model_path} --dataset-config {data_config} \
                        --log {os.path.dirname(simulator.curr_model_path)}/test_metric.log --batch-size 4"
        )
        with open(
            f"{os.path.dirname(simulator.curr_model_path)}/test_metric.log", "r"
        ) as f:
            test_log = f.read().splitlines()
            for i, line in enumerate(test_log):
                if "Final result" in line:
                    test_log = test_log[(i + 1) :]
                    break
            test_metrics = {}
            for line in test_log:
                k, v = line.split("=")
                k = k.strip()
                v = float(v.strip())
                test_metrics[k] = v
        return (
            test_metrics["e_mae"],
            test_metrics["f_rmse"],
            test_metrics["e_mae"],
            test_metrics["f_mae"],
        )
    # non-Nequip models use OCP calculator
    else:
        simulator.model_config["model"]["name"] = simulator.model_type
        calculator = OCPCalculator(
            config_yml=simulator.model_config,
            checkpoint=simulator.curr_model_path,
            test_data_src=simulator.DATAPATH_TEST,
            energy_units_to_eV=1.0,
        )
        print(f"Computing bottom-up (energy-force) error on test set")
        test_metrics = calculator.trainer.validate("test", max_points=1000)
        test_metrics = {k: v["metric"] for k, v in test_metrics.items()}
        return (
            test_metrics["energy_rmse"],
            test_metrics["forces_rmse"],
            test_metrics["energy_mae"],
            test_metrics["forces_mae"],
        )


def save_checkpoint(simulator, best=False, name_=None):
    """
    Saves a checkpoint of the current state of the simulator's MLFF model
    """
    if simulator.model_type == "nequip":
        if name_ is not None:
            name = f"{name_}.pth"
        else:
            name = "best_ckpt.pth" if best else "ckpt.pth"
        checkpoint_path = os.path.join(simulator.save_dir, name)
        with atomic_write(checkpoint_path, blocking=True, binary=True) as write_to:
            torch.save(simulator.model.state_dict(), write_to)
    else:
        if name_ is not None:
            name = f"{name_}.pt"
        else:
            name = "best_ckpt.pt" if best else "ckpt.pt"
        checkpoint_path = os.path.join(simulator.save_dir, name)
        new_state_dict = OrderedDict(
            ("module." + k if "module" not in k else k, v)
            for k, v in simulator.model.state_dict().items()
        )
        torch.save(
            {
                "epoch": simulator.epoch,
                "step": simulator.epoch,
                "state_dict": new_state_dict,
                "normalizers": {
                    key: value.state_dict()
                    for key, value in simulator.trainer.normalizers.items()
                },
                "config": simulator.model_config,
                "ema": (
                    simulator.trainer.ema.state_dict()
                    if simulator.trainer.ema
                    else None
                ),
                "amp": (
                    simulator.trainer.scaler.state_dict()
                    if simulator.trainer.scaler
                    else None
                ),
            },
            checkpoint_path,
        )
    # also save in 'ckpt.pt'
    shutil.copyfile(
        checkpoint_path,
        os.path.join(
            simulator.save_dir,
            "ckpt.pth" if simulator.model_type == "nequip" else "ckpt.pt",
        ),
    )
    return checkpoint_path


def create_frame(
    radii, velocities, cell, bonds, pbc, diameter_viz, n_atoms, dt, name, frame
):
    """
    Creates a gsd.hoomd.Frame() object from an instantaneous MD snapshot.
    Used to produce animations of the simulations.
    """
    # Particle positions, velocities, diameter
    radii = radii[0]
    if pbc:
        # wrap for visualization purposes (last subtraction is for cell alignment in Ovito)
        # assumes cubic cell
        radii = ((radii / torch.diag(cell)) % 1) * torch.diag(cell) - torch.diag(
            cell
        ) / 2
    partpos = detach_numpy(radii).tolist()
    velocities = detach_numpy(velocities[0]).tolist()
    diameter = 10 * diameter_viz * np.ones((n_atoms,))
    diameter = diameter.tolist()
    # Now make gsd file
    s = gsd.hoomd.Frame()
    s.configuration.step = frame
    s.particles.N = n_atoms
    s.particles.position = partpos
    s.particles.velocity = velocities
    s.particles.diameter = diameter
    s.configuration.box = [
        cell[0][0].cpu(),
        cell[1][1].cpu(),
        cell[2][2].cpu(),
        0,
        0,
        0,
    ]
    s.configuration.step = dt

    if name != "lips":  # don't show bonds if lips
        s.bonds.N = bonds.shape[0]
        s.bonds.group = detach_numpy(bonds)

    return s


def thermo_log(
    radii,
    velocities,
    pe,
    masses,
    n_atoms,
    stability_criterion,
    bond_length_dev,
    rdf_mae,
    pbc,
):
    """
    Given an instantaneous snapshot of an MD trajectory, compute instantaneous
    quantities like temperature, energy, momentum, stability metrics, etc. Mainly for sanity checking.
    """
    # Log energies and instabilities
    p_dof = 3 * n_atoms
    ke = 1 / 2 * (masses * torch.square(velocities)).sum(axis=(1, 2)).unsqueeze(-1)
    temp = (2 * ke / p_dof).mean() / units.kB
    instability = stability_criterion(radii.unsqueeze(0))
    if isinstance(instability, tuple):
        instability = instability[-1]
    results_dict = {
        "Temperature": temp.item(),
        "Potential Energy": pe.mean().item(),
        "Total Energy": (ke + pe).mean().item(),
        "Momentum Magnitude": torch.norm(
            torch.sum(masses * velocities, axis=-2)
        ).item(),
        "Max Bond Length Deviation": (
            bond_length_dev(radii.unsqueeze(0))[1].mean().item()
            if pbc
            else instability.mean().item()
        ),
    }
    if pbc:
        results_dict["Minimum Intermolecular Distance"] = instability.mean().item()
        results_dict["RDF MAE"] = rdf_mae(radii.unsqueeze(0))[-1].mean().item()
    return results_dict
