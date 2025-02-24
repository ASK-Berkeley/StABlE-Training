import numpy as np
import torch
import logging
import gc
import os
import shutil
import random
import types
from torchmd.observable import DifferentiableVACF
import time
from nequip.train.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from mdsim.common.utils import (
    extract_cycle_epoch,
    build_config,
    setup_logging,
    compare_gradients,
)
from simulator_utils import energy_force_error, save_checkpoint, calculate_final_metrics
from mdsim.observables.md17_22 import find_hr_adf_from_file
from mdsim.observables.water import find_water_rdfs_diffusivity_from_file
from mdsim.models.load_models import load_pretrained_model
from mdsim.common.flags import flags
from boltzmann_estimator import BoltzmannEstimator
from simulator import Simulator

MAX_SIZES = {"md17": "10k", "md22": "100percent", "water": "10k"}

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
    setup_logging()
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()

    config = build_config(args, override_args)
    params = types.SimpleNamespace(**config)

    params.results_dir = os.path.join(params.log_dir, params.results_dir)

    # Set random seeds
    np.random.seed(seed=params.seed)
    torch.manual_seed(params.seed)
    random.seed(params.seed)
    # GPU
    try:
        device = torch.device(torch.cuda.current_device())
    except:
        device = "cpu"

    # set up model
    data_path = config["src"]
    name = config["name"]
    molecule = f"{config['molecule']}" if name == "md17" or name == "md22" else ""
    molecule_for_name = name if name == "water" else molecule
    size = config["size"]
    model_type = config["model"]

    # make directories
    results_dir = (
        os.path.join(
            params.results_dir,
            f"{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}",
        )
        if params.train
        else os.path.join(
            params.results_dir,
            f"{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}",
            "inference",
            params.eval_model,
        )
    )
    os.makedirs(results_dir, exist_ok=True)

    ######## Load appropriate model checkpoint ##############

    print(f"Loading pretrained {model_type} model")
    lmax_string = f"lmax={params.l_max}_" if model_type == "nequip" else ""
    # load the correct checkpoint based on whether we're doing train or val
    load_cycle = None
    load_epoch = None

    print("Pretrained model:", config["eval_model"])
    if (
        params.train or "pre" in config["eval_model"]
    ):  # load energies/forces trained model
        pretrained_model_path = os.path.join(
            config["model_dir"],
            model_type,
            f"{name}-{molecule}_{size}_{lmax_string}{model_type}",
        )

    elif (
        "k" in config["eval_model"] or "percent" in config["eval_model"]
    ):  # load energies/forces model trained on a different dataset size
        new_size = config["eval_model"]
        pretrained_model_path = os.path.join(
            config["model_dir"],
            model_type,
            f"{name}-{molecule}_{new_size}_{lmax_string}{model_type}",
        )

    elif (
        "lmax" in config["eval_model"]
    ):  # load energies/forces model trained with a different lmax
        assert (
            model_type == "nequip"
        ), f"l option is only valid with nequip models, you have selected {model_type}"
        new_lmax_string = config["eval_model"] + "_"
        pretrained_model_path = os.path.join(
            config["model_dir"],
            model_type,
            f"{name}-{molecule}_{size}_{new_lmax_string}{model_type}",
        )

    elif (
        "post" in config["eval_model"]
    ):  # load observable finetuned model at some point in training
        pretrained_model_path = os.path.join(
            params.results_dir,
            f"{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}",
        )
        if (
            "cycle" in config["eval_model"]
        ):  # load observable finetuned model at specified cycle
            load_cycle, load_epoch = extract_cycle_epoch(config["eval_model"])
    else:
        RuntimeError("Invalid eval model choice")

    if model_type == "nequip":
        if "post" in config["eval_model"] and not params.train:
            # get config from pretrained directory
            if load_cycle is not None:
                if load_epoch is not None:
                    cname = f"cycle{load_cycle}_epoch{load_epoch}.pth"
                else:
                    cname = f"end_of_cycle{load_cycle}.pth"
            else:
                cname = "ckpt.pth"
            print(
                f"Loading model weights from {os.path.join(pretrained_model_path, cname)}"
            )
            pre_path = os.path.join(
                config["model_dir"],
                model_type,
                f"{name}-{molecule}_{size}_{lmax_string}{model_type}",
            )
            _, model_config = Trainer.load_model_from_training_session(
                pre_path, model_name="best_model.pth", device=torch.device(device)
            )
            # get model from finetuned directory
            model, _ = Trainer.load_model_from_training_session(
                pretrained_model_path,
                config_dictionary=model_config,
                model_name=cname,
                device=torch.device(device),
            )
        else:
            ckpt_epoch = config["checkpoint_epoch"]
            cname = "best_model.pth" if ckpt_epoch == -1 else f"ckpt{ckpt_epoch}.pth"
            print(
                f"Loading model weights from {os.path.join(pretrained_model_path, cname)}"
            )
            model, model_config = Trainer.load_model_from_training_session(
                pretrained_model_path, model_name=cname, device=torch.device(device)
            )
            shutil.copy(
                os.path.join(pretrained_model_path, cname),
                os.path.join(results_dir, "best_model.pth"),
            )
            shutil.copy(
                os.path.join(pretrained_model_path, "config.yaml"),
                os.path.join(results_dir, "config.yaml"),
            )
            shutil.copy(
                os.path.join(pretrained_model_path, "trainer.pth"),
                os.path.join(results_dir, "trainer.pth"),
            )
        model_path = os.path.join(pretrained_model_path, cname)
        model_config = {"model": model_config}
    else:
        model, model_path, model_config = load_pretrained_model(
            model_type,
            path=pretrained_model_path,
            ckpt_epoch=config["checkpoint_epoch"],
            cycle=load_cycle,
            post_epoch=load_epoch,
            device=torch.device(device),
            train=params.train or "post" not in params.eval_model,
        )
        # copy original model config to results directory
        if params.train:
            shutil.copy(
                os.path.join(pretrained_model_path, "checkpoints", "config.yml"),
                os.path.join(results_dir, "config.yml"),
            )
    # count number of trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"{num_params} trainable parameters in {model_type} model")

    integrator_config = config["integrator_config"]
    timestep = config["timestep"]
    ttime = integrator_config["ttime"]

    ###########Compute ground truth observables###############

    print("Computing ground truth observables from datasets")

    if name == "water":
        # TODO: compute the ground truth imd and bond length dev here instead of hardcoding
        (
            gt_rdf,
            gt_rdf_local,
            gt_diffusivity,
            gt_msd,
            gt_adf,
            oxygen_atoms_mask,
        ) = find_water_rdfs_diffusivity_from_file(
            data_path, MAX_SIZES[name], params, device
        )
        gt_rdf_package = (gt_rdf, gt_rdf_local)

    else:
        gt_rdf, gt_adf = find_hr_adf_from_file(
            data_path, name, molecule, MAX_SIZES[name], params, device
        )
        gt_rdf_package = (gt_rdf,)

    contiguous_path = os.path.join(
        data_path, f"contiguous-{name}", molecule, MAX_SIZES[name], "val/nequip_npz.npz"
    )
    gt_data = np.load(contiguous_path)
    if name == "water":
        gt_vels = torch.FloatTensor(gt_data.f.velocities).to(device)
    else:
        gt_traj = torch.FloatTensor(gt_data.f.R).to(device)
        gt_vels = gt_traj[1:] - gt_traj[:-1]  # finite difference approximation

    gt_vacf = DifferentiableVACF(params, device)(gt_vels)
    if name == "water":
        for _type, _rdf in gt_rdf.items():
            np.save(os.path.join(results_dir, f"gt_{_type}_rdf.npy"), _rdf[0].cpu())
            np.save(
                os.path.join(results_dir, f"gt_{_type}_rdf_local.npy"),
                gt_rdf_local[_type][0].cpu(),
            )
    else:
        np.save(os.path.join(results_dir, "gt_rdf.npy"), gt_rdf.cpu())
    np.save(os.path.join(results_dir, "gt_adf.npy"), gt_adf.cpu())
    np.save(os.path.join(results_dir, "gt_vacf.npy"), gt_vacf.cpu())
    if name == "water":
        np.save(os.path.join(results_dir, "gt_diffusivity.npy"), gt_diffusivity.cpu())
        np.save(os.path.join(results_dir, "gt_msd.npy"), gt_msd.cpu())

    ########## Training setup ############

    min_lr = params.lr / (
        5**params.max_times_reduce_lr
    )  # LR reduction factor is 0.2 each time
    losses = []
    obs_losses = []
    adf_losses = []
    diffusion_losses = []
    all_vacfs_per_replica = []
    vacf_losses = []
    mean_instabilities = []
    bond_length_devs = []
    rdf_maes = []
    energy_rmses = []
    force_rmses = []
    energy_maes = []
    force_maes = []
    grad_times = []
    sim_times = []
    grad_norms = []
    lrs = []
    resets = []
    writer = SummaryWriter(log_dir=results_dir)
    changed_lr = False
    cycle = 0
    learning_epochs_in_cycle = 0

    ######### Begin StABlE Training Loop #############

    for epoch in range(params.n_epochs):

        best = False
        grad_cosine_similarity = 0
        ratios = 0
        print(f"Epoch {epoch+1}")

        if epoch == 0:
            # initialize MD simulator
            simulator = Simulator(
                config, params, model, model_path, model_config, gt_rdf
            )
            # initialize Boltzmann_estimator
            boltzmann_estimator = BoltzmannEstimator(
                gt_rdf_package,
                simulator.mean_bond_lens,
                gt_vacf,
                gt_adf,
                params,
                device,
            )
            # initialize outer loop optimizer/scheduler
            if params.optimizer == "Adam":
                simulator.optimizer = torch.optim.Adam(
                    list(simulator.model.parameters()),
                    0 if params.only_learn_if_unstable_threshold_reached else params.lr,
                )
            elif params.optimizer == "SGD":
                simulator.optimizer = torch.optim.SGD(
                    list(simulator.model.parameters()),
                    0 if params.only_learn_if_unstable_threshold_reached else params.lr,
                )
            else:
                raise RuntimeError("Optimizer must be either Adam or SGD")
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                simulator.optimizer, mode="min", factor=0.2, patience=10
            )
            print(f"Initialize {simulator.n_replicas} random ICs in parallel")

        simulator.optimizer.zero_grad()
        simulator.epoch = epoch

        # set starting states based on instability metric
        num_unstable_replicas = simulator.set_starting_states()

        # start learning with focus on instability
        if simulator.all_unstable and not changed_lr:
            simulator.optimizer.param_groups[0]["lr"] = params.lr
            # reset scheduler
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                simulator.optimizer, mode="min", factor=0.2, patience=10
            )
            changed_lr = True

        start = time.time()
        # Run MD simulation
        print("Collect MD Simulation Data")
        equilibriated_simulator = simulator.solve()
        # If instability threshold reached, estimate gradients via Boltzmann estimator
        # Otherwise, the gradients are None
        (
            obs_package,
            vacf_package,
            energy_force_grad_batches,
        ) = boltzmann_estimator.compute(equilibriated_simulator)

        # unpack results
        obs_grad_batches, mean_obs, obs_loss, mean_adf, adf_loss = obs_package
        vacfs_per_replica, vacf_loss = vacf_package
        if not params.train:
            all_vacfs_per_replica.append(vacfs_per_replica)

        # Compute observable loss
        outer_loss = params.obs_loss_weight * obs_loss
        print(f"Observable Loss ={outer_loss.item()}")

        # make lengths match for iteration
        if energy_force_grad_batches is None:
            energy_force_grad_batches = obs_grad_batches

        # Learning phase gradient update
        if obs_grad_batches is not None:
            grad_cosine_similarity = []
            ratios = []
            for obs_grads, energy_force_grads in zip(
                obs_grad_batches, energy_force_grad_batches
            ):  # loop through gradients - should only be a single step
                # since we don't allow off-policy gradients
                simulator.optimizer.zero_grad()

                # modify gradients according to loss weights
                ef_grads = tuple(
                    [
                        params.energy_force_loss_weight * ef_grad
                        for ef_grad in energy_force_grads
                    ]
                )
                cosine_similarity, ratio = compare_gradients(obs_grads, ef_grads)
                grad_cosine_similarity.append(
                    cosine_similarity
                )  # compute gradient similarities
                ratios.append(ratio)

                # Loop through each group of parameters and set gradients
                for param, obs_grad, ef_grad in zip(
                    model.parameters(), obs_grads, ef_grads
                ):
                    param.grad = obs_grad + ef_grad

                if params.gradient_clipping:  # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params.grad_clip_norm_threshold
                    )

                simulator.optimizer.step()
            # If we are focusing on accuracy, step based on observable loss. If we are focusing on stability, step based on number of unstable replicas
            simulator.scheduler.step(
                num_unstable_replicas if simulator.all_unstable else outer_loss
            )
            grad_cosine_similarity = sum(grad_cosine_similarity) / len(
                grad_cosine_similarity
            )
            ratios = sum(ratios) / len(ratios)

        if (
            simulator.all_unstable
            and params.train
            and (
                simulator.optimizer.param_groups[0]["lr"] < min_lr
                or num_unstable_replicas <= params.min_frac_unstable_threshold
            )
        ):
            # Switch from learning to simulation phase
            print(f"Back to data collection")
            simulator.all_unstable = False
            simulator.first_simulation = True
            changed_lr = False
            cycle = cycle + 1
            learning_epochs_in_cycle = 0
            # save checkpoint at the end of learning cycle
            if params.train:
                simulator.curr_model_path = save_checkpoint(
                    simulator, name_=f"end_of_cycle{cycle}"
                )
            # reinitialize optimizer and scheduler with LR = 0
            if params.optimizer == "Adam":
                simulator.optimizer = torch.optim.Adam(
                    list(simulator.model.parameters()),
                    0 if params.only_learn_if_unstable_threshold_reached else params.lr,
                )
            elif params.optimizer == "SGD":
                simulator.optimizer = torch.optim.SGD(
                    list(simulator.model.parameters()),
                    0 if params.only_learn_if_unstable_threshold_reached else params.lr,
                )
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                simulator.optimizer, mode="min", factor=0.2, patience=10
            )

        end = time.time()
        sim_time = end - start

        # checkpointing
        if params.train and simulator.mode == "learning":
            simulator.curr_model_path = save_checkpoint(
                simulator, name_=f"cycle{cycle+1}_epoch{learning_epochs_in_cycle+1}"
            )
            learning_epochs_in_cycle += 1

        torch.cuda.empty_cache()
        gc.collect()
        max_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
                if norm > max_norm:
                    max_norm = norm

        # log stats
        losses.append(outer_loss.item())
        obs_losses.append(obs_loss.item())
        adf_losses.append(adf_loss.item())
        vacf_losses.append(vacf_loss.item())
        mean_instabilities.append(equilibriated_simulator.mean_instability)
        if simulator.pbc:
            bond_length_devs.append(equilibriated_simulator.mean_bond_length_dev)
            rdf_maes.append(equilibriated_simulator.mean_rdf_mae)
        resets.append(num_unstable_replicas)
        lrs.append(simulator.optimizer.param_groups[0]["lr"])
        # energy/force error
        if epoch == 0 or (
            simulator.optimizer.param_groups[0]["lr"] > 0 and params.train
        ):  # don't compute it unless we are in the learning phase
            energy_rmse, force_rmse, energy_mae, force_mae = energy_force_error(
                simulator
            )
            energy_rmses.append(energy_rmse)
            force_rmses.append(force_rmse)
            energy_maes.append(energy_mae)
            force_maes.append(force_mae)
        sim_times.append(sim_time)
        if isinstance(grad_norms, torch.Tensor):
            grad_norms.append(max_norm.item())
        else:
            grad_norms.append(max_norm)
        if simulator.all_unstable and not params.train:
            # reached instability point, can stop
            break
        writer.add_scalar("Loss", losses[-1], global_step=epoch + 1)
        writer.add_scalar("Observable Loss", obs_losses[-1], global_step=epoch + 1)
        writer.add_scalar("ADF Loss", adf_losses[-1], global_step=epoch + 1)
        writer.add_scalar("VACF Loss", vacf_losses[-1], global_step=epoch + 1)
        if params.stability_criterion == "imd":
            writer.add_scalar(
                "Min Intermolecular Distances",
                mean_instabilities[-1],
                global_step=epoch + 1,
            )
        elif params.stability_criterion == "bond_length_deviation":
            writer.add_scalar(
                "Max Bond Length Deviation",
                mean_instabilities[-1],
                global_step=epoch + 1,
            )

        if simulator.pbc:
            writer.add_scalar(
                "Max Bond Length Deviation", bond_length_devs[-1], global_step=epoch + 1
            )
            writer.add_scalar("RDF MAE", rdf_maes[-1], global_step=epoch + 1)

        writer.add_scalar(
            "Fraction of Unstable Replicas", resets[-1], global_step=epoch + 1
        )
        writer.add_scalar("Learning Rate", lrs[-1], global_step=epoch + 1)
        writer.add_scalar("Energy RMSE", energy_rmses[-1], global_step=epoch + 1)
        writer.add_scalar("Force RMSE", force_rmses[-1], global_step=epoch + 1)
        writer.add_scalar("Energy MAE", energy_maes[-1], global_step=epoch + 1)
        writer.add_scalar("Force MAE", force_maes[-1], global_step=epoch + 1)
        writer.add_scalar("Simulation Time", sim_times[-1], global_step=epoch + 1)
        writer.add_scalar("Gradient Norm", grad_norms[-1], global_step=epoch + 1)
        writer.add_scalar(
            "Gradient Cosine Similarity (Observable vs Energy-Force)",
            grad_cosine_similarity,
            global_step=epoch + 1,
        )
        writer.add_scalar(
            "Gradient Ratios (Observable vs Energy-Force)",
            ratios,
            global_step=epoch + 1,
        )

        # log hyperparams and final metrics at inference time (do it every 5 epochs in case we time-out before the end)
        if not params.train:
            if epoch % 5 == 0 and epoch > 175:  # to save time
                if name == "md17" or name == "md22":
                    hparams_logging = calculate_final_metrics(
                        simulator,
                        params,
                        device,
                        results_dir,
                        energy_maes,
                        force_maes,
                        gt_rdf,
                        gt_adf,
                        gt_vacf,
                        all_vacfs_per_replica=all_vacfs_per_replica,
                    )
                elif name == "water":
                    hparams_logging = calculate_final_metrics(
                        simulator,
                        params,
                        device,
                        results_dir,
                        energy_maes,
                        force_maes,
                        gt_rdf,
                        gt_adf,
                        gt_diffusivity=gt_diffusivity,
                        oxygen_atoms_mask=oxygen_atoms_mask,
                    )
                for i in hparams_logging:
                    writer.file_writer.add_summary(i)

    ######### End StABlE Training Loop #############

    # save metrics at the very end
    if not params.train:
        if name == "md17" or name == "md22":
            hparams_logging = calculate_final_metrics(
                simulator,
                params,
                device,
                results_dir,
                energy_maes,
                force_maes,
                gt_rdf,
                gt_adf,
                gt_vacf,
                all_vacfs_per_replica=all_vacfs_per_replica,
            )
        elif name == "water":
            hparams_logging = calculate_final_metrics(
                simulator,
                params,
                device,
                results_dir,
                energy_maes,
                force_maes,
                gt_rdf,
                gt_adf,
                gt_diffusivity=gt_diffusivity,
                oxygen_atoms_mask=oxygen_atoms_mask,
            )
        for i in hparams_logging:
            writer.file_writer.add_summary(i)

    writer.close()
    if not params.train:
        # close simulation file
        if hasattr(equilibriated_simulator, "t"):
            equilibriated_simulator.t.close()
    print("Done!")
