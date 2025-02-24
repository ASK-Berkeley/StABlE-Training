"""
Adapted from https://github.com/kyonofx/MDsim/blob/main/main.py
"""

import copy
import logging
import os
import yaml
import time
import torch

import tempfile


import submitit

from mdsim.common import distutils
from mdsim.common.flags import flags
from mdsim.common.registry import registry
from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg,
)

sizes_dict = dict(
    double_walled_nanotube=5032,
    stachyose=27272,
    ac_Ala3_NHMe=85109,
    AT_AT_CG_CG=10153,
    AT_AT=20001,
    buckyball_catcher=6102,
    DHA=69753,
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        setup_logging()
        self.config = copy.deepcopy(config)

        if config["distributed"]:
            distutils.setup(config)

        try:
            setup_imports()

            # compose dataset configs.
            train_data_cfg = config["dataset"]
            train_data_cfg = compose_data_cfg(train_data_cfg)
            config["dataset"] = [
                train_data_cfg,
                {"src": os.path.join(os.path.dirname(train_data_cfg["src"]), "val")},
            ]

            self.config = copy.deepcopy(config)

            # initialize trainer.
            self.trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                timestamp_id=config.get("timestamp_id", None),
                run_dir=config.get("run_dir", None),
                is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 100),
                seed=config.get("seed", 0),
                logger=config.get("logger", "wandb"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                no_energy=config.get("no_energy", False),
            )

            # save config.
            with open(
                os.path.join(
                    self.trainer.config["cmd"]["checkpoint_dir"], "config.yml"
                ),
                "w",
            ) as yf:
                yaml.dump(self.config, yf, default_flow_style=False)

            self.task = registry.get_task_class(config["mode"])(self.config)
            self.task.setup(self.trainer)
            start_time = time.time()
            self.task.run()
            distutils.synchronize()
            if distutils.is_master():
                logging.info(f"Total time taken: {time.time() - start_time}")
        finally:
            if config["distributed"]:
                distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


def modify_config(molecule, original_path, l_max=None, size=None):
    # Function that modifies nequip config to include command line specified l_max, size, and
    # amount of data.
    if molecule == None or molecule not in sizes_dict:
        raise Exception("You need to specify a [correct] molecule for nequip!!")
    n_points = sizes_dict[molecule]
    size_factor_dict = {
        "100percent": 1,
        "50percent": 0.5,
        "10percent": 0.1,
        "5percent": 0.05,
        "25percent": 0.25,
    }
    factor = size_factor_dict[size]

    with open(original_path, "r") as f:
        config = yaml.safe_load(f)

    # Overwrite the config parameters if provided
    if l_max is not None:
        config["l_max"] = l_max
        run_name = config.get("run_name", "")
        config["run_name"] = run_name.replace("-lmax-", f"lmax={l_max}_")
    else:
        raise Exception("Specify l_max!!")
    if size is not None:
        run_name = config.get("run_name", "")
        config["run_name"] = run_name.replace("_size_", f"_{size}_")

        dataset_file_name = config.get("dataset_file_name", "")
        config["dataset_file_name"] = dataset_file_name.replace("/size/", f"/{size}/")

        validation_dataset_file_name = config.get("validation_dataset_file_name", "")
        config["validation_dataset_file_name"] = validation_dataset_file_name.replace(
            "/size/", f"/{size}/"
        )

        # nequip takes in the number of data points used, not the percentage, so we need to convert from percentages to  num points
        train_size = int(n_points * 0.7 * factor)
        val_size = int(n_points * 0.1 * factor)

        config["n_train"] = train_size
        config["n_val"] = val_size
    else:
        raise Exception("Specify size!!")

    # Set the directory where you want to create the temporary file that the training run will use
    temp_dir = os.path.expanduser(
        "~/MDsim/temp_configs"
    )  # or use '.' for the current working directory
    _, temp_filename = tempfile.mkstemp(
        dir=temp_dir, prefix=f"{molecule}_l={l_max}_{size}_"
    )

    with open(temp_filename, "w") as f:
        yaml.safe_dump(config, f)

    print(
        "Temporary file created at:", temp_filename
    )  # Print the temporary file location
    return temp_filename


if __name__ == "__main__":
    setup_logging()
    parser = flags.get_parser()
    torch.set_num_threads(10)

    parser.add_argument(
        "--l_max", type=int, help="The seed to overwrite in the config file"
    )
    # parser.add_argument('--size', help='The size to replace "all" in run_name in the config file')
    # parser.add_argument('--molecule', help='The molecule you are training on')
    args, override_args = parser.parse_known_args()

    if args.nequip:
        os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.local/bin")
        os.system(f"nequip-train {args.config_yml}")
    else:
        config = build_config(args, override_args)
        if args.submit:  # Run on cluster
            slurm_add_params = config.get("slurm", None)  # additional slurm arguments
            if args.sweep_yml:  # Run grid search
                configs = create_grid(config, args.sweep_yml)
            else:
                configs = [config]

            logging.info(f"Submitting {len(configs)} jobs")
            executor = submitit.AutoExecutor(
                folder=args.logdir / "%j", slurm_max_num_timeout=3
            )
            executor.update_parameters(
                name=args.identifier,
                mem_gb=args.slurm_mem,
                timeout_min=args.slurm_timeout * 60,
                slurm_partition=args.slurm_partition,
                gpus_per_node=args.num_gpus,
                cpus_per_task=(config["optim"]["num_workers"] + 1),
                tasks_per_node=(args.num_gpus if args.distributed else 1),
                nodes=args.num_nodes,
                slurm_additional_parameters=slurm_add_params,
            )
            for config in configs:
                config["slurm"] = copy.deepcopy(executor.parameters)
                config["slurm"]["folder"] = str(executor.folder)
            jobs = executor.map_array(Runner(), configs)
            logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
            log_file = save_experiment_log(args, jobs, configs)
            logging.info(f"Experiment log saved to: {log_file}")

        else:  # Run locally
            Runner()(config)
