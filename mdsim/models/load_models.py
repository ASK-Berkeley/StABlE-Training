import torch
import os
import yaml
from mdsim.models.schnet import SchNetWrap

# from mdsim.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from mdsim.models.forcenet import ForceNet
from mdsim.models.gemnet.gemnet import GemNetT
from mdsim.common.registry import registry


def load_pretrained_model(
    model_type,
    path=None,
    ckpt_epoch=-1,
    cycle=None,
    post_epoch=None,
    device="cpu",
    train=True,
):
    if train:
        cname = (
            "best_checkpoint.pt" if ckpt_epoch == -1 else f"checkpoint{ckpt_epoch}.pt"
        )
        ckpt_and_config_path = os.path.join(path, "checkpoints", cname)
    else:  # load from specified cycle and/or epoch if needed
        if cycle is not None:
            if post_epoch is not None:
                ckpt_and_config_path = os.path.join(
                    path, f"cycle{cycle}_epoch{post_epoch}.pt"
                )
            else:
                ckpt_and_config_path = os.path.join(path, f"end_of_cycle{cycle}.pt")
        else:
            ckpt_and_config_path = os.path.join(path, "ckpt.pt")
    # load model
    if train:
        config = yaml.safe_load(
            open(os.path.join(path, "checkpoints", "config.yml"), "r")
        )
    else:
        config = yaml.safe_load(open(os.path.join(path, "config.yml"), "r"))

    name = config["model"].pop("name")
    model = registry.get_model_class(model_type)(**config["model"]).to(device)
    config["model"]["name"] = name

    # get checkpoint
    print(f"Loading model weights from {ckpt_and_config_path}")
    try:
        checkpoint = {
            k: v.to(device)
            for k, v in torch.load(
                ckpt_and_config_path, map_location=torch.device("cpu")
            )["model_state"].items()
        }
    except:
        checkpoint = {
            k: v.to(device)
            for k, v in torch.load(
                ckpt_and_config_path, map_location=torch.device("cpu")
            )["state_dict"].items()
        }
    try:
        new_dict = {k[7:]: v for k, v in checkpoint.items()}
        try:
            model.load_state_dict(new_dict)
        except:
            if "atomic_mass" in new_dict:
                del new_dict["atomic_mass"]
            model.load_state_dict(new_dict)

    except:
        model.load_state_dict(checkpoint)

    return model, ckpt_and_config_path, config
