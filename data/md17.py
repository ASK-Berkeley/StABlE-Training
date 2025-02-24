# Script to download MD17 datasets: Adapted from https://github.com/kyonofx/MDsim/tree/main/preprocessing


import os
import argparse
from pathlib import Path
import pickle

import lmdb
import numpy as np
from tqdm import tqdm
from urllib import request as request
from sklearn.model_selection import train_test_split

from arrays_to_graphs import AtomsToGraphs
from mdsim.common.utils import EV_TO_KCAL_MOL

MD17_mols = [
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic_acid",
    "toluene",
    "uracil",
]

datasets_dict = dict(
    aspirin="md17_aspirin.npz",
    benzene="md17_benzene2017.npz",
    ethanol="md17_ethanol.npz",
    malonaldehyde="md17_malonaldehyde.npz",
    naphthalene="md17_naphthalene.npz",
    salicylic_acid="md17_salicylic.npz",
    toluene="md17_toluene.npz",
    uracil="md17_uracil.npz",
)


def download(molecule, data_path):
    url = "http://www.quantum-machine.org/gdml/data/npz/" + datasets_dict[molecule]
    request.urlretrieve(url, os.path.join(data_path, datasets_dict[molecule]))
    print(f"{molecule} downloaded.")


def write_to_lmdb(molecule, data_path, db_path, size, contiguous=False):
    print(f"process MD17 molecule: {molecule}.")
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device="cpu",
    )

    npzname = datasets_dict[molecule]
    data_file = Path(data_path) / npzname
    Path(data_path).mkdir(parents=True, exist_ok=True)

    if not data_file.is_file():
        download(molecule, data_path)
    all_data = np.load(data_file)

    n_points = all_data.f.R.shape[0]
    atomic_numbers = all_data.f.z
    atomic_numbers = atomic_numbers.astype(np.int64)
    positions = all_data.f.R
    force = all_data.f.F / EV_TO_KCAL_MOL
    energy = all_data.f.E / EV_TO_KCAL_MOL
    lengths = np.ones(3)[None, :] * 30.0
    angles = np.ones(3)[None, :] * 90.0

    size_dict = {"1k": 1000, "10k": 10000, "50k": 50000}
    lim = size_dict[size]

    train_size = int(0.95 * lim)
    val_size = int(0.05 * lim)
    test_size = 10000

    for dataset_size, train_size, val_size in zip([size], [train_size], [val_size]):
        print(
            f"processing molecule {molecule} dataset with size {dataset_size}, train size {train_size}, val size {val_size}, test size {test_size}"
        )

        if contiguous:
            # extract contiguous samples so we can compute dynamical observables
            train = np.linspace(0, train_size - 1, train_size, dtype=np.int32)
            val = np.linspace(
                train_size, train_size + val_size - 1, val_size, dtype=np.int32
            )
            test = np.linspace(
                train_size + val_size,
                train_size + val_size + test_size - 1,
                test_size,
                dtype=np.int32,
            )
        else:
            train_val_pool, test = train_test_split(
                np.arange(n_points),
                train_size=n_points - test_size,
                test_size=test_size,
                random_state=123,
            )
            size = train_size + val_size
            train_val = train_val_pool[: train_size + val_size]
            train, val = train_test_split(
                train_val, train_size=train_size, test_size=val_size, random_state=123
            )
        ranges = [train, val, test]

        norm_stats = {
            "e_mean": energy[train].mean(),
            "e_std": energy[train].std(),
            "f_mean": force[train].mean(),
            "f_std": force[train].std(),
        }
        save_path = Path(db_path) / molecule / dataset_size
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "metadata", norm_stats)

        for spidx, split in enumerate(["train", "val", "test"]):
            print(f"processing split {split}.")
            save_path = Path(db_path) / molecule / dataset_size / split
            save_path.mkdir(parents=True, exist_ok=True)
            db = lmdb.open(
                str(save_path / "data.lmdb"),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )
            for i, idx in enumerate(tqdm(ranges[spidx])):
                natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
                data = a2g.convert(
                    natoms,
                    positions[idx],
                    atomic_numbers,
                    lengths,
                    angles,
                    energy[idx],
                    force[idx],
                )
                data.sid = 0
                data.fid = idx
                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()

            # Save count of objects in lmdb.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
            txn.commit()

            db.sync()
            db.close()

            # nequip
            data = {
                "z": atomic_numbers,
                "E": energy[ranges[spidx]],
                "F": force[ranges[spidx]],
                "R": all_data.f.R[ranges[spidx]],
            }
            np.savez(save_path / "nequip_npz", **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--data_path", type=str, default="./DATAPATH/md17")
    parser.add_argument("--db_path", type=str, default="./DATAPATH/md17")
    parser.add_argument("--size", type=str, default="10k")
    parser.add_argument("--contiguous", action="store_true")
    args = parser.parse_args()
    assert (
        args.molecule in MD17_mols
    ), "<molecule> must be one of the 8 molecules in MD17."
    write_to_lmdb(
        args.molecule, args.data_path, args.db_path, args.size, args.contiguous
    )
