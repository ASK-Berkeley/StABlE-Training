import bisect
import logging
import pickle
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch_geometric.data import Batch, Dataset

from mdsim.common.registry import registry


@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(LmdbDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


def data_list_collater(data_list, otf_graph=True):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )
    return batch
