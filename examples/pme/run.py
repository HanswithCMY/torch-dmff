# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    rcut = 6.0
    n_atoms = 500
    ethresh = 1e-4

    l_box = 20.0

    positions = np.random.rand(n_atoms, 3) * l_box
    box = np.diag([l_box, l_box, l_box])
    charges = np.random.uniform(-1.0, 1.0, (n_atoms))
    charges -= charges.mean()

    positions = torch.tensor(
        positions,
        requires_grad=True,
    )
    box = torch.tensor(
        box,
        requires_grad=False,
    )
    charges = torch.tensor(
        charges,
        requires_grad=False,
    )

    # calculate pairs
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
    energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    forces = -calc_grads(energy, positions)
