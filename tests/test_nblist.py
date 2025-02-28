# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import freud
import numpy as np
import torch

from torch_dmff.nblist import TorchNeighborList
from torch_dmff.utils import to_numpy_array


class TestTorchNeighborList(unittest.TestCase):
    def setUp(self):
        # reference data
        rcut = 4.0
        l_box = 10.0
        box = np.diag([l_box, l_box, l_box])
        positions = np.random.rand(20, 3) * l_box

        fbox = freud.box.Box.from_matrix(box)
        aq = freud.locality.AABBQuery(fbox, positions)
        res = aq.query(positions, dict(r_max=rcut, exclude_ii=True))
        nblist = res.toNeighborList()
        nblist = np.vstack((nblist[:, 0], nblist[:, 1])).T
        nblist = nblist.astype(np.int32)
        msk = (nblist[:, 0] - nblist[:, 1]) < 0
        self.nblist_ref = nblist[msk]

        self.nblist = TorchNeighborList(rcut)
        self.positions = torch.tensor(positions)
        self.box = torch.tensor(box)

    def test_pairs(self):
        """
        Check that pairs are in the neighbor list.
        """
        pairs = self.nblist(self.positions, self.box)
        pairs = to_numpy_array(pairs)
        mask = pairs[:, 0] < pairs[:, 1]
        assert len(pairs[mask]) == len(self.nblist_ref)
        for p in pairs[mask]:
            mask = (self.nblist_ref[:, 0] == p[0]) & (self.nblist_ref[:, 1] == p[1])
            self.assertTrue(mask.any())


if __name__ == "__main__":
    unittest.main()
