# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np
import torch

from torch_admp.env import DEVICE
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.qeq import GaussianDampingForceModule, QEqForceModule, SiteForceModule
from torch_admp.utils import calc_grads, to_numpy_array

# torch.set_default_dtype(torch.float64)

rcut = 4.0
ethresh = 1e-5
l_box = 10.0
n_atoms = 100


class JITTest:
    def test(
        self,
    ):
        positions = np.random.rand(n_atoms, 3) * l_box
        if self.periodic:
            box = np.diag([l_box, l_box, l_box])
        else:
            box = None
        charges = np.random.uniform(-1.0, 1.0, (n_atoms))
        charges -= charges.mean()

        positions = torch.tensor(positions, requires_grad=True).to(DEVICE)
        if self.periodic:
            box = torch.tensor(box).to(DEVICE)
        charges = torch.tensor(charges, requires_grad=True).to(DEVICE)

        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        params = {
            "charge": charges,
            "eta": torch.ones(n_atoms).to(DEVICE),
            "chi": torch.ones(n_atoms).to(DEVICE),
            "hardness": torch.zeros(n_atoms).to(DEVICE),
        }
        energy = self.module(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            params,
        )
        jit_energy = self.jit_module(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            params,
        )
        grad = calc_grads(energy, charges)
        jit_grad = calc_grads(jit_energy, charges)

        self.assertAlmostEqual(energy.item(), jit_energy.item())
        self.assertTrue(
            np.allclose(
                to_numpy_array(grad),
                to_numpy_array(jit_grad),
            )
        )

        torch.jit.save(self.jit_module, "./frozen_model.pth", {})

    def tearDown(self):
        for f in os.listdir("."):
            if f == "frozen_model.pth":
                os.remove(f)


class TestOBCCoulombForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = False
        self.module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestPBCCoulombForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestSlabCorrXForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = CoulombForceModule(
            rcut=rcut,
            ethresh=ethresh,
            slab_corr=True,
            slab_axis=0,
        )
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestSlabCorrYForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = CoulombForceModule(
            rcut=rcut,
            ethresh=ethresh,
            slab_corr=True,
            slab_axis=1,
        )
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestSlabCorrZForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = CoulombForceModule(
            rcut=rcut,
            ethresh=ethresh,
            slab_corr=True,
            slab_axis=2,
        )
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestGaussianDampingForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = GaussianDampingForceModule()
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestSiteForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = SiteForceModule()
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


class TestQEqForceModule(unittest.TestCase, JITTest):
    def setUp(self):
        self.periodic = True
        self.module = QEqForceModule(
            rcut=rcut,
            ethresh=ethresh,
        )
        self.jit_module = torch.jit.script(self.module)

    def tearDown(self):
        JITTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
