# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import openmm as mm
import torch
from ase import io
from openmm import app
from openmm.unit import angstrom
from scipy import constants

from torch_dmff.nblist import TorchNeighborList
from torch_dmff.pme import CoulombForceModule
from torch_dmff.utils import calc_grads, to_numpy_array

# torch.set_default_dtype(torch.float64)

# kJ/mol to eV/particle
energy_coeff = (
    constants.physical_constants["joule-electron volt relationship"][0]
    * constants.kilo
    / constants.Avogadro
)
# kJ/(mol nm) to eV/particle/A
force_coeff = energy_coeff * constants.angstrom / constants.nano



class TestOpenMMSimulation:
    def __init__(self) -> None:
        self.rcut = 5.0
        self.l_box = 20.0
        self.ethresh = 1e-5
        self.n_atoms = 100

        self.charges = np.random.uniform(-1.0, 1.0, (self.n_atoms))
        self.charges -= self.charges.mean()
        self.positions = np.random.rand(self.n_atoms, 3) * self.l_box

    def setup(self, real_space=True):
        self.system = mm.System()
        self.system.setDefaultPeriodicBoxVectors(
            (self.l_box * angstrom, 0, 0),
            (0, self.l_box * angstrom, 0),
            (0, 0, self.l_box * angstrom),
        )
        # NonbondedForce, Particle Mesh Ewald
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(self.rcut * angstrom)
        nonbonded.setEwaldErrorTolerance(self.ethresh)
        nonbonded.setIncludeDirectSpace(real_space)

        self.system.addForce(nonbonded)
        # add ions to the system
        for ii in range(self.n_atoms):
            self.system.addParticle(1.0)  # assume the mass is 1 for simplicity
            nonbonded.addParticle(self.charges[ii], 0, 0)

        dummy_integrator = mm.CustomIntegrator(0)
        # platform = mm.Platform.getPlatformByName("CUDA")
        # create simulation
        self.simulation = app.Simulation(
            topology=None,
            system=self.system,
            integrator=dummy_integrator,
            # platform=platform,
        )
        self.simulation.context.setPositions(self.positions * angstrom)

    def run(self):
        """
        return energy [eV], forces [eV/A]
        """
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        forces = state.getForces(asNumpy=True)
        energy = state.getPotentialEnergy()
        return (
            np.atleast_1d(energy._value)[0] * energy_coeff,
            forces._value.reshape(-1, 3) * force_coeff,
        )


class TestOBCCoulombForceModule(unittest.TestCase):
    """
    Coulomb interaction under open boundary condition
    """

    def setUp(self) -> None:
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_obc/system.data"),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        self.box = None
        charges = atoms.get_initial_charges()

        self.positions = torch.tensor(positions, requires_grad=True)
        self.charges = torch.tensor(charges)

        self.nblist = TorchNeighborList(cutoff=4.0)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()

        self.module = CoulombForceModule(rcut=5.0, ethresh=1e-5)
        # test jit-able
        self.jit_module = torch.jit.script(self.module)

    def test_numerical(self):
        ref_energy = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_obc/thermo.out")
        ).reshape(-1)[1]
        ref_atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_obc/dump.lammpstrj")
        )
        ref_forces = ref_atoms.get_forces()

        energy = self.module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = self.jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        # energy [eV]
        self.assertAlmostEqual(energy.item(), ref_energy, places=5)
        self.assertAlmostEqual(jit_energy.item(), ref_energy, places=5)
        # force [eV/A]
        self.assertTrue(
            np.allclose(
                to_numpy_array(forces).reshape(-1, 3),
                ref_forces,
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                to_numpy_array(jit_forces).reshape(-1, 3),
                ref_forces,
                atol=1e-5,
            )
        )


class TestPBCCoulombForceModule(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_system = TestOpenMMSimulation()
        self.ref_system.setup(real_space=True)

        self.positions = torch.tensor(self.ref_system.positions, requires_grad=True)
        self.charges = torch.tensor(self.ref_system.charges)
        self.box = torch.tensor(
            np.diag(
                [self.ref_system.l_box, self.ref_system.l_box, self.ref_system.l_box]
            )
        )

        self.nblist = TorchNeighborList(cutoff=self.ref_system.rcut)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()

        self.module = CoulombForceModule(
            rcut=self.ref_system.rcut,
            ethresh=self.ref_system.ethresh,
        )
        # test jit-able
        self.jit_module = torch.jit.script(self.module)

    def test_numerical(self):
        energy = self.module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = self.jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        nonbonded = self.ref_system.system.getForce(0)
        # A^-1 to nm^-1 for kappa
        nonbonded.setPMEParameters(
            self.module.kappa * 10.0,
            self.module.kmesh[0].item(),
            self.module.kmesh[1].item(),
            self.module.kmesh[2].item(),
        )
        # simulation = self.ref_system.simulation
        # ewald_params = nonbonded.getPMEParametersInContext(simulation.context)
        # self.kappa = ewald_params[0] / 10.0
        # self.kmesh = tuple(ewald_params[1:])
        ref_energy, ref_forces = self.ref_system.run()

        # energy [eV]
        self.assertTrue(
            np.isclose(
                energy.item(),
                ref_energy,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.isclose(
                jit_energy.item(),
                ref_energy,
                atol=1e-4,
            )
        )
        # force [eV/A]
        self.assertTrue(
            np.allclose(
                to_numpy_array(forces).reshape(-1, 3),
                ref_forces,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                to_numpy_array(jit_forces).reshape(-1, 3),
                ref_forces,
                atol=1e-4,
            )
        )

        # self.ref_system.setup(real_space=False)
        # ref_energy_reciprocal, ref_forces_reciprocal = self.ref_system.run()
        # print(ref_energy - ref_energy_reciprocal)
        # print(self.module.real_energy)

        # print("non-neutral energy: ", self.module.non_neutral_energy)
        # print("reciprocal")
        # print(ref_energy_reciprocal)
        # print((self.module.reciprocal_energy + self.module.self_energy).item())


class TestPBCSlabCorrCoulombForceModule(unittest.TestCase):
    def setUp(self) -> None:
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_pbc/system.data"),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        box = atoms.get_cell().array
        charges = atoms.get_initial_charges()

        self.positions = torch.tensor(positions, requires_grad=True)
        self.box = torch.tensor(box)
        self.charges = torch.tensor(charges)

        self.nblist = TorchNeighborList(cutoff=4.0)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()

    # def make_ref_data(self, axis: int):
    #     atoms = io.read(
    #         str(Path(__file__).parent / "data/lmp_coul_pbc/system.data"),
    #         format="lammps-data",
    #     )
    #     positions = atoms.get_positions()
    #     box = atoms.get_cell()
    #     charges = atoms.get_initial_charges()

    #     self.dipole = np.sum(
    #         positions * charges[:, None], axis=0
    #     )
    #     self.tot_charge = np.sum(charges)

    #     dipole = self.dipole[axis]
    #     z = positions[:, axis]

    #     volume = atoms.get_volume()
    #     epsilon = constants.epsilon_0 / constants.elementary_charge * constants.angstrom
    #     coeff = 2 * np.pi / volume / (4 * np.pi * epsilon)
    #     e = (
    #         dipole**2
    #         - self.tot_charge * np.sum(charges * z**2)
    #         + self.tot_charge**2 * box[axis, axis]**2 / 12
    #     )
    #     e *= coeff
    #     return e

    def lammps_ref_data(self):
        e1 = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_pbc/thermo.out")
        ).reshape(-1)[1]
        e2 = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_pbc_slab_corr/thermo.out")
        ).reshape(-1)[1]
        ref_energy = e2 - e1

        atoms = io.read(str(Path(__file__).parent / "data/lmp_coul_pbc/dump.lammpstrj"))
        f1 = atoms.get_forces()
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_pbc_slab_corr/dump.lammpstrj")
        )
        f2 = atoms.get_forces()
        ref_force = f2 - f1
        return ref_energy, ref_force

    def test_numerical(self):
        # rcut and ethresh are not used in slab correction calculation
        module = CoulombForceModule(
            rcut=self.nblist.cutoff,
            ethresh=1e-3,
            kspace=False,
            rspace=False,
            slab_corr=True,
            slab_axis=2,
        )
        jit_module = torch.jit.script(module)

        energy = module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        # ref_energy = self.make_ref_data(axis=2)
        ref_energy, ref_forces = self.lammps_ref_data()

        # energy [eV]
        self.assertAlmostEqual(energy.item(), ref_energy, places=5)
        self.assertAlmostEqual(jit_energy.item(), ref_energy, places=5)
        # force [eV/A]
        self.assertTrue(
            np.allclose(
                to_numpy_array(forces).reshape(-1, 3),
                ref_forces,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                to_numpy_array(jit_forces).reshape(-1, 3),
                ref_forces,
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
