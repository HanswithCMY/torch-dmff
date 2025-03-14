# SPDX-License-Identifier: LGPL-3.0-or-later
import jax.numpy as jnp
import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from scipy import constants

from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule
from torch_admp.utils import calc_grads

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)


def load_test_data():
    xml = XMLIO()
    xml.loadXML("qeq.xml")
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    # data in nm
    pdb = app.PDBFile("qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = jnp.array(positions)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    n_atoms = dmfftop.getNumAtoms()
    eta = np.zeros([n_atoms])
    chi = np.zeros([n_atoms])
    hardness = np.zeros([n_atoms])
    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        eta[types == _data["attrib"]["type"]] = float(_data["attrib"]["eta"])
        chi[types == _data["attrib"]["type"]] = float(_data["attrib"]["chi"])
        hardness[types == _data["attrib"]["type"]] = float(_data["attrib"]["J"])

    # kJ/mol to eV/particle
    # kJ/mol to eV
    j2ev = constants.physical_constants["joule-electron volt relationship"][0]
    # kJ/mol to eV/particle
    energy_coeff = j2ev * constants.kilo / constants.Avogadro
    # length: angstrom, energy: eV
    data_dict = {
        "n_atoms": n_atoms,
        "position": np.array(positions),
        "box": np.array([a._value, b._value, c._value]) * 10.0,
        "chi": chi * energy_coeff,
        "hardness": hardness * energy_coeff,
        "eta": eta,
        "charge": charges,
    }
    return data_dict


if __name__ == "__main__":
    rcut = 8.0
    ethresh = 1e-5

    data_dict = load_test_data()

    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # calculate pairs
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)
    energy, q_opt = module.solve_pgrad(
        charges,
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        constraint_vals,
    )
    forces = -calc_grads(energy, positions)

    print("QEq converges in %d step(s)" % module.converge_iter)
