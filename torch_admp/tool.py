# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import torch
import jax.numpy as jnp
from scipy import constants, stats, integrate
from openmm import app
from openmm.unit import angstrom

from dmff.api.xmlio import XMLIO
from dmff.api import DMFFTopology

from torch_dmff.base_force import BaseForceModule
from torch_dmff.utils import calc_grads
from torch_dmff.nblist import TorchNeighborList
from torch_dmff.pme import CoulombForceModule
from torch_dmff.qeq import GaussianDampingForceModule
from .qeq import QEqForceModule

from torch_dmff.utils import safe_inverse



class XMLDataLoader:
    """
    data preprocessor for XML file and PDB file
    """
    @staticmethod

    def load_xml(
        file_name : str,
        pdb_name : str,
        slab_res_list : list,
        electroyte_res_list : list,
        left_electrode : str,
        right_electrode : str,):
        xml = XMLIO()
        xml.loadXML(file_name)


        res = xml.parseResidues()
        ffinfo = xml.parseXML()

        charges = []
        types = []

        for r in res:
            res_charges = [a["charge"] for a in r["particles"]]
            charges.extend(res_charges)
            res_types = [int(a["type"]) for a in r["particles"]]
            types.extend(res_types)
        types = np.array(types)
    
        # data in nm
        pdb = app.PDBFile(pdb_name)
        dmfftop = DMFFTopology(from_top=pdb.topology)
        positions = pdb.getPositions(asNumpy=True).value_in_unit(angstrom)
        positions = jnp.array(positions)
        a, b, c = dmfftop.getPeriodicBoxVectors()
        n_atoms = dmfftop.getNumAtoms()

        #eta is different with lammps PEM,sigma(eta) = 1/sqrt(2)/η
        eta = np.zeros([n_atoms])
        chi = np.zeros([n_atoms])
        hardness = np.zeros([n_atoms])
    
        num_electrode_atoms_dict = {}

        for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
            type_idx = int(_data["attrib"]["type"])
            mask = types == type_idx
            eta[mask] = float(_data["attrib"]["eta"])
            chi[mask] = float(_data["attrib"]["chi"])
            hardness[mask] = float(_data["attrib"]["J"])

        # kJ/mol to eV/particle
        j2ev = constants.physical_constants["joule-electron volt relationship"][0]
        energy_coeff = j2ev * constants.kilo / constants.Avogadro

        # generate electrode mask
        res_list = []
        electrode_mask = np.zeros([n_atoms])
        i = 0
        for r in res:
            if r["name"] in slab_res_list:
                for _ in r["particles"]:
                    res_list.append(r["name"])
                    electrode_mask[i] = 1
                    i += 1
                if r["name"] == left_electrode:
                    l_num_ele = len(r["particles"])
                    num_electrode_atoms_dict.update({"l_num_ele": l_num_ele})
                elif r["name"] == right_electrode:
                    r_num_ele = len(r["particles"])
                    num_electrode_atoms_dict.update({"r_num_ele": r_num_ele})
            
                
            elif r["name"] in electroyte_res_list:
                for _ in r["particles"]:
                    res_list.append(r["name"])
                    i += 1

        # length: angstrom, energy: eV
        data_dict = {
            "n_atoms": n_atoms,
            "res_list": res_list,
            "electrode_mask": electrode_mask,
            "num_electrode_atoms_dict": num_electrode_atoms_dict,
            "position": np.array(positions),
            "box": np.array([a._value, b._value, c._value]) * 10.0,
            "chi": chi * energy_coeff,
            "hardness": hardness * energy_coeff,
            "eta": eta,
            "charge": charges,
        }
        return data_dict




class PoissonSolver:
    """
    code from xhyang
    """
    @staticmethod

    def run_poisson_1D(data_dict: Dict,
                   ffield: bool = False,
                   APPLIED_F: float = None,
                   periodic: bool = False,
                   bins: int = None):
        
        """data processing for calculate Poisson's equation"""
        lx = data_dict["box"][0][0]
        ly = data_dict["box"][1][1]
        z = data_dict["position"][:, 2]
        lz = max(z) - min(z)
        charge = data_dict["charge"]
    
        if bins is None:
            bins = int(lz / 0.2)
            density_factor = 1 / lx / ly / 0.2
        else:
            density_factor = 1 / lx / ly / (lz / bins)
    
        bin_mins, bin_edges, binnumber = stats.binned_statistic(
            z, charge, statistic="sum", bins=bins
        )
        charge_dens_profile = bin_mins * density_factor
    
        if ffield:
            if APPLIED_F is None:
                raise KeyError("APPLIED_F is required for finite field")
            int1, phi , r= PoissonSolver.integrate_poisson_1D_ff(APPLIED_F, bin_edges[:-1], charge_dens_profile, periodic)
        else:
            int1, phi, r = PoissonSolver.integrate_poisson_1D(bin_edges[:-1], charge_dens_profile, periodic)
    
        return int1,phi,r, charge_dens_profile

    
    


    def integrate_poisson_1D(r, charge_dens_profile, periodic=True):
        """Integrate Poisson's equation twice to get electrostatic potential from charge density.

            Inputs:
            r : 1D numpy array. Values of coordinates, in Angstroms.
            charge_dens_profile : 1D numpy array. Values of charge density, 
                in e*Angstrom^-3.
            periodic : Boolean. Optional keyword. If True, adds linear function to 
                ensure values at boundaries are periodic.
            Outputs:
            phi : Numpy array of same length as r. Electrostatic potential in V. 
                The value of phi at the first r value is set to be 0.
        """

        eps_0_factor = 8.854e-12/1.602e-19*1e-10 # Note: be careful about units!
        #    terms_1 = 4.359744650e-18/1.602176620898e-19
        #    charge_densi_2_SI = 1.602e-19*1e-30
        #    ang_2_SI = 1e-10
        int1 = integrate.cumulative_trapezoid(charge_dens_profile, r, initial=0)/eps_0_factor
        int2 = -integrate.cumulative_trapezoid(int1, r, initial=0)
        #    int2 = -scipy.integrate.cumtrapz(int1, r, initial=0)
        if periodic:
            # Ensure periodic boundary conditions by adding a linear function such
            # that the values at the boundaries are equal
            phi = int2 - ((int2[-1]-int2[0])/(r[-1]-r[0])*(r-r[0]) + int2[0])
        else:
            phi = int2
        return int1, phi, r

    #################################################################################################

    def integrate_poisson_1D_ff(APPLIED_F, r, charge_dens_profile, periodic=False):
        """Integrate Poisson's equation twice to get electrostatic potential from charge density.

            Inputs:
            r : 1D numpy array. Values of coordinates, in Angstroms.
            charge_dens_profile : 1D numpy array. Values of charge density, 
                in e*Angstrom^-3.
            periodic : Boolean. Optional keyword. If True, adds linear function to 
                ensure values at boundaries are periodic.
            Outputs:
            phi : Numpy array of same length as r. Electrostatic potential in V. 
                The value of phi at the first r value is set to be 0.
        """

        eps_0_factor = 8.854e-12/1.602e-19*1e-10 # Note: be careful about units!
        #    terms_1 = 4.359744650e-18/1.602176620898e-19
        #    charge_densi_2_SI = 1.602e-19*1e-30
        #    ang_2_SI = 1e-10
        int1 = integrate.cumulative_trapezoid(charge_dens_profile, r, initial=0)/eps_0_factor+APPLIED_F/(r[-1]-r[0])
        int2 = -integrate.cumulative_trapezoid(int1, r)
        #    int2 = -scipy.integrate.cumtrapz(int1, r, initial=0)
        if periodic:
            # Ensure periodic boundary conditions by adding a linear function such
            # that the values at the boundaries are equal
            phi = int2 - ((int2[-1]-int2[0])/(r[-1]-r[0])*(r-r[0]) + int2[0])
        else:
            phi = int2
        return int1, phi, r

