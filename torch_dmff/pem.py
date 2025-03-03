# SPDX-License-Identifier: LGPL-3.0-or-later
from dmff.api.xmlio import XMLIO
from openmm import app
from dmff.api import DMFFTopology
from scipy import constants
from scipy import stats
import scipy.integrate

import numpy as np
import jax.numpy as jnp
import openmm.unit as unit
from . import BaseForceModule

from .qeq import QEqForceModule
from typing import Optional, Dict
from torch_dmff.utils import calc_grads
from torch_dmff.nblist import TorchNeighborList
from torch_dmff.pme import CoulombForceModule
import torch

def load_xml(file_name : str,
             pdb_name : str,
             slab_res_list : list,
             eletroyte_res_list : list,
             left_electrode : str,
             right_electrode : str,):
    xml = XMLIO()
    xml.loadXML(file_name)


    res = xml.parseResidues()
    ffinfo = xml.parseXML()

    charges = []
    for r in res:
        res_charges = [a["charge"] for a in r["particles"]]
        charges.extend(res_charges)
    types = []
    for r in res:
        res_types = [int(a["type"]) for a in r["particles"]]
        types.extend(res_types)
    types = np.array(types)
    print(f"types: {types}")
    # data in nm
    pdb = app.PDBFile(pdb_name)
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = jnp.array(positions)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    n_atoms = dmfftop.getNumAtoms()
    eta = np.zeros([n_atoms])
    chi = np.zeros([n_atoms])
    hardness = np.zeros([n_atoms])

    num_electrode_atoms_dict = {}
    l_num_ele = 0
    r_num_ele = 0
    print(ffinfo["Forces"]["ADMPQeqForce"]["node"])
    '''    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        type_idx = int(_data["attrib"]["type"]) 
        print(f"type_idx: {type_idx}")
        print(f"_data_eta: {_data['attrib']['eta']}")
        eta[types == type_idx] = float(_data["attrib"]["eta"])
        print(f"eta_idx: {eta[types == type_idx]}")
        chi[types == type_idx] = float(_data["attrib"]["chi"])
        hardness[types == type_idx] = float(_data["attrib"]["J"])'''
    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        type_idx = int(_data["attrib"]["type"])
        print(f"type_idx: {type_idx}")
        print(f"_data_eta: {_data['attrib']['eta']}")
        mask = types == type_idx
        eta[mask] = float(_data["attrib"]["eta"])
        print(f"eta_idx: {eta[mask]}")
        chi[mask] = float(_data["attrib"]["chi"])
        hardness[mask] = float(_data["attrib"]["J"])
    print(f"eta: {eta}")
    print(f"chi: {chi}")
    print(f"hardness: {hardness}")
    # kJ/mol to eV/particle
    # kJ/mol to eV
    j2ev = constants.physical_constants["joule-electron volt relationship"][0]
    # kJ/mol to eV/particle
    energy_coeff = j2ev * constants.kilo / constants.Avogadro

    res_list = []
    electrode_mask = np.zeros([n_atoms])
    i = 0
    for r in res:
        
        if r["name"] in slab_res_list:
            for particle in r["particles"]:
                res_list.append(r["name"])
                electrode_mask[i] = 1
                i += 1
            if r["name"] == left_electrode:
                l_num_ele = len(r["particles"])
                num_electrode_atoms_dict.update({"l_num_ele": l_num_ele})
            elif r["name"] == right_electrode:
                r_num_ele = len(r["particles"])
                num_electrode_atoms_dict.update({"r_num_ele": r_num_ele})
            
                
        elif r["name"] in eletroyte_res_list:
            for particle in r["particles"]:
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


class PEMModule(QEqForceModule):
    """Polarizable Electrode Model

    Parameters
    ----------
    rcut : float
        cutoff radius for short-range interactions
    ethresh : float, optional
        energy threshold for electrostatic interaction, by default 1e-5
    kspace: bool
        whether the reciprocal part is included
    rspace: bool
        whether the real space part is included
    slab_corr: bool
        whether the slab correction is applied
    ≈
        axis at which the slab correction is applied
    max_iter: int, optional
        maximum number of iterations for optimization, by default 20
        only used for projected gradient method
    ls_eps: float, optional
        threshold for line search, by default 1e-4
        only used for projected gradient method
    eps: float, optional
        threshold for convergence, by default 1e-4
        only used for projected gradient method
    units_dict: Dict, optional
        dictionary of units, by default None
    """
    def __init__(
        self,
        rcut: float,
        ethresh: float = 1e-5,
        kspace: bool = True,
        rspace: bool = True,
        slab_corr: bool = False,
        slab_axis: int = 2,
        max_iter: int = 20,
        ls_eps: float = 1e-4,
        eps: float = 1e-4,
        units_dict: Optional[Dict] = None,
        damping: bool = True,
        sel: list[int] = None,
    ) -> None:
        QEqForceModule.__init__(self, rcut, ethresh, kspace, 
                                rspace,slab_corr,slab_axis,
                                max_iter, ls_eps, eps, units_dict,
                                damping, sel)
        print(f"slab_corr:{slab_corr}")
        print(slab_axis)
        print(type(slab_axis))
        self.slab_axis = slab_axis
        self.models: Dict[str, BaseForceModule] = {
            "coulomb": CoulombForceModule(rcut=rcut, ethresh=ethresh, 
                                          kspace=kspace, rspace=rspace,
                                          slab_corr=slab_corr, slab_axis=slab_axis, 
                                          units_dict=units_dict,sel=sel),
            "nblist": TorchNeighborList(cutoff=rcut),

        }

    def conp(self,
        n_atoms: int,
        electrode_mask: np.array,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        num_electrode_atoms_dict: Dict,
        params: Dict[str, torch.Tensor],
        potential: np.array,
        ffield: bool=False,
        method: str="pgrad", 
        symm: bool=True,
        fi_cons_charge: float=None,
        se_cons_charge: float=None,
        conq: bool=False,
        ) -> torch.Tensor:
        lz = box[2]
        print("lz:", lz)
        params["chi"] = self.coulomb_potential_add_chi(n_atoms, electrode_mask, positions, box, params["chi"], params["charge"])
        print(f"params[chi] in the first:{params['chi']}")

        electrode_params = {k: v[electrode_mask == 1] for k, v in params.items()}
        print
        electrode_positions = positions[electrode_mask == 1]
        #if len() != 2:
        #    raise KeyError("Only two electrodes are supported")
        potential_drop = potential[0] - potential[1]
        energy = torch.zeros(1, device=positions.device)


        #nblist = TorchNeighborList(cutoff=self.rcut)
        nblist = self.models["nblist"]
        
        pairs = nblist(electrode_positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()
        ##here we impose the constraint that the total charge of the electrode is zero

        if symm:
            elec_num = num_electrode_atoms_dict["l_num_ele"] + num_electrode_atoms_dict["r_num_ele"]
            constraint_matrix = torch.ones([1, elec_num], dtype=torch.float64)
            constraint_vals = torch.zeros(1, dtype=torch.float64)

        if fi_cons_charge is not None:
            if symm:
                raise KeyError("symm and cons_charge cannot be used together")
            if se_cons_charge is None:
                raise KeyError("cons_charge requires both first and second electrode charges")
            first_key = list(num_electrode_atoms_dict.keys())[0]
            second_key = list(num_electrode_atoms_dict.keys())[1]
            n1 = num_electrode_atoms_dict[first_key] 
            n2 = num_electrode_atoms_dict[second_key]   

    
            row1 = torch.cat([torch.ones((1, n1), dtype=torch.float64, device=positions.device),
                      torch.zeros((1, n2), dtype=torch.float64, device=positions.device)], dim=1)

            row2 = torch.cat([torch.zeros((1, n1), dtype=torch.float64, device=positions.device),
                      torch.ones((1, n2), dtype=torch.float64, device=positions.device)], dim=1)

            constraint_matrix = torch.cat([row1, row2], dim=0)
    

            constraint_vals = torch.tensor([fi_cons_charge, se_cons_charge], dtype=torch.float64, device=positions.device)

        ##if we have neutral electrode, we can use D to simplify the potential drop vector

        first_key = list(num_electrode_atoms_dict.keys())[0]
        second_key = list(num_electrode_atoms_dict.keys())[1]
        ## left electrode - right electrode
        #if first_key == "r_num_ele":
            #potential_drop = - potential_drop

        #D = torch.cat([
        #    torch.full((num_electrode_atoms_dict[first_key],), num_electrode_atoms_dict[first_key]/elec_num, dtype=torch.float64),
        #    torch.full((num_electrode_atoms_dict[second_key],), - num_electrode_atoms_dict[second_key]/elec_num, dtype=torch.float64)
        #    ])
        #potential_term = potential_drop * D
        if not conq:

            potential_term = torch.cat([
            torch.full((num_electrode_atoms_dict[first_key],), potential[0], dtype=torch.float64),
            torch.full((num_electrode_atoms_dict[second_key],), potential[1], dtype=torch.float64)
            ])
        


            electrode_params["chi"] -= potential_term 
            if ffield:
                print(f"before: electrode_params[chi]:{electrode_params['chi']}")
                electrode_params["chi"] = self.finite_field_add_chi(n_atoms, electrode_mask,num_electrode_atoms_dict, 
                                                      electrode_positions, "None", "None",
                                                      box, electrode_params["chi"], potential_drop,self.slab_axis)
                print(f"after: electrode_params[chi]:{electrode_params['chi']}")

        if method == "pgrad":
            energy, q_opt = self.solve_pgrad(
            electrode_params["charge"],
            electrode_positions,
            box,
            electrode_params["chi"],
            electrode_params["hardness"],
            electrode_params["eta"],
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
            )
        if method == "mat_inv":
            energy, q_opt, hessian_diag, fermi = self.solve_matrix_inversion(
            electrode_positions,
            box,
            electrode_params["chi"],
            electrode_params["hardness"],
            electrode_params["eta"],
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
            )            
        #forces = -calc_grads(energy, positions)

        print("QEq converges in %d step(s)" % self.converge_iter)
        print(f"final electrode result: {q_opt}")

        charges = torch.zeros(n_atoms, dtype=torch.float64)
        charges[electrode_mask == 1] = q_opt

        #nblist = TorchNeighborList(cutoff=self.rcut)
        nblist = self.models["nblist"]

        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()
        #module = CoulombForceModule(rcut=self.rcut, ethresh=self.ethresh, periodic=True)
        module = self.models["coulomb"]

        energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        forces = -calc_grads(energy, positions)

        return energy, forces, charges
    
    def conq(self,
        n_atoms: int,
        electrode_mask: np.array,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        num_electrode_atoms_dict: Dict,
        params: Dict[str, torch.Tensor],
        potential: np.array,
        ffield: bool=False,
        method: str="pgrad", 
        symm: bool=False,
        fi_cons_charge: float=None,
        se_cons_charge: float=None,
        conq: bool=True,
        ) -> torch.Tensor:
            if fi_cons_charge is None or se_cons_charge is None:
                raise KeyError("conq requires both first and second electrode charges")
            return self.conp(n_atoms, electrode_mask, positions, box, num_electrode_atoms_dict, params, potential, ffield, method, symm, fi_cons_charge, se_cons_charge, conq)
        
        



    def calc_coulomb_potential(self, 
                               positions : torch.Tensor, 
                               box : torch.Tensor, 
                               charges : torch.tensor,
                               ):


        # calculate pairs
        #nblist = TorchNeighborList(cutoff=self.rcut)
        nblist = self.models["nblist"]

        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        #module = CoulombForceModule(rcut=self.rcut, ethresh=self.ethresh, kspace=self.kspace, rspace=self.rspace,
        #                            slab_corr=self.slab_corr, slab_axis=self.slab_axis, units_dict=self.units_dict,sel=self.sel)
        module = self.models["coulomb"]

        energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})

        potential =  calc_grads(energy, charges)
        #forces = -calc_grads(energy, positions)
        return potential
    

    def coulomb_potential_add_chi(self, 
                                  n_atoms: int,
                                  electrode_mask : np.array, 
                                  positions : torch.tensor, 
                                  box : torch.tensor, 
                                  chi : torch.Tensor,
                                  charges : torch.Tensor):
        modified_charges = charges.clone().detach()
        modified_charges[electrode_mask == 1] = 0 
        modified_charges.requires_grad_(True)
        potential = self.calc_coulomb_potential(positions, box, modified_charges)
        print(potential)
        return potential + chi



    #lammps implementation
    def finite_field_add_chi(self,
                             n_atoms: int,
                             electrode_mask : np.array, 
                             num_electrode_atoms_dict: Dict,
                             electrode_positions : torch.tensor,
                             left_electrode : str,
                             right_electrode : str, 
                               box : torch.Tensor, 
                               chi : torch.Tensor,
                             potential_drop: float,
                             slab_axis: int = 2,
                             
                             ):
        

        first_key = list(num_electrode_atoms_dict.keys())[0]
        second_key = list(num_electrode_atoms_dict.keys())[1]
        ## find max position in slab_axis for left electrode
        '''if first_key == left_electrode:
            max_pos_left = torch.max(
                electrode_positions[:num_electrode_atoms_dict[first_key],slab_axis])
            max_pos_right = torch.max(
                electrode_positions[num_electrode_atoms_dict[first_key]:,slab_axis])
        else:
            max_pos_left = torch.max(
                electrode_positions[num_electrode_atoms_dict[first_key]:,slab_axis])
            max_pos_right = torch.max(
                electrode_positions[:num_electrode_atoms_dict[first_key],slab_axis])'''

        max_pos_first = torch.max(
                electrode_positions[:num_electrode_atoms_dict[first_key],slab_axis])
        max_pos_second = torch.max(
                electrode_positions[num_electrode_atoms_dict[first_key]:,slab_axis])
        print("box shape:", box.shape)

        lz = box[slab_axis][2]
        print("lz:", lz)
        ##cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and delta_r
        '''
        if max_pos_first > max_pos_second:
            first_part = electrode_positions[:num_electrode_atoms_dict[first_key], slab_axis]/lz
            second_part = electrode_positions[num_electrode_atoms_dict[first_key]:, slab_axis]/lz + 1

            zprd_offset = -1 * torch.cat([first_part, second_part], dim=0)
        else:
            first_part = electrode_positions[:num_electrode_atoms_dict[first_key], slab_axis]/lz + 1
            second_part = electrode_positions[num_electrode_atoms_dict[first_key]:, slab_axis]/lz 
            zprd_offset = torch.cat([first_part, second_part], dim=0)
        '''
        print(f"potential_drop:{potential_drop}")

        
        first_part = electrode_positions[:num_electrode_atoms_dict[first_key], slab_axis]/lz 
        second_part = electrode_positions[num_electrode_atoms_dict[first_key]:, slab_axis]/lz 
        zprd_offset =   -1 * torch.cat([first_part, second_part], dim=0)       
        potential =  potential_drop * zprd_offset
        return potential + chi
            
def run_poisson_1D(data_dict: Dict,
                   ffield: bool = False,
                   APPLIED_F: float = None,
                   periodic: bool = False,
                   bins: int = None):
    ##### data processing for calculate Poisson's equation
    lx = data_dict["box"][0][0]
    ly = data_dict["box"][1][1]
    lz = data_dict["box"][2][2]
    z = data_dict["position"][:, 2]
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
        int1, phi , r= integrate_poisson_1D_ff(APPLIED_F, bin_edges[:-1], charge_dens_profile, periodic)
    else:
        int1, phi, r = integrate_poisson_1D(bin_edges[:-1], charge_dens_profile, periodic)
    
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
    int1 = scipy.integrate.cumulative_trapezoid(charge_dens_profile, r, initial=0)/eps_0_factor
    int2 = -scipy.integrate.cumulative_trapezoid(int1, r, initial=0)
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
    int1 = scipy.integrate.cumulative_trapezoid(charge_dens_profile, r, initial=0)/eps_0_factor+APPLIED_F/(r[-1]-r[0])
    int2 = -scipy.integrate.cumulative_trapezoid(int1, r)
    #    int2 = -scipy.integrate.cumtrapz(int1, r, initial=0)
    if periodic:
        # Ensure periodic boundary conditions by adding a linear function such
        # that the values at the boundaries are equal
        phi = int2 - ((int2[-1]-int2[0])/(r[-1]-r[0])*(r-r[0]) + int2[0])
    else:
        phi = int2
    return int1, phi, r

