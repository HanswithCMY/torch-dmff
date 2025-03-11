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

class PEMGaussianDampingForceModule(GaussianDampingForceModule):
    def __init__(
        self,
        units_dict: Optional[Dict] = None,
    ) -> None:
        GaussianDampingForceModule.__init__(self, units_dict)

    def forward(
        self,
        electrode_mask: np.ndarray,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Gaussian short-range damping energy model

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        params : Dict[str, torch.Tensor]
            {
                "charge": t_charges, # atomic charges
                "eta": t_eta, # Gaussian width in length unit
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        ##init charge and position with unit metal
        eta = params["eta"] * self.const_lib.length_coeff
        charges = params["charge"]
        ds = ds * self.const_lib.length_coeff

        #calculate effective gaussian width for gaussian damping
        q_i = charges[pairs[:, 0]].reshape(-1)
        q_j = charges[pairs[:, 1]].reshape(-1)
        electrode_mask = torch.tensor(electrode_mask, dtype=eta.dtype)
        i_is_electrode = electrode_mask[pairs[:, 0]] == 1
        j_is_electrode = electrode_mask[pairs[:, 1]] == 1

        eta_ij = torch.ones_like(q_i) * 1e-10

        both_electrode = i_is_electrode & j_is_electrode
        if torch.any(both_electrode):
            #2*ij = 2*sqrt(i^2 + j^2)
            eta_ij[both_electrode] = torch.sqrt((eta[pairs[:, 0]][both_electrode]**2 + 
                                                eta[pairs[:, 1]][both_electrode]**2) * 2) 
        i_only_electrode = i_is_electrode & (~j_is_electrode)
        if torch.any(i_only_electrode):
                eta_ij[i_only_electrode] = eta[pairs[:, 0]][i_only_electrode] * (2.0 ** 0.5)
        j_only_electrode = (~i_is_electrode) & j_is_electrode
        if torch.any(j_only_electrode):
                eta_ij[j_only_electrode] = eta[pairs[:, 1]][j_only_electrode] * (2.0 ** 0.5)
        
        #calculate correction short-range energy(eV)
        pre_pair = -self.eta_piecewise(eta_ij, ds)
            
        e_sr_pair = torch.sum(
            pre_pair * q_i * q_j * safe_inverse(ds, threshold=1e-4) * buffer_scales
        )
        
        pre_self = torch.zeros_like(charges)
        pre_self = safe_inverse(eta, threshold=1e-4) / (2 * self.const_lib.sqrt_pi)
        e_sr_self = torch.sum(pre_self * charges * charges)
        
        e_sr = (e_sr_pair + e_sr_self) * self.const_lib.dielectric
    

        '''
        ####print information detect wrong
        print("\n===== GAUSSIAN DAMPING POTENTIAL AND PAIR DETAILS =====")
        print("\n----- POTENTIAL VALUES -----")
        print("Atom ID | Is Electrode | Charge | Potential")
        print("-" * 45)
        for i in range(len(potential)):
            is_elec = "Yes" if electrode_mask[i] == 1 else "No"
            print(f"{i:7d} | {is_elec:^12s} | {charges[i].item():6.4f} | {potential[i].item():9.6f}")
        
    
        max_pairs_to_print = 20  
        print("\n----- PAIR INFORMATION (Sample) -----")
        print("Pair ID | Atom i | Atom j | i-elec | j-elec | Distance | eta_ij | pre_pair | Contrib")
        print("-" * 90)
        
        
        pair_contrib = pre_pair * q_i * q_j * safe_inverse(ds, threshold=1e-4) * buffer_scales
        
        
        _, top_indices = torch.sort(torch.abs(pair_contrib), descending=True)
        top_indices = top_indices[:max_pairs_to_print]
        
        for idx in top_indices:
            i, j = pairs[idx]
            i_elec = "Yes" if i_is_electrode[idx] else "No"
            j_elec = "Yes" if j_is_electrode[idx] else "No"
            print(f"{idx.item():7d} | {i.item():6d} | {j.item():6d} | {i_elec:^6s} | {j_elec:^6s} | "
                  f"{ds[idx].item()/self.const_lib.length_coeff:8.4f} | {eta_ij[idx].item():6.4f} | "
                  f"{pre_pair[idx].item():8.4f} | {pair_contrib[idx].item():8.4f}")
        
        
        print("\n----- PAIR TYPE STATISTICS -----")
        print(f"Total pairs: {len(pairs)}")
        print(f"Both electrode pairs: {torch.sum(both_electrode).item()}")
        print(f"i-only electrode pairs: {torch.sum(i_only_electrode).item()}")
        print(f"j-only electrode pairs: {torch.sum(j_only_electrode).item()}")
        print(f"No electrode pairs: {torch.sum((~i_is_electrode) & (~j_is_electrode)).item()}")
        
        print("\n----- ENERGY COMPONENTS -----")
        print(f"Pair interaction energy: {e_sr_pair.item()}")
        print(f"Self interaction energy: {e_sr_self.item()}")
        print(f"Total energy: {e_sr.item()}")
        print("================================================\n")
        '''
        # eV to user-defined energy unit
        return e_sr / self.const_lib.energy_coeff



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
        super().__init__(rcut, ethresh, kspace, 
                                rspace,slab_corr,slab_axis,
                                max_iter, ls_eps, eps, units_dict,
                                damping, sel)
        self.slab_axis = slab_axis
        self.slab_corr = slab_corr
        self.rcut = rcut

        self.models: Dict[str, BaseForceModule] = {
            "coulomb": CoulombForceModule(rcut=rcut, ethresh=ethresh, 
                                          kspace=kspace, rspace=rspace,
                                          slab_corr=slab_corr, slab_axis=slab_axis, 
                                          units_dict=units_dict,sel=sel),
            "nblist": TorchNeighborList(cutoff=rcut),
            "gaussian": PEMGaussianDampingForceModule(units_dict=units_dict),
            }
        self._logger = self._setup_logger()
        self.efield = 0.0
        self.conp_flag = False
        self.conq_flag = False
        self.ffield_flag = False

        if units_dict != None:
            self.units_dict = units_dict
    
    def _setup_logger(self):
        import logging
        logger = logging.getLogger("PEMModule")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def input_data_loader(
        self,
        n_atoms: int,
        electrode_mask: np.ndarray,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        num_electrode_atoms_dict: Dict,
        params: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constrained Potential Method implementation
        An instantiation of QEq Module for electrode systems totally

        Parameters
        ----------
        n_atoms : int
            number of atoms
        electrode_mask : np.ndarray
            mask for electrode atoms
            the order is same with positions, 1 for electrode atoms, 0 for others
        positions : torch.Tensor
            atomic positions
        box : Optional[torch.Tensor]
            simulation box
        num_electrode_atoms_dict : Dict
            number of electrode atoms for each sideelectrode
        params : Dict[str, torch.Tensor]
            parameters for optimization

        """
        if len(num_electrode_atoms_dict) != 2:
            raise KeyError("Only two electrodes are supported Now")

        self.num_electrode_atoms_dict = num_electrode_atoms_dict
        self.first_key = list(num_electrode_atoms_dict.keys())[0]
        self.second_key = list(num_electrode_atoms_dict.keys())[1]
        self.box = box
        self.positions = positions
        self.electrode_mask = electrode_mask
        self.params = params

        self.charge = params["charge"]

        params["chi"] = self.coulomb_potential_add_chi(n_atoms, electrode_mask, positions, box, params["chi"],  params["eta"],params["charge"])
        self.chi = params["chi"]

        #only consider the electrode atoms
        self.electrode_params = {k: v[electrode_mask == 1] for k, v in params.items()}        
        self.electrode_positions = positions[electrode_mask == 1]
        self.nblist = self.models["nblist"]
        self.pairs = self.nblist(self.electrode_positions, box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()


    
    def conp(self,potential: np.array,ffield: bool=False,method: str="pgrad",symm: bool=True,):
        #Add constant potential condition in chi 
        self.conp_flag = True

        potential_term = torch.cat([
            torch.full((self.num_electrode_atoms_dict[self.first_key],), potential[0], dtype=torch.float64),
            torch.full((self.num_electrode_atoms_dict[self.second_key],), potential[1], dtype=torch.float64)
            ])
        self.electrode_params["chi"] -= potential_term 
            
        if ffield :
            
            if self.slab_corr :
                raise KeyError("Slab correction and finite field cannot be used together")
            potential_drop = potential[0] - potential[1]
            self.electrode_params["chi"] = self.finite_field_add_chi(self.num_electrode_atoms_dict, 
                                                            self.electrode_positions,self.box, self.electrode_params["chi"], potential_drop,self.slab_axis)
            self.ffield_flag = True
        #input for optimization
        args = [
        self.electrode_positions,
        self.box,
        self.electrode_params["chi"],
        self.electrode_params["hardness"],
        self.electrode_params["eta"],
        self.pairs,
        self.ds,
        self.buffer_scales,
        ]

        if symm:
            ##here we impose the constraint that the total charge of the electrode is zero
            elec_num = self.num_electrode_atoms_dict[self.first_key] + self.num_electrode_atoms_dict[self.second_key]
            constraint_matrix = torch.ones([1, elec_num])
            constraint_vals = torch.zeros(1)
            args.append(constraint_matrix)
            args.append(constraint_vals)

        if method == "pgrad":
            args.insert(0, self.electrode_params["charge"])
            energy, q_opt = self.solve_pgrad(*args)
        if method == "mat_inv":
            energy, q_opt, hessian_diag, fermi = self.solve_matrix_inversion(*args)      
        charges = self.params["charge"].clone()     
        charges[self.electrode_mask == 1] = q_opt
        self.charge_opt = torch.tensor(charges, requires_grad=True)
        return self.charge_opt
    
    def conq(self,fi_cons_charge: float=None,se_cons_charge: float=None,ffield: bool=False,method: str="pgrad"):
        self.conq_flag = True

        if fi_cons_charge is None and se_cons_charge is None:
            raise KeyError("conq requires both first or second electrode charges constraints")
        n1 = self.num_electrode_atoms_dict[self.first_key] 
        n2 = self.num_electrode_atoms_dict[self.second_key]   
        if se_cons_charge is None or fi_cons_charge is None:
            if se_cons_charge is None:
                con_charge = fi_cons_charge
            else:
                con_charge = se_cons_charge
            n = n1 + n2
            constraint_matrix = torch.ones([1, n])
            constraint_vals = torch.tensor([con_charge])
        else:
            row1 = torch.cat([torch.ones((1, n1)),torch.zeros((1, n2))], dim=1)
            row2 = torch.cat([torch.zeros((1, n1)),torch.ones((1, n2))], dim=1)
            constraint_matrix = torch.cat([row1, row2], dim=0)
            constraint_vals = torch.tensor([fi_cons_charge, se_cons_charge])
        if ffield:
            raise KeyError("conq with finite field has not been implemented")
        
        args = [
        self.electrode_positions,
        self.box,
        self.electrode_params["chi"],
        self.electrode_params["hardness"],
        self.electrode_params["eta"],
        self.pairs,
        self.ds,
        self.buffer_scales,
        ]        
        args.append(constraint_matrix)
        args.append(constraint_vals)
        if method == "pgrad":
            args.insert(0, self.electrode_params["charge"])
            energy, q_opt = self.solve_pgrad(*args)
        if method == "mat_inv":
            energy, q_opt, hessian_diag, fermi = self.solve_matrix_inversion(*args)
        charges = self.params["charge"].clone()
        charges[self.electrode_mask == 1] = q_opt
        self.charge_opt = torch.tensor(charges, requires_grad=True)

        return self.charge_opt 
    

    
    def calc_coulomb_potential(self, 
                               electrode_mask : np.array,
                               positions : torch.Tensor, 
                               box : torch.Tensor, 
                               charges : torch.tensor,
                               eta : torch.tensor
                               ):
        """
        calculate the coulomb potential for the system
        """

        # calculate pairs
        nblist = self.models["nblist"]
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        module = self.models["coulomb"]
        energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        module = self.models["gaussian"]
        energy += module(electrode_mask, positions, box, pairs, ds, buffer_scales, {"charge": charges, "eta": eta})
        #user-defined energy unit to eV 
        energy =  energy * self.const_lib.energy_coeff
        
        potential =  calc_grads(energy, charges)

        return potential 
    

    def coulomb_potential_add_chi(self, 
                                  n_atoms: int,
                                  electrode_mask : np.array, 
                                  positions : torch.tensor, 
                                  box : torch.tensor, 
                                  chi : torch.Tensor,
                                  eta : torch.Tensor,
                                  charges : torch.Tensor):
        """
        Calculate the vector b and add it in chi
        """
        modified_charges = charges.clone().detach()
        modified_charges[electrode_mask == 1] = 0 
        modified_charges.requires_grad_(True)
        potential = self.calc_coulomb_potential(electrode_mask,positions, box, modified_charges, eta)
        return potential + chi

    def finite_field_add_chi(self,
                             num_electrode_atoms_dict: Dict,
                             electrode_positions : torch.tensor,
                             box : torch.Tensor, 
                             chi : torch.Tensor,
                             potential_drop: float,
                             slab_axis: int = 2,
                             ):
        """
        Compute the correction term for the finite field

        potential drop need to be the potentials of the first electrode minus the second
        """        

        first_key = list(num_electrode_atoms_dict.keys())[0]
        second_key = list(num_electrode_atoms_dict.keys())[1]
        ## find max position in slab_axis for left electrode
        max_pos_first = torch.max(
                electrode_positions[:num_electrode_atoms_dict[first_key],slab_axis])
        max_pos_second = torch.max(
                electrode_positions[num_electrode_atoms_dict[first_key]:,slab_axis])
        #only valid for orthogonality cell
        lz = box[slab_axis][slab_axis]
        normalized_positions = electrode_positions[:, slab_axis] / lz
        ### lammps fix electrode implementation
        ### cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and r
        if max_pos_first > max_pos_second:
            zprd_offset = -1 * -1 * normalized_positions
            self.efield = -1 * potential_drop / lz
        else:
            zprd_offset = -1 * normalized_positions
            self.efield = potential_drop / lz
        
     
        potential =  potential_drop * zprd_offset
        return potential + chi
    def Coulomb_Calculator(self):
        """
        Compute the Coulomb force for the system
        """
        # Set up neighbor list
        nblist = TorchNeighborList(cutoff=self.rcut)
        pairs = nblist(self.positions, self.box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # calculate point charge force
        if self.conp_flag or self.conq_flag:
            charge = self.charge_opt
        else:
            charge = self.charge

        module = self.models["coulomb"]
        energy = module(self.positions, self.box, pairs, ds, buffer_scales, {"charge": charge})
        module = self.models["gaussian"]
        energy += module(self.electrode_mask, self.positions, self.box, pairs, ds, buffer_scales, {"charge": charge, "eta": self.params["eta"]})

        forces = -calc_grads(energy, self.positions)

        if self.ffield_flag:
            forces += self.efield * charge.unsqueeze(1) * torch.tensor([0, 0, 1])
        

        return energy, forces


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

