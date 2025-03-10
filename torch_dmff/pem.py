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
        eta = params["eta"] * self.const_lib.length_coeff
        charges = params["charge"]

        ds = ds * self.const_lib.length_coeff


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
            #eta_ij[both_electrode] = 0
        i_only_electrode = i_is_electrode & (~j_is_electrode)
        if torch.any(i_only_electrode):
                eta_ij[i_only_electrode] = eta[pairs[:, 0]][i_only_electrode] * (2.0 ** 0.5)
        j_only_electrode = (~i_is_electrode) & j_is_electrode
        if torch.any(j_only_electrode):
                eta_ij[j_only_electrode] = eta[pairs[:, 1]][j_only_electrode] * (2.0 ** 0.5)
        #pre_pair = torch.zeros_like(q_i)
        
        pre_pair = -self.eta_piecewise(eta_ij, ds)
            
        e_sr_pair = torch.sum(
            pre_pair * q_i * q_j * safe_inverse(ds, threshold=1e-4) * buffer_scales
        )
        
        pre_self = torch.zeros_like(charges)
        pre_self = safe_inverse(eta, threshold=1e-4) / (2 * self.const_lib.sqrt_pi)
        e_sr_self = torch.sum(pre_self * charges * charges)
        
        e_sr = (e_sr_pair + e_sr_self) * self.const_lib.dielectric
        # eV to user-defined energy unit
        return e_sr / self.const_lib.energy_coeff
    
    @staticmethod
    def eta_piecewise(eta, ds, threshold: float = 1e-4):
        safe_ratio = torch.where(eta > threshold, ds / eta, torch.ones_like(ds))
        return torch.where(eta > threshold, torch.erfc(safe_ratio), torch.zeros_like(eta))


class XMLDataLoadder:
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
        self.models: Dict[str, BaseForceModule] = {
            "coulomb": CoulombForceModule(rcut=rcut, ethresh=ethresh, 
                                          kspace=kspace, rspace=rspace,
                                          slab_corr=slab_corr, slab_axis=slab_axis, 
                                          units_dict=units_dict,sel=sel),
            "nblist": TorchNeighborList(cutoff=rcut),
            "gaussian": PEMGaussianDampingForceModule(units_dict=units_dict),
            }
        self._logger = self._setup_logger()
    
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
    
    def conp(
        self,
        n_atoms: int,
        electrode_mask: np.ndarray,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        num_electrode_atoms_dict: Dict,
        params: Dict[str, torch.Tensor],
        potential: np.array,
        ffield: bool=False,
        method: str="pgrad", 
        symm: bool=True,
        fi_cons_charge: Optional[float]=None,
        se_cons_charge: Optional[float]=None,
        conq: bool=False,
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
        potential : np.array
            potential information between electrodes, the order is 
            same with num_electrode_atoms_dict
        ffield : bool, optional
            whether the finite field is applied, by default False
        method : str, 
            optimization method, by default "pgrad"
        symm : bool, optional
            whether the total charge of two electrode is same, by default True

        fi_cons_charge : Optional[float], optional
            charge constraint for the first electrode, by default None
        se_cons_charge : Optional[float], optional
            charge constraint for the second electrode, by default None
        conq : bool, optional
            whether the constrained potential method is applied, by default False

        """
        if len(num_electrode_atoms_dict) != 2:
            raise KeyError("Only two electrodes are supported Now")
        if ffield :
            if self.slab_corr :
                raise KeyError("Slab correction and finite field cannot be used together")

        params["chi"] = self.coulomb_potential_add_chi(n_atoms, electrode_mask, positions, box, params["chi"],  params["eta"],params["charge"])
        

        electrode_params = {k: v[electrode_mask == 1] for k, v in params.items()}
        
        electrode_positions = positions[electrode_mask == 1]
        
       


        nblist = self.models["nblist"]
        
        pairs = nblist(electrode_positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()
        ##here we impose the constraint that the total charge of the electrode is zero

        first_key = list(num_electrode_atoms_dict.keys())[0]
        second_key = list(num_electrode_atoms_dict.keys())[1]

        if symm:
            elec_num = num_electrode_atoms_dict[first_key] + num_electrode_atoms_dict[second_key]
            constraint_matrix = torch.ones([1, elec_num], dtype=torch.float64)
            constraint_vals = torch.zeros(1, dtype=torch.float64)

        if fi_cons_charge is not None:
            if symm:
                raise KeyError("symm and cons_charge cannot be used together")
            if se_cons_charge is None:
                raise KeyError("cons_charge requires both first and second electrode charges")
            
            n1 = num_electrode_atoms_dict[first_key] 
            n2 = num_electrode_atoms_dict[second_key]   

            row1 = torch.cat([torch.ones((1, n1), dtype=torch.float64, device=positions.device),
                      torch.zeros((1, n2), dtype=torch.float64, device=positions.device)], dim=1)

            row2 = torch.cat([torch.zeros((1, n1), dtype=torch.float64, device=positions.device),
                      torch.ones((1, n2), dtype=torch.float64, device=positions.device)], dim=1)
            constraint_matrix = torch.cat([row1, row2], dim=0)
            constraint_vals = torch.tensor([fi_cons_charge, se_cons_charge], dtype=torch.float64, device=positions.device)

        ##if we have neutral electrode, we can use D to simplify the potential drop vector


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
                potential_drop = potential[0] - potential[1]
                electrode_params["chi"] = self.finite_field_add_chi(n_atoms, electrode_mask,num_electrode_atoms_dict, 
                                                      electrode_positions, "None", "None",
                                                      box, electrode_params["chi"], potential_drop,self.slab_axis)
                

        
        args = [
        electrode_positions,
        box,
        electrode_params["chi"],
        electrode_params["hardness"],
        electrode_params["eta"],
        pairs,
        ds,
        buffer_scales,
        ]
        
        if constraint_matrix is not None:
            args.append(constraint_matrix)
        if constraint_vals is not None:
            args.append(constraint_vals)

        if method == "pgrad":
            args.insert(0, electrode_params["charge"])

            energy, q_opt = self.solve_pgrad(*args)
        if method == "mat_inv":
            energy, q_opt, hessian_diag, fermi = self.solve_matrix_inversion(*args)            
        #print("QEq converges in %d step(s)" % self.converge_iter)

        charges = params["charge"].clone()
        charges[electrode_mask == 1] = q_opt

        ###compute the energy and force for entire system
        nblist = self.models["nblist"]
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()
        module = self.models["coulomb"]
        energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        '''
        module = self.models["gaussian"]
        energy += module(electrode_mask, positions, box, pairs, ds, buffer_scales, {"charge": charges, "eta": params["eta"]})
        '''
        forces = -calc_grads(energy, positions)
        

        return energy, forces, charges
    
    def conq(self,
        n_atoms: int,
        electrode_mask: np.array,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        num_electrode_atoms_dict: Dict,
        params: Dict[str, torch.Tensor],
        potential: np.array=[],
        ffield: bool=False,
        method: str="pgrad", 
        symm: bool=False,
        fi_cons_charge: float=None,
        se_cons_charge: float=None,
        conq: bool=True,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if fi_cons_charge is None or se_cons_charge is None:
                raise KeyError("conq requires both first and second electrode charges")
            return self.conp(
                n_atoms, electrode_mask, positions, box, num_electrode_atoms_dict, 
                params, potential, ffield, method, symm, fi_cons_charge, 
                se_cons_charge, conq
            ) 
    
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

        potential =  calc_grads(energy, charges)
        module = self.models["gaussian"]
        energy_gaussian = module(electrode_mask, positions, box, pairs, ds, buffer_scales, {"charge": charges, "eta": eta})
        potential_gaussian = calc_grads(energy_gaussian, charges)
        #potential =  calc_grads(energy, charges)
        
        #return potential + potential_gaussian
        return potential + potential_gaussian
    
    '''
    def calc_coulomb_potential(self, 
                           electrode_mask: np.array,
                           positions: torch.Tensor, 
                           box: torch.Tensor, 
                           charges: torch.tensor,
                           eta: torch.tensor
                           ):
        """
    Calculate the Coulomb potential for the system, including electrostatic interactions 
    and Gaussian short-range corrections for electrodes.
    
    Parameters:
    -----------
    electrode_mask: np.array
        Binary array indicating which atoms are part of electrodes
    positions: torch.Tensor
        Atom coordinates
    box: torch.Tensor
        Simulation box dimensions
    charges: torch.tensor
        Atomic charges
    eta: torch.tensor
        Gaussian width parameters for short-range corrections
        
    Returns:
    --------
    torch.Tensor
        Electrostatic potential at each atom site
        """
        # Calculate neighbor pairs and distances
        nblist = self.models["nblist"]
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()
    
        # Method 1: Diagnostic calculation to separately analyze contributions 
        # from Coulomb and Gaussian terms (for debugging/analysis only)
        with torch.no_grad():  # Use no_grad to prevent gradient accumulation
            module = self.models["coulomb"]
            coulomb_energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
            print(f"Before gaussian energy: {coulomb_energy}")
        
            # Create temporary copy to calculate Coulomb potential separately
            # without affecting the main computation graph
            temp_charges = charges.clone().detach().requires_grad_(True)
            temp_params = {"charge": temp_charges}
            temp_energy = module(positions, box, pairs, ds, buffer_scales, temp_params)
            coulomb_potential = calc_grads(temp_energy, temp_charges)
            print(f"Before Potential: {coulomb_potential}")
    
        # Method 2: Calculate the complete potential (recommended approach)
        # Reset main computation and start from scratch
        module_coulomb = self.models["coulomb"]
        module_gaussian = self.models["gaussian"]
    
        # Use fresh parameter dictionary to ensure clean computation graph
        params = {"charge": charges, "eta": eta}
    
        # Calculate total energy (Coulomb + Gaussian correction)
        total_energy = module_coulomb(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        total_energy += module_gaussian(electrode_mask, positions, box, pairs, ds, buffer_scales, params)
        print(f"After gaussian energy: {total_energy}")
    
        # Calculate gradients only once on the total energy
        potential = calc_grads(total_energy, charges)
        print(f"After Potential: {potential}")
    
        return potential
    '''
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
        
        lz = box[slab_axis][2]
        
        ### lammps fix electrode implementation
        ### cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and r
        if max_pos_first > max_pos_second:
            first_part = electrode_positions[:num_electrode_atoms_dict[first_key], slab_axis]/lz
            second_part = electrode_positions[num_electrode_atoms_dict[first_key]:, slab_axis]/lz 

            zprd_offset = -1 * -1 * torch.cat([first_part, second_part], dim=0)
        else:
            first_part = electrode_positions[:num_electrode_atoms_dict[first_key], slab_axis]/lz 
            second_part = electrode_positions[num_electrode_atoms_dict[first_key]:, slab_axis]/lz 
            zprd_offset = -1 * torch.cat([first_part, second_part], dim=0)
        
     
        potential =  potential_drop * zprd_offset
        return potential + chi


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

