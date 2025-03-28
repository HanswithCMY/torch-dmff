# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Dict, List, Tuple, Union
import torch



from torch_admp.base_force import BaseForceModule
from torch_admp.utils import calc_grads
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.qeq import GaussianDampingForceModule
from torch_admp.qeq import pgrad_optimize
from .qeq import QEqForceModule

from torch_admp.utils import safe_inverse

class PEMGaussianDampingForceModule(GaussianDampingForceModule):
    def __init__(
        self,
        units_dict: Optional[Dict] = None,
    ) -> None:
        GaussianDampingForceModule.__init__(self, units_dict)

    def forward(
        self,
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



        #2*ij = 2*sqrt(i^2 + j^2)
        eta_ij = torch.sqrt((eta[pairs[:, 0]]**2 + eta[pairs[:, 1]]**2) * 2) 
        eta_ij[eta_ij == 0] = 1e-10
        #calculate correction short-range energy(eV)
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
        ffield: bool = False,
        ) -> None:
        super().__init__(rcut, ethresh, kspace, 
                                rspace,slab_corr,slab_axis,
                                max_iter, ls_eps, eps, units_dict,
                                damping, sel)
        self.slab_axis = slab_axis
        self.slab_corr = slab_corr
        self.rcut = rcut
        
        
        self.model_coulomb =  CoulombForceModule(rcut=rcut, ethresh=ethresh, 
                                          kspace=kspace, rspace=rspace,
                                          slab_corr=slab_corr, slab_axis=slab_axis, 
                                          units_dict=units_dict,sel=sel)
        self.model_nblist = TorchNeighborList(cutoff=rcut)
        self.model_gaussian = PEMGaussianDampingForceModule(units_dict=units_dict)
        
        
        self.models: Dict[str, BaseForceModule] = {
            "coulomb": CoulombForceModule(rcut=rcut, ethresh=ethresh, 
                                          kspace=kspace, rspace=rspace,
                                          slab_corr=slab_corr, slab_axis=slab_axis, 
                                          units_dict=units_dict,sel=sel),
            "nblist": TorchNeighborList(cutoff=rcut),
            "gaussian": PEMGaussianDampingForceModule(units_dict=units_dict),
            }
        
        self.efield = 0.0
        self.conp_flag = False
        self.conq_flag = False
        self.ffield_flag = ffield
        self.charge_opt = None

        if units_dict != None:
            self.units_dict = units_dict



    

    @torch.jit.export
    def calc_coulomb_potential(self, 
                               positions : torch.Tensor, 
                               box : torch.Tensor, 
                               charges : torch.Tensor,
                               eta : torch.Tensor
                               )-> torch.Tensor:
        """
        calculate the coulomb potential for the system
        """

        # calculate pairs
        nblist = self.model_nblist
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        energy = self.model_coulomb(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        energy += self.model_gaussian(positions, box, pairs, ds, buffer_scales, {"charge": charges, "eta": eta})
        #user-defined energy unit to eV 
        energy =  energy * self.const_lib.energy_coeff
        potential =  calc_grads(energy, charges)

        return potential 
    
    @torch.jit.export
    def coulomb_potential_add_chi(self, 
                                  electrode_mask : torch.Tensor, 
                                  positions : torch.Tensor, 
                                  box : torch.Tensor, 
                                  chi : torch.Tensor,
                                  eta : torch.Tensor,
                                  charges : torch.Tensor)-> torch.Tensor:
        """
        Calculate the vector b and add it in chi
        """
        modified_charges = torch.zeros_like(charges)
        modified_charges[electrode_mask == 0] = charges[electrode_mask == 0]
        modified_charges.requires_grad_(True)
        potential = self.calc_coulomb_potential(positions, box, modified_charges, eta)
        return potential + chi
    
    @torch.jit.export
    def finite_field_add_chi(self,
                                 positions : torch.Tensor,
                                 box : torch.Tensor,
                                 electrode_mask : torch.Tensor,
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the correction term for the finite field

        potential  need to be same in the electrode_mask
        potential drop is potential[0] - potential[1]
        """        
        

        potential = torch.tensor([0.0, 0.0])
        electrode_mask_mid_1 = electrode_mask[electrode_mask != 0]
        if len(electrode_mask_mid_1) == 0:
            raise ValueError("No nonzero electrode values found in electrode_mask.")
        potential[0] = electrode_mask_mid_1[0]
        electrode_mask_mid_2 = electrode_mask_mid_1[electrode_mask_mid_1 != electrode_mask_mid_1[0]]
        if len(electrode_mask_mid_2) == 0:
            potential[1] = electrode_mask_mid_1[0]
        else:
            potential[1] = electrode_mask_mid_2[0]

        if not torch.all(electrode_mask_mid_2 == electrode_mask_mid_2[0]):
            raise KeyError("Only two electrodes are supported now")
        

        slab_axis = self.slab_axis
        
        first_electrode = torch.zeros_like(electrode_mask)
        second_electrode = torch.zeros_like(electrode_mask)

        first_electrode[electrode_mask == potential[0]] = 1
        second_electrode[electrode_mask == potential[1]] = 1
        potential_drop = potential[0] - potential[1]
        
        ## find max position in slab_axis for left electrode
        max_pos_first = torch.max(
                positions[first_electrode==1,slab_axis])
        max_pos_second = torch.max(
                positions[second_electrode==1,slab_axis])
        #only valid for orthogonality cell
        lz = box[slab_axis][slab_axis]
        normalized_positions = positions[:, slab_axis] / lz
        ### lammps fix electrode implementation
        ### cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and r
        if max_pos_first > max_pos_second:
            zprd_offset = -1 * -1 * normalized_positions
            efield = -1 * potential_drop / lz
        else:
            zprd_offset = -1 * normalized_positions
            efield = potential_drop / lz
        
     
        potential =  potential_drop * zprd_offset
        mask = (second_electrode == 1) | (first_electrode == 1)
        return potential[mask] , efield

    @torch.jit.export
    def Coulomb_Calculator(self,
                               electrode_mask : torch.Tensor,
                               positions : torch.Tensor,
                               box : torch.Tensor,
                               charges : torch.Tensor,
                               eta : torch.Tensor,
                               efield : torch.Tensor = None
                               )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Coulomb force for the system
        """
        # Set up neighbor list
        nblist = self.model_nblist
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()


        energy = self.model_coulomb(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        energy += self.model_gaussian( positions, box, pairs, ds, buffer_scales, {"charge": charges, "eta": eta})
        forces = -calc_grads(energy, positions)

        vector = torch.tensor([0, 0, 0])
        vector[self.slab_axis] = 1

        if efield is not None:
            forces += efield * charges.unsqueeze(1) * vector
            energy += torch.sum(efield * charges * positions[:, self.slab_axis])

        return energy, forces


def _conp(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",    
    ffield: Optional[bool] = False, 
    
)-> torch.Tensor:
    """
    Constrained Potential Method implementation
    An instantiation of QEq Module for electrode systems totally

    The electrode_mask not only contains information about which atoms are electrode atoms, 
    but also the potential(in volt) of the electrode atoms 
    """
    n_atoms = len(electrode_mask)
    box = box if box is not None else torch.zeros(3, 3)

    if "chi" not in params:
        params["chi"] = torch.zeros(n_atoms)
    if "hardness" not in params:
        params["hardness"] = torch.zeros(n_atoms)

    electrode_params = {k: v[electrode_mask != 0] for k, v in params.items()}
    electrode_positions = positions[electrode_mask != 0]
    charge = params["charge"]

    chi = module.coulomb_potential_add_chi(electrode_mask, positions, box, params["chi"], params["eta"], params["charge"])
    electrode_params["chi"] = chi[electrode_mask != 0]

    ##Apply the constant potential condition
    electrode_params["chi"] -= electrode_mask[electrode_mask != 0] 

    ##Apply the finite field condition
    if ffield:
        if module.slab_corr:
            raise KeyError("Slab correction and finite field cannot be used together")
        potential, efield = module.finite_field_add_chi(positions, box, electrode_mask)
        electrode_params["chi"] += potential
        module.ffield_flag = True
    
    # Neighbor list calculations
    nblist = module.model_nblist
    pairs = nblist(electrode_positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()


    constraint_matrix = constraint_matrix[:, electrode_mask != 0]
    q0 = charge[electrode_mask != 0]
    args = [
        module,
        q0,
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
        None,
        True,
        method
    ]
    energy,q_opt = pgrad_optimize(*args)
    charges = params["charge"].clone()
    charges[electrode_mask != 0] = q_opt
    charge_opt = torch.Tensor(charges)
    charge_opt.requires_grad_(True)

    return charge_opt

def conp(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    symm: bool = True,
    ffield: Optional[bool] = False,    
)-> torch.Tensor:
    """
    Lammps like implementation for User which is more convenient
    """
    #n_electrode_atoms = len(electrode_mask[electrode_mask != 0])
    n_atoms = len(electrode_mask)
    if symm:
        constraint_matrix = torch.ones([1, n_atoms])
        constraint_vals = torch.zeros(1)
    else:
        constraint_matrix = torch.zeros([0, n_atoms])
        constraint_vals = torch.zeros(0)        
    if ffield:
        if not symm:
            raise KeyError("Finite field only support charge neutral condition")
    
    return _conp(module, electrode_mask, positions, constraint_matrix, constraint_vals, box, params, method, ffield)

def _conq(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",    
    ffield: Optional[bool] = False, 
)-> torch.Tensor:
    """
    Constrained Potential Method implementation
    An instantiation of QEq Module for electrode systems totally

    The electrode_mask not only contains information about which atoms are electrode atoms, 
    but also the potential(in volt) of the electrode atoms 
    """
    n_atoms = len(electrode_mask)
    box = box if box is not None else torch.zeros(3, 3)

    if "chi" not in params:
        params["chi"] = torch.zeros(n_atoms)
    if "hardness" not in params:
        params["hardness"] = torch.zeros(n_atoms)

    electrode_params = {k: v[electrode_mask != 0] for k, v in params.items()}
    electrode_positions = positions[electrode_mask != 0]
    charge = params["charge"]

    chi = module.coulomb_potential_add_chi(electrode_mask, positions, box, params["chi"], params["eta"], params["charge"])
    electrode_params["chi"] = chi[electrode_mask != 0]

    ##Apply the finite field condition
    if ffield:
        if module.slab_corr:
            raise KeyError("Slab correction and finite field cannot be used together")
        
        raise KeyError("conq with finite field has not been implemented")
    
    # Neighbor list calculations
    nblist = module.model_nblist
    pairs = nblist(electrode_positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()


    constraint_matrix = constraint_matrix[:, electrode_mask != 0]
    
    q0 = charge[electrode_mask != 0].reshape(-1,1)

    args = [
        module,
        q0,
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
        None,
        True,
        method
    ]
    energy,q_opt = pgrad_optimize(*args)
    charges = params["charge"].clone()
    charges[electrode_mask != 0] = q_opt
    charge_opt = torch.Tensor(charges)
    charge_opt.requires_grad_(True)

    return charge_opt

def conq(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    charge_constraint_dict: Dict[int, torch.Tensor],
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    ffield: Optional[bool] = False,
)-> torch.Tensor:
    """
    Lammps like implementation for User which is more convenient
    which also can realize by conp
    charge_constraint_dict: Dict
        key is int data correspond to the electrode mask
        value is the constraint charge value
    """
    n_atoms = len(electrode_mask)
    tolerance = 1e-6
    if len(charge_constraint_dict) > 2:
        raise KeyError("Only one or two electrodes are supported Now")
    if len(charge_constraint_dict) == 1:
        constraint_matrix = torch.ones([1, n_atoms])
        constraint_vals = torch.tensor([list(charge_constraint_dict.values())[0]])
    else:
        key1 = list(charge_constraint_dict.keys())[0]
        key2 = list(charge_constraint_dict.keys())[1]

        row1 = torch.zeros([1, n_atoms])
        row1[0, torch.abs(electrode_mask - key1) < tolerance] = 1
        row2 = torch.zeros([1, n_atoms])
        row2[0, torch.abs(electrode_mask - key2) < tolerance] = 1

        constraint_matrix = torch.cat([row1, row2], dim=0)
        constraint_vals = torch.tensor([
            [list(charge_constraint_dict.values())[0]], 
            [list(charge_constraint_dict.values())[1]]
        ])
    if ffield:
        raise KeyError("conq with finite field has not been implemented")
    
    return _conq(module, electrode_mask, positions, constraint_matrix, constraint_vals, box, params, method, ffield)


def conq_aimd_data(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
) -> torch.Tensor:
    charge = params["charge"]
    constraint_vals = torch.sum(charge[electrode_mask == 0]) * -1
    electrode_positions = positions[electrode_mask == 1]
    constraint_matrix = torch.ones([1, len(electrode_mask)])
    
    return _conq(
        module=module,
        electrode_mask=electrode_mask,  
        positions=positions,
        constraint_matrix=constraint_matrix,
        constraint_vals=constraint_vals,
        box=box,
        params=params,
        method=method,
        ffield=False
    )