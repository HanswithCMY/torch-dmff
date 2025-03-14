# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Dict, List, Tuple, Union
import torch



from torch_dmff.base_force import BaseForceModule
from torch_dmff.utils import calc_grads
from torch_dmff.nblist import TorchNeighborList
from torch_dmff.pme import CoulombForceModule
from torch_dmff.qeq import GaussianDampingForceModule
from torch_dmff.qeq import pgrad_optimize
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
        self.ffield_flag = False
        self.charge_opt = None

        if units_dict != None:
            self.units_dict = units_dict

    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Charge equilibrium (QEq) model

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
                "charge": t_charges, # (optional) initial guess for atomic charges,
                "chi": t_chi, # eletronegativity in energy / charge unit
                "hardness": t_hardness, # atomic hardness in energy / charge^2 unit
                "eta": t_eta, # Gaussian width in length unit
                "electrode_mask": electrode_mask, # mask for electrode atoms
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        energy = torch.zeros(1, device=positions.device)
        for model in self.submodels.values():
            energy = energy + model(positions, box, pairs, ds, buffer_scales, params)
        return energy


    @torch.jit.ignore 
    def input_data_loader(
        self,
        electrode_mask: torch.Tensor,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        electrode_atoms_dict: Dict[Tuple,torch.Tensor],
        params: Dict[str, torch.Tensor],
        ) -> None:
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
            key is a mask for different electrode atoms
            value is a constraint charge value
            
        params : Dict[str, torch.Tensor]
            parameters for optimization

        """
        if len(electrode_atoms_dict) > 2:
            raise KeyError("Only one or two electrodes are supported Now")
        self.n_atoms = len(electrode_mask)
        self.electrode_atoms_dict = electrode_atoms_dict
        self.first_electrode = torch.tensor(list(electrode_atoms_dict.keys())[0])
        self.first_cons = electrode_atoms_dict[list(electrode_atoms_dict.keys())[0]]


        if len(electrode_atoms_dict) == 2:
            self.second_electrode = torch.tensor(list(electrode_atoms_dict.keys())[1])
            self.second_cons = electrode_atoms_dict[list(electrode_atoms_dict.keys())[1]]
        else:
            self.second_cons = None
        self.box = box
        self.positions = positions
        self.electrode_mask = electrode_mask
        self.params = params
        
        self.charge = params["charge"]

        params["chi"] = self.coulomb_potential_add_chi(electrode_mask, positions, box, params["chi"],  params["eta"],params["charge"])
        self.chi = params["chi"]

        #only consider the electrode atoms  
        self.electrode_params = {}
        mask = electrode_mask == 1
        for k, v in params.items():
            self.electrode_params[k] = v[mask]      
        self.electrode_positions = positions[electrode_mask == 1]
        self.nblist = self.models["nblist"]
        self.pairs = self.nblist(self.electrode_positions, box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()


    @torch.jit.ignore
    def conp(self,potential: torch.tensor,ffield: bool=False,method: str="pgrad",symm: bool=True,
             constraint: Optional[bool]=None) -> torch.Tensor:
        #Add constant potential condition in chi 
        self.conp_flag = True


        potential_term = (self.first_electrode * potential[0] 
                         + self.second_electrode * potential[1])
        potential_term = potential_term[self.electrode_mask == 1]
        self.electrode_params["chi"] -= potential_term 
            
        if ffield :
            if self.slab_corr :
                raise KeyError("Slab correction and finite field cannot be used together")
            potential_drop = potential[0] - potential[1]
            self.electrode_params["chi"] = self.finite_field_add_chi(potential_drop)
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
            elec_num = len(self.electrode_mask[self.electrode_mask == 1])
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
    @torch.jit.ignore
    def conq(self,ffield: bool=False,method: str="pgrad") -> torch.Tensor:
        self.conq_flag = True

        fi_cons_charge = self.first_cons
        se_cons_charge = self.second_cons
        if fi_cons_charge is None and se_cons_charge is None:
            raise KeyError("conq requires both first or second electrode charges constraints")
        
        if se_cons_charge is None:
            n = len(self.electrode_mask[self.first_electrode==1])
            constraint_matrix = torch.ones([1, n])
            constraint_vals = torch.tensor([fi_cons_charge])
        else:
            row1 = self.first_electrode[self.electrode_mask == 1].reshape(1, -1)
            row2 = self.second_electrode[self.electrode_mask == 1].reshape(1, -1)
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
    @torch.jit.ignore
    def finite_field_add_chi(self,potential_drop:float) -> torch.Tensor:
        """
        Compute the correction term for the finite field

        potential drop need to be the potentials of the first electrode minus the second
        """        
        positions =  self.positions
        box = self.box
        electrode_chi = self.electrode_params["chi"] 
        slab_axis = self.slab_axis

        first_electrode = self.first_electrode
        second_electrode = self.second_electrode

        
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
            self.efield = -1 * potential_drop / lz
        else:
            zprd_offset = -1 * normalized_positions
            self.efield = potential_drop / lz
        
     
        potential =  potential_drop * zprd_offset
        mask = (second_electrode == 1) | (first_electrode == 1)
        return electrode_chi + potential[mask] 
    def Coulomb_Calculator(self)-> Tuple[torch.Tensor, torch.Tensor]:
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

        energy = self.models["coulomb"](self.positions, self.box, pairs, ds, buffer_scales, {"charge": charge})
        energy += self.models["gaussian"]( self.positions, self.box, pairs, ds, buffer_scales, {"charge": charge, "eta": self.params["eta"]})
        forces = -calc_grads(energy, self.positions)
        if self.ffield_flag:
            forces += self.efield * charge.unsqueeze(1) * torch.tensor([0, 0, 1])

        return energy, forces


def input_data_loader_and_conq(
    module: PEMModule,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
) -> torch.Tensor:
    
    n_atoms = len(electrode_mask)
    box = box if box is not None else torch.zeros(3, 3)

    if "chi" not in params:
        params["chi"] = torch.zeros(n_atoms)
    if "hardness" not in params:
        params["hardness"] = torch.zeros(n_atoms)

    electrode_params = {k: v[electrode_mask == 1] for k, v in params.items()}
    electrode_positions = positions[electrode_mask == 1]
    charge = params["charge"]


    chi = module.coulomb_potential_add_chi(electrode_mask, positions, box, params["chi"], params["eta"], params["charge"])
    electrode_params["chi"] = chi[electrode_mask == 1]

    # Neighbor list calculations
    nblist = module.model_nblist
    pairs = nblist(electrode_positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    constraint_vals = torch.sum(charge[electrode_mask == 0]) * -1
    
    constraint_matrix = torch.ones([1, len(electrode_positions)])
    print(constraint_matrix)
    # Preparing args
    q0 = charge[electrode_mask == 1]
    print(q0)
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
    charges[electrode_mask == 1] = q_opt
    charge_opt = torch.Tensor(charges)
    charge_opt.requires_grad_(True)

    return charge_opt