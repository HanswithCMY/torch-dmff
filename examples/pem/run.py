import jax.numpy as jnp
import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from scipy import constants

from torch_dmff.nblist import TorchNeighborList
from torch_dmff.qeq import QEqForceModule
from torch_dmff.utils import calc_grads

from torch_dmff.pem import PEMModule
from torch_dmff.pem import XMLDataLoadder

#from torch_dmff.pem import run_poisson_1D

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    rcut = 6.0
    ethresh = 1e-5
    module = PEMModule(rcut=rcut,ethresh=ethresh,slab_corr=False)
    data_dict = XMLDataLoadder.load_xml(file_name="pem.xml",pdb_name="pem.pdb",slab_res_list=["left_slab","right_slab"],electroyte_res_list=["water"],
                         left_electrode="left_slab",right_electrode="right_slab")
    print(data_dict["charge"])
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
        requires_grad=True,
    )
    n_atoms = data_dict["n_atoms"]
    electrode_mask = data_dict["electrode_mask"]
    num_electrode_atoms_dict = data_dict["num_electrode_atoms_dict"]
    parms  = {}
    parms["charge"] = charges
    parms ["chi"] = chi
    parms["hardness"] = hardness
    parms["eta"] = eta
    print(charges)
    
    ###conp mat_inv
    energy, forces, charges = module.conp(
        n_atoms,
        electrode_mask,
        positions,
        box,
        num_electrode_atoms_dict,
        parms,
        potential=np.array([0,4]),
        method="mat_inv",
        ffield=True,
    )
    ###
    
    '''
    ###conq mat_inv
    energy, forces, charges = module.conq(
        n_atoms,
        electrode_mask,
        positions,
        box,
        num_electrode_atoms_dict,
        parms,
        potential=np.array([0,4]),
        method="mat_inv",
        fi_cons_charge=5.0,
        se_cons_charge=-5.0,

    )
    '''
    '''
    ###conp mat_inv ffield
    ffield = True
    potential = np.array([0,4])
    energy, forces, charges = module.conp(
        n_atoms,
        electrode_mask,
        positions,
        box,
        num_electrode_atoms_dict,
        parms,
        potential=potential,
        ffield=ffield,
        symm=True,
        method="pgrad",
    )
    '''
    ###result
    print(f"charges value:{charges}")
    print(forces)
    print(energy)
    print("done")
    
'''    ###poisson 1D
    ffield = False
    potential = np.array([0,4])

    data_dict["charge"] = charges.detach().cpu().numpy()
    data_dict["position"] = positions.detach().cpu().numpy()
    data_dict["box"] = box.detach().cpu().numpy()
    print(data_dict["box"])
    int1, phi, r, charge_dens_profile = run_poisson_1D(data_dict=data_dict, ffield=ffield,
                    APPLIED_F=(potential[0]-potential[1])/data_dict["box"][2][2])
   
    print(f"int1: {int1}")
    print(f"phi: {phi}")
    print(f"r: {r}")
    print(f"charge_dens_profile: {charge_dens_profile}")

    with open ("1d_potential_with_charge.csv","w") as f:
        f.write("coordinate,charge_density,phi\n")
        for i in range(len(phi)):
            f.write(f"{r[i]},{charge_dens_profile[i]},{phi[i]}\n")'''
