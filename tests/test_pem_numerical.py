# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
import os
from pathlib import Path
import numpy as np
import torch
from scipy import constants
from ase import io

from torch_admp.nblist import TorchNeighborList
from torch_admp.pem import PEMModule
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads, to_numpy_array
from torch_admp.pem import PEMGaussianDampingForceModule

from torch_admp.pem import conp
from torch_admp.pem import conq
from torch_admp.pem import conq_aimd_data
# Unit conversion factors
ENERGY_COEFF = (
    constants.physical_constants["joule-electron volt relationship"][0]
    * constants.kilo
    / constants.Avogadro
)
FORCE_COEFF = ENERGY_COEFF * 4.184  # kcal/(mol A) to eV/particle/A
POTENTIAL_COEFF = ENERGY_COEFF * 4.184  # kcal/(mol) to V/electron

# Test configuration
OUTPUT_CSV = False  # Set to True to output comparison CSV files

torch.set_default_dtype(torch.float64)

class TestPEMModule(unittest.TestCase):
    """Test PEM module functionality by comparing with LAMMPS reference results"""
    
    def setUp(self) -> None:
        """Set up basic test parameters"""
        # Model parameters
        self.rcut = 6.0
        self.ethresh = 1e-6
        self.data_root = Path(__file__).parent / "data/pem"
        

    
    def _write_csv(self, filename, data_dict):
        """Output CSV file to compare data (optional)"""
        if not OUTPUT_CSV:
            return
            
        import csv
        with open(filename, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(data_dict.keys()))
            writer.writeheader()
            rows = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
            writer.writerows(rows)
    
    def _load_pem_test_data(self, data_path):
        """Load PEM test data"""
        atoms = io.read(
            str(self.data_root / data_path),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        box = atoms.get_cell().array
        charges = atoms.get_initial_charges()
        
        # Update test data
        self.positions = torch.tensor(positions, requires_grad=True)
        self.box = torch.tensor(box)
        self.charges = torch.tensor(charges, requires_grad=True)
        self.ref_charge = torch.tensor(charges, requires_grad=True)
        
        n_atoms = charges.shape[0]
        self.n_atoms = n_atoms
        # Update neighbor list
        self.nblist = TorchNeighborList(cutoff=self.rcut)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()
        
        # Set electrode mask and parameters
        self.electrode_mask = torch.zeros(n_atoms)
        
        
        left_slab_mask = torch.zeros(n_atoms)
        left_slab_mask[:108] = 1
        right_slab_mask = torch.zeros(n_atoms)
        right_slab_mask[108:216] = 1
        left_slab_tuple = tuple(left_slab_mask.tolist())
        right_slab_tuple = tuple(right_slab_mask.tolist())
        self.electrode_atoms_dict = {left_slab_tuple: torch.tensor(0), right_slab_tuple: torch.tensor(1)}
        self.electrode_mask[:216] = 1
        
        # Set eta parameters (for electrode and non-electrode parts)
        self.eta = torch.cat([
            torch.full((108,), 0.4419417),  # Left electrode
            torch.full((108,), 0.4419417),  # Right electrode
            torch.full((len(charges)-216,), 0)  # Non-electrode part
        ])
        
        self.chi = torch.full_like(self.charges, 0)
        self.hardness = torch.full_like(self.charges, 0)

    
    def _get_params_dict(self):
        """Get parameters dictionary"""
        return {
            "charge": self.charges,
            "chi": self.chi,
            "hardness": self.hardness,
            "eta": self.eta
        }
    
    def _create_pem_module(self, slab_corr=False):
        """Create PEM module and Coulomb model"""
        module = PEMModule(
            rcut=self.rcut,
            ethresh=self.ethresh,
            kspace=True,
            rspace=True,
            slab_corr=slab_corr,
            slab_axis=2,
        )
        self.module_pgrad = PEMModule(
            self.rcut, self.ethresh, eps=1e-5, max_iter=100
        )

        return module
    
    def _calculate_forces(self, module):
        """Calculate force field forces"""

        energy, forces = module.Coulomb_Calculator(self.electrode_mask, 
                                                   self.positions, 
                                                   self.box,
                                                   self.charges, 
                                                   self.eta,
                                                   self.efield)

        return forces

    def _verify_results(self, forces, ref_forces, charges=None, ref_charges=None, 
                   tol_force=1e-4, tol_charge=1e-4, applied_potential=None):
        """
    Verify if results match reference values within tolerance.

    Parameters
    ----------
    forces : torch.Tensor or numpy.ndarray
        Calculated forces to verify
    ref_forces : numpy.ndarray
        Reference forces to compare against
    charges : torch.Tensor, optional
        Calculated charges to verify
    ref_charges : torch.Tensor, optional
        Reference charges to compare against
    tol_force : float, default=1e-3
        Tolerance for force comparisons
    tol_charge : float, default=1e-3
        Tolerance for charge comparisons
    applied_potential : list, optional
        Applied electrode potentials [V_left, V_right]
        """
        # Convert forces to numpy arrays for comparison
        forces_np = to_numpy_array(forces).reshape(-1, 3)
        ref_forces_np = ref_forces

        # Verify forces match within tolerance
        forces_match = np.allclose(
        forces_np,
        ref_forces_np,
        atol=tol_force,
        )

        # If forces don't match, find problematic indices
        if not forces_match:
            force_diff = np.abs(forces_np - ref_forces_np)
            error_indices = np.where(np.any(force_diff > tol_force, axis=1))[0]
        
            # Print detailed comparison information if needed
            print("\n=== FORCE COMPARISON ERROR DETAILS ===")
            print(f"Number of mismatched atoms: {len(error_indices)}/{forces_np.shape[0]}")
            print("Atom indices with mismatched forces:", error_indices)

            print("\nDetails of mismatched forces:")
            print("Index |    Component    |   Calculated   |   Reference    |   Difference   |")
            print("-" * 75)
        
            for idx in error_indices[:10]:  # Limit to first 10 mismatches to avoid overwhelming output
                for component, label in enumerate(['x', 'y', 'z']):
                    if force_diff[idx][component] > tol_force:
                        print(f"{idx:5d} | {label:^15} | {forces_np[idx][component]:13.6e} | "
                            f"{ref_forces_np[idx][component]:13.6e} | "
                            f"{force_diff[idx][component]:13.6e} |")
        else:
            # Print success message when forces match
            print("\n FORCE VERIFICATION SUCCESSFUL")
            print(f"All forces match within tolerance of {tol_force}")
        
            # Optionally print statistics about the forces
            force_diffs = np.abs(forces_np - ref_forces_np)
            max_diff = np.max(force_diffs)
            avg_diff = np.mean(force_diffs)
            print(f"Max difference: {max_diff:.6e}, Average difference: {avg_diff:.6e}")
    
        # Assert forces match within tolerance
        self.assertTrue(forces_match, "Forces do not match reference values")

        # Verify charges if provided
        if charges is not None and ref_charges is not None:
            charges_np = charges.detach().cpu().numpy()
            ref_charges_np = ref_charges.detach().cpu().numpy()
    
            charges_match = np.allclose(
            charges_np,
            ref_charges_np,
            atol=tol_charge,
            )
    
            # Output detailed charge comparison information if they don't match
            if not charges_match:
                charge_diff = np.abs(charges_np - ref_charges_np)
                error_indices = np.where(charge_diff > tol_charge)[0]
            
                print("\n=== CHARGE COMPARISON ERROR DETAILS ===")
                print(f"Number of mismatched charges: {len(error_indices)}/{charges_np.shape[0]}")
                print("Index |   Calculated   |   Reference    |   Difference   |")
                print("-" * 65)
            
                for idx in error_indices[:10]:  # Limit to first 10 mismatches
                    print(f"{idx:5d} | {charges_np[idx]:13.6e} | "
                      f"{ref_charges_np[idx]:13.6e} | {charge_diff[idx]:13.6e} |")
            else:
                # Print success message when charges match
                print("\n CHARGE VERIFICATION SUCCESSFUL")
                print(f"All charges match within tolerance of {tol_charge}")
            
                # Optionally print statistics about the charges
                charge_diffs = np.abs(charges_np - ref_charges_np)
                max_diff = np.max(charge_diffs)
                avg_diff = np.mean(charge_diffs)
                print(f"Max difference: {max_diff:.6e}, Average difference: {avg_diff:.6e}")
    
            # Assert charges match within tolerance
            self.assertTrue(charges_match, "Charges do not match reference values")
        
        # Print overall success message if everything matched
        if (charges is None or charges_match) and forces_match:
            print("\n ALL VERIFICATIONS PASSED SUCCESSFULLY \n")
    def _run_pem_test(self, data_subdir, lammpstrj_name, potential, test_name):
        """Generic method to run PEM tests"""
        print(f"Testing {test_name}")
        
        # Load test data
        data_path = f"{data_subdir}/after_pem.data"
        self._load_pem_test_data(data_path)
        
        # Create module and run simulation
        module = self._create_pem_module()
        params = self._get_params_dict()
        
        module.ffield_flag = True
        module.efield = -(potential[0]-potential[1])/self.box[2,2]
        self.efield = module.efield
        forces_ref_charge = self._calculate_forces(module)
        
        # Run constant potential simulation
        
        electrode_mask = torch.zeros(self.n_atoms)
        
        
        left_indices = torch.arange(self.n_atoms)[:108]  
        right_indices = torch.arange(self.n_atoms)[108:216]
        
        
        electrode_mask[left_indices] = potential[0]
        electrode_mask[right_indices] = potential[1]
        
        
        charges = conp(
            module=module, 
            electrode_mask=electrode_mask, 
            positions=self.positions, 
            box=self.box, 
            params={"charge": self.charges, "chi": self.chi, "hardness": self.hardness, "eta": self.eta},
            ffield=True
        )
        
        
        self.charges = charges
        
        # Output charge comparison
        self._write_csv(f"charge_comparison_{test_name}.csv", {
            "charge": charges.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })

        forces_calc_charge = self._calculate_forces(module)

        # Read LAMMPS reference forces
        atoms = io.read(str(self.data_root / f"{data_subdir}/{lammpstrj_name}"))
        ref_force = atoms.get_forces() * FORCE_COEFF
        
        # Extract z-direction forces for comparison
        forces_ref_z = to_numpy_array(forces_ref_charge).reshape(-1, 3)[:, 2]
        forces_calc_z = to_numpy_array(forces_calc_charge).reshape(-1, 3)[:, 2]
        ref_forces_z = ref_force[:, 2]
        
        # Output force comparison
        self._write_csv(f"forces_comparison_{test_name}.csv", {
            "forces_ref_z": forces_ref_z,
            "forces_calc_z": forces_calc_z, 
            "ref_forces_z": ref_forces_z
        })
        
        # Verify results with reference charge forces 
        self._verify_results(forces_calc_charge, ref_force, charges, self.ref_charge, applied_potential=potential)



    def test_far(self):
        """Test constant potential simulation with high potential difference"""
        self._run_pem_test(
            data_subdir="conp/far",
            lammpstrj_name="conp.lammpstrj",
            potential=[201, 1],
            test_name="conp_far"
        )
    
    def test_near(self):
        """Test constant potential simulation with electrodes in close proximity"""
        self._run_pem_test(
            data_subdir="conp/near",
            lammpstrj_name="conp.lammpstrj",
            potential=[201, 1],
            test_name="conp_near"
        )
    def test_conq(self):
        """Test constant charge simulation"""
        print("Testing conq")
        # Load test data
        data_path = "conq/after_pem.data"
        self._load_pem_test_data(data_path)
        
        # Create module and run simulation
        module = self._create_pem_module()
        
        params = self._get_params_dict()
        
        
        left_slab_mask = torch.zeros(self.n_atoms)
        left_slab_mask[:108] = 1
        right_slab_mask = torch.zeros(self.n_atoms)
        right_slab_mask[108:216] = 1
        
        
        electrode_mask = torch.zeros(self.n_atoms)
        electrode_mask[:108] = 1  
        electrode_mask[108:216] = 2  
        
        
        charge_constraint_dict = {1: torch.tensor(-5.0), 2: torch.tensor(-5.0)}
        

        
        self.efield = 0
        forces_ref_charge = self._calculate_forces(module)

        
        charges = conq(
            module=module,
            electrode_mask=electrode_mask,
            positions=self.positions,
            charge_constraint_dict=charge_constraint_dict,
            box=self.box,
            params=params
        )
        
        
        self.charges = charges

        
        self._write_csv("charge_comparison_conq.csv", {
            "charge": charges.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })
        
        # compare forces

        forces_calc_charge = self._calculate_forces(module)
        
    
        atoms = io.read(str(self.data_root / "conq/system_pem.lammpstrj"))
        ref_force = atoms.get_forces() * FORCE_COEFF
        
        
        forces_ref_z = to_numpy_array(forces_ref_charge).reshape(-1, 3)[:, 2]
        forces_calc_z = to_numpy_array(forces_calc_charge).reshape(-1, 3)[:, 2]
        ref_forces_z = ref_force[:, 2]
        
        
        self._write_csv("forces_comparison_conq.csv", {
            "forces_ref_z": forces_ref_z,
            "forces_calc_z": forces_calc_z,
            "ref_forces_z": ref_forces_z
        })
        
        
        self._verify_results(forces_calc_charge, ref_force, charges, self.ref_charge)

    def test_conq_jit(self):
        """Test constant charge simulation with JIT compilation"""
        print("Testing conq with JIT")
       
        data_path = "conq/jit/after_pem.data"
        self._load_pem_test_data(data_path)
    
        # Create module and JIT version
        module = self._create_pem_module()
        jit_module = torch.jit.script(module)
        params = self._get_params_dict()
    
        
        electrode_mask = torch.zeros(self.n_atoms)
        electrode_mask[:108] = 1  
        electrode_mask[108:216] = 1 
        self.efield = 0

        forces_ref_charge = self._calculate_forces(module)
    
        
        
    
        # non-JIT version
        params["chi"] = torch.zeros_like(params["chi"])
        charges_ref = conq_aimd_data(
            module=module,
            electrode_mask=electrode_mask, 
            positions=self.positions, 
            box=self.box, 
            params=params
        )
    
        # jit module
        charges_jit = conq_aimd_data(
            module=jit_module,
            electrode_mask=electrode_mask,
            positions=self.positions, 
            box=self.box, 
            params=params
        )
    
        # compare JIT and non-JIT versions results about charges
        torch.testing.assert_close(charges_ref, charges_jit, rtol=1e-5, atol=1e-5)
        print("JIT compilation test passed: JIT and non-JIT versions produce identical results")
    
        
        module.charge = charges_jit
    
        
        self._write_csv("charge_comparison_conq_jit.csv", {
            "charge_jit": charges_jit.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })
    
        # compute forces
        forces_calc_charge = self._calculate_forces(module)
    
        
        atoms = io.read(str(self.data_root / "conq/jit/system.lammpstrj"))
        ref_force = atoms.get_forces() * FORCE_COEFF
    
        
        forces_ref_z = to_numpy_array(forces_ref_charge).reshape(-1, 3)[:, 2]
        forces_calc_z = to_numpy_array(forces_calc_charge).reshape(-1, 3)[:, 2]
        ref_forces_z = ref_force[:, 2]
    
        
        self._write_csv("forces_comparison_conq_jit.csv", {
            "forces_ref_z": forces_ref_z,
            "forces_calc_z": forces_calc_z,
            "ref_forces_z": ref_forces_z
        })
    
        
        self._verify_results(forces_calc_charge, ref_force, charges_jit, self.ref_charge)

    def test_conp_jit(self):
        """Test constant potential simulation with JIT compilation"""
        print("Testing conp with JIT")
        # Load test data
        data_path = "conp/near/after_pem.data"
        self._load_pem_test_data(data_path)
    
        # Create module and JIT version
        module = self._create_pem_module()
        jit_module = torch.jit.script(module)
        params = self._get_params_dict()
    

        left_slab_mask = torch.zeros(self.n_atoms)
        left_slab_mask[:108] = 1
        right_slab_mask = torch.zeros(self.n_atoms)
        right_slab_mask[108:216] = 1

        electrode_mask = torch.zeros(self.n_atoms)
        electrode_mask[:108] = 201
        electrode_mask[108:216] = 1
        potential = [201, 1]
        module.efield = -(potential[0]-potential[1])/self.box[2,2]
        self.efield = module.efield
    
        forces_ref_charge = self._calculate_forces(module)

        charges_ref = conp(
            module=module,
            electrode_mask=electrode_mask,
            positions=self.positions,
            box=self.box,
            params=params,
            ffield=True
        )

        # jit module 
        charges_jit = conp(
            module=jit_module,
            electrode_mask=electrode_mask,
            positions=self.positions,
            box=self.box,
            params=params,
            ffield=True
        )

        # compare JIT and non-JIT versions results about charges
        torch.testing.assert_close(charges_ref, charges_jit, rtol=1e-5, atol=1e-5)
        print("JIT compilation test passed: JIT and non-JIT versions produce identical results")


        module.charge = charges_jit

    
        self._write_csv("charge_comparison_conp_jit.csv", {
            "charge_jit": charges_jit.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })

        # Compute forces

        forces_calc_charge = self._calculate_forces(module)


        atoms = io.read(str(self.data_root / "conp/near/conp.lammpstrj"))
        ref_force = atoms.get_forces() * FORCE_COEFF


        forces_ref_z = to_numpy_array(forces_ref_charge).reshape(-1, 3)[:, 2]
        forces_calc_z = to_numpy_array(forces_calc_charge).reshape(-1, 3)[:, 2]

        ref_forces_z = ref_force[:, 2]

        self._write_csv("forces_comparison_conp_jit.csv", {
            "forces_ref_z": forces_ref_z,
            "forces_calc_z": forces_calc_z,
            "ref_forces_z": ref_forces_z
        })

        self._verify_results(forces_calc_charge, ref_force, charges_jit, self.ref_charge)




if __name__ == "__main__":

    unittest.main()