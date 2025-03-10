# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
import os
from pathlib import Path
import numpy as np
import torch
from scipy import constants
from ase import io

from torch_dmff.nblist import TorchNeighborList
from torch_dmff.pem import PEMModule
from torch_dmff.pme import CoulombForceModule
from torch_dmff.utils import calc_grads, to_numpy_array
from torch_dmff.pem import PEMGaussianDampingForceModule

# Unit conversion factors
ENERGY_COEFF = (
    constants.physical_constants["joule-electron volt relationship"][0]
    * constants.kilo
    / constants.Avogadro
)
FORCE_COEFF = ENERGY_COEFF * 4.184  # kcal/(mol A) to eV/particle/A
POTENTIAL_COEFF = ENERGY_COEFF * 4.184  # kcal/(mol) to V/electron

# Test configuration
OUTPUT_CSV = True  # Set to True to output comparison CSV files

torch.set_default_dtype(torch.float64)

class TestPEMModule(unittest.TestCase):
    """Test PEM module functionality by comparing with LAMMPS reference results"""
    
    def setUp(self) -> None:
        """Set up basic test parameters"""
        # Model parameters
        self.rcut = 6.0
        self.ethresh = 1e-6
        self.data_root = Path(__file__).parent / "data/pem"
        
        # Load initial test data
        atoms = io.read(
            str(self.data_root / "after_pem.data"),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        box = atoms.get_cell().array
        charges = atoms.get_initial_charges()

        # Convert to PyTorch tensors
        self.positions = torch.tensor(positions, requires_grad=True)
        self.box = torch.tensor(box)
        self.charges = torch.tensor(charges)

        # Set up neighbor list
        self.nblist = TorchNeighborList(cutoff=self.rcut)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()
        
        # Initialize parameters
        self.eta = torch.full_like(self.charges, 0.4419417)
        self.chi = torch.full_like(self.charges, 0)
        self.hardness = torch.full_like(self.charges, 0)
    
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
        self.charges = torch.tensor(charges)
        self.ref_charge = torch.tensor(charges)
        
        n_atoms = charges.shape[0]
        
        # Update neighbor list
        self.nblist = TorchNeighborList(cutoff=self.rcut)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()
        
        # Set electrode mask and parameters
        self.electrode_mask = np.zeros(n_atoms)
        self.electrode_mask[:216] = 1
        self.num_electrode_atoms_dict = {"left_slab": 108, "right_slab": 108}
        
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
        
        # Replace Coulomb model to ensure consistent settings
        module.models["coulomb"] = CoulombForceModule(
            rcut=self.rcut,
            ethresh=self.ethresh,
            kspace=True,
            rspace=True,
            slab_corr=slab_corr,
            slab_axis=2,
            kappa=0.45131303,
            spacing=1,
        )
        
        return module
    
    def _calculate_forces(self, charges, applied_potential=None):
        """Calculate force field forces"""
        # Create Coulomb model
        coul_model = self._create_pem_module(slab_corr=False).models["coulomb"]
        # Calculate energy
        energy = coul_model(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": charges},
        )
        
        # Add Gaussian damping term
        
        model = PEMGaussianDampingForceModule()
        energy += model(
            self.electrode_mask,
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": charges, "eta": self.eta}
        )
        
        # Calculate forces
        forces = -calc_grads(energy, self.positions)
        
        # Add electric field force (if potential is provided)
        if applied_potential is not None:
            efield = -(applied_potential[0] - applied_potential[1]) / self.box[2, 2]
            forces += efield * charges.unsqueeze(1) * torch.tensor([0, 0, 1], device=forces.device)
        
        return forces
    '''
    def _verify_results(self, forces, ref_forces, charges=None, ref_charges=None, tol_force=1e-4, tol_charge=1e-3):
        """Verify if results match reference values within tolerance"""
        # Verify forces
        forces_match = np.allclose(
            to_numpy_array(forces).reshape(-1, 3),
            ref_forces,
            atol=tol_force,
        )
        self.assertTrue(forces_match, "Forces do not match reference values")
        
        # Verify charges (if provided)
        if charges is not None and ref_charges is not None:
            charges_match = np.allclose(
                charges.detach().cpu().numpy(),
                ref_charges.detach().cpu().numpy(),
                atol=tol_charge,
            )
            self.assertTrue(charges_match, "Charges do not match reference values")
            
            # If charges don't match, output detailed information
            if not charges_match:
                diff = np.abs(charges.detach().cpu().numpy() - ref_charges.detach().cpu().numpy())
                error_indices = np.where(diff > tol_charge)
                print("Differences found at indices:", error_indices)
                print("Differences:", diff[error_indices])
                print("Charges:", charges.detach().cpu().numpy()[error_indices])
                print("Reference Charges:", ref_charges.detach().cpu().numpy()[error_indices])
    '''
    def _verify_results(self, forces, ref_forces, charges=None, ref_charges=None, tol_force=1e-3, tol_charge=1e-3,applied_potential=None):
        """
        Verify if results match reference values within tolerance.

    Parameters:
    -----------
    forces : torch.Tensor or numpy.ndarray
        Calculated forces to verify
    ref_forces : numpy.ndarray
        Reference forces to compare against
    charges : torch.Tensor, optional
        Calculated charges to verify
    ref_charges : torch.Tensor, optional
        Reference charges to compare against
    tol_force : float, default=1e-4
        Tolerance for force comparisons
    tol_charge : float, default=1e-3
        Tolerance for charge comparisons
        """
        if applied_potential == [4,0]:
            tol_force = 1e-6
        # Convert forces to numpy arrays for comparison
        forces_np = to_numpy_array(forces).reshape(-1, 3)
        ref_forces_np = ref_forces

        # Verify forces
        forces_match = np.allclose(
        forces_np,
        ref_forces_np,
        atol=tol_force,
        )
    
        # If forces don't match, output detailed information BEFORE assertion
        if not forces_match:
            # Calculate absolute differences
            force_diff = np.abs(forces_np - ref_forces_np)
        
            # Find indices where differences exceed tolerance
            error_indices = np.where(np.any(force_diff > tol_force, axis=1))[0]
        
            # Print detailed comparison information
            print("\n=== FORCE COMPARISON ERROR DETAILS ===")
            print(f"Number of mismatched atoms: {len(error_indices)}/{forces_np.shape[0]}")
            print("Atom indices with mismatched forces:", error_indices)
        
            # Create a formatted table for easier comparison
            print("\nDetails of mismatched forces:")
            print("Index |    Component    |   Calculated   |   Reference    |   Difference   |")
            print("-" * 75)
        
            for idx in error_indices:
                for component, label in enumerate(['x', 'y', 'z']):
                    if force_diff[idx][component] > tol_force:
                        print(f"{idx:5d} | {label:^15} | {forces_np[idx][component]:13.6e} | {ref_forces_np[idx][component]:13.6e} | {force_diff[idx][component]:13.6e} |")
    
        # Assert AFTER printing detailed information
        self.assertTrue(forces_match, "Forces do not match reference values")

        # Verify charges (if provided)
        if charges is not None and ref_charges is not None:
            charges_np = charges.detach().cpu().numpy()
            ref_charges_np = ref_charges.detach().cpu().numpy()
        
            charges_match = np.allclose(
            charges_np,
            ref_charges_np,
            atol=tol_charge,
            )
        
            # If charges don't match, output detailed information BEFORE assertion
            if not charges_match:
                # Calculate absolute differences
                charge_diff = np.abs(charges_np - ref_charges_np)
            
                # Find indices where differences exceed tolerance
                error_indices = np.where(charge_diff > tol_charge)[0]
            
                # Print detailed comparison information
                print("\n=== CHARGE COMPARISON ERROR DETAILS ===")
                print(f"Number of mismatched charges: {len(error_indices)}/{len(charges_np)}")
                print("Indices with mismatched charges:", error_indices)
            
                # Create a formatted table for easier comparison
                print("\nDetails of mismatched charges:")
                print("Index |   Calculated   |   Reference    |   Difference   |")
                print("-" * 65)
            
                for idx in error_indices:
                    print(f"{idx:5d} | {charges_np[idx]:13.6e} | {ref_charges_np[idx]:13.6e} | {charge_diff[idx]:13.6e} |")
        
            # Assert AFTER printing detailed information
            self.assertTrue(charges_match, "Charges do not match reference values")


    def test_numerical(self):
        """Test numerical calculation consistency with LAMMPS reference values"""
        self._run_pem_test(
        data_subdir=".",  # 使用与test_conp相同的数据目录
        lammpstrj_name="conp.lammpstrj",
        potential=[4, 0],
        test_name="numerical"  # 使用不同的测试名称以区分
        )
    def _run_pem_test(self, data_subdir, lammpstrj_name, potential, test_name):
        """Generic method to run PEM tests"""
        print(f"Testing {test_name}")
        
        # Load test data
        data_path = f"{data_subdir}/after_pem.data"
        self._load_pem_test_data(data_path)
        
        # Create module and run simulation
        module = self._create_pem_module()
        params = self._get_params_dict()
        
        # Run constant potential simulation
        energy, forces, charges = module.conp(
            n_atoms=self.charges.shape[0],
            electrode_mask=self.electrode_mask,
            positions=self.positions,
            box=self.box,
            num_electrode_atoms_dict=self.num_electrode_atoms_dict,
            params=params,
            potential=np.array(potential),
            ffield=True,
            method="mat_inv", 
            symm=True,
        )
        
        # Output charge comparison
        self._write_csv(f"charge_comparison_{test_name}.csv", {
            "charge": charges.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })

        #rounded_ref_charge = torch.round(self.ref_charge * 1e12) / 1e12
        #rounded_charges = torch.round(charges * 1e12) / 1e12
        #print(f"rounded_charges: {rounded_charges}")
        #print(f"rounded_ref_charge: {rounded_ref_charge}")
        #print(f"rounded_charge-rounded_ref_charge: {rounded_charges - rounded_ref_charge}")
        print(f"before definite charges: {charges}")
        #ref_charge = torch.tensor(self.ref_charge, requires_grad=True)
        charges = torch.tensor(charges, requires_grad=True)
        print(f"after definite charges: {charges}")
        # Calculate forces with both reference charges and calculated charges
        #forces_ref_charge = self._calculate_forces(self.ref_charge, potential)
        forces_ref_charge = self._calculate_forces(self.ref_charge, potential)
        forces_calc_charge = self._calculate_forces(charges, potential)
        #forces_ref_charge = self._calculate_forces(rounded_ref_charge, potential)
        #forces_calc_charge = self._calculate_forces(rounded_charges, potential)
        print(f"charges: {charges}")
        print(f"ref_charge: {self.ref_charge}")
        print(f"charge-ref_charge: {charges - self.ref_charge}")
        print(f"forces_ref_charge: {forces_ref_charge}")
        print(f"forces_calc_charge: {forces_calc_charge}")
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
        
        # Verify results with reference charge forces (original behavior)
        self._verify_results(forces_ref_charge, ref_force, charges, self.ref_charge,applied_potential=potential)
        
        # Also print comparison of calculated charge forces
        forces_calc_match = np.allclose(
            to_numpy_array(forces_calc_charge).reshape(-1, 3),
            ref_force,
            atol=1e-4,
        )
        print(f"Forces calculated with optimized charges match reference: {forces_calc_match}")
    
    def test_conp(self):
        """Test basic constant potential simulation"""
        self._run_pem_test(
            data_subdir="conp",
            lammpstrj_name="system.lammpstrj",
            potential=[4, 0],
            test_name="conp"
        )
    
    def test_far(self):
        """Test constant potential simulation with high potential difference"""
        self._run_pem_test(
            data_subdir="conp/far",
            lammpstrj_name="conp.lammpstrj",
            potential=[200, 0],
            test_name="conp_far"
        )
    
    def test_near(self):
        """Test constant potential simulation with electrodes in close proximity"""
        self._run_pem_test(
            data_subdir="conp/near",
            lammpstrj_name="conp.lammpstrj",
            potential=[200, 0],
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
        
        # Run constant charge simulation
        energy, forces, charges = module.conq(
            n_atoms=self.charges.shape[0],
            electrode_mask=self.electrode_mask,
            positions=self.positions,
            box=self.box,
            num_electrode_atoms_dict=self.num_electrode_atoms_dict,
            params=params,
            potential=None,  # No potential needed for constant charge mode
            method="mat_inv", 
            symm=False,
            fi_cons_charge=-5.0,
            se_cons_charge=-5.0,
        )
        
        # Output charge comparison
        self._write_csv("charge_comparison_conq.csv", {
            "charge": charges.detach().cpu().numpy(),
            "ref_charge": self.ref_charge.detach().cpu().numpy()
        })
        rounded_ref_charge = torch.round(self.ref_charge * 1e12) / 1e12
        rounded_charges = torch.round(charges * 1e12) / 1e12
        # Calculate forces with both reference charges and calculated charges
        #forces_ref_charge = self._calculate_forces(self.ref_charge)
        #forces_calc_charge = self._calculate_forces(charges)
        forces_ref_charge = self._calculate_forces(rounded_ref_charge)
        forces_calc_charge = self._calculate_forces(rounded_charges)
        # Read LAMMPS reference forces
        atoms = io.read(str(self.data_root / "conq/system.lammpstrj"))
        ref_force = atoms.get_forces() * FORCE_COEFF
        
        # Extract z-direction forces for comparison
        forces_ref_z = to_numpy_array(forces_ref_charge).reshape(-1, 3)[:, 2]
        forces_calc_z = to_numpy_array(forces_calc_charge).reshape(-1, 3)[:, 2]
        ref_forces_z = ref_force[:, 2]
        
        # Output force comparison
        self._write_csv("forces_comparison_conq.csv", {
            "forces_ref_z": forces_ref_z,
            "forces_calc_z": forces_calc_z,
            "ref_forces_z": ref_forces_z
        })
        
        # Verify results with reference charge forces (original behavior)
        self._verify_results(forces_ref_charge, ref_force, charges, self.ref_charge)
        
        # Also print comparison of calculated charge forces
        forces_calc_match = np.allclose(
            to_numpy_array(forces_calc_charge).reshape(-1, 3),
            ref_force,
            atol=1e-4,
        )
        print(f"Forces calculated with optimized charges match reference: {forces_calc_match}")


if __name__ == "__main__":

    unittest.main()