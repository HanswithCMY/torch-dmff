# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from ase import Atoms, io

n_atoms = 100
l_box = 10.0

charges = np.random.uniform(-1.0, 1.0, n_atoms)
charges -= np.mean(charges)
positions = np.random.uniform(0.0, l_box, (n_atoms, 3))

atoms = Atoms(
    n_atoms * "H", positions=positions, charges=charges, cell=[l_box, l_box, l_box]
)
io.write("coord.xyz", atoms)
io.write("system.data", atoms, format="lammps-data", atom_style="full")
