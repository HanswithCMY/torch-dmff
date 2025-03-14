# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional

import torch

from torch_admp.spatial import pbc_shift


# todo: fix bug or deprecated this class (for crossing-boundary images)
class TorchNeighborList(torch.nn.Module):
    """
    Adapt from below code for jitable:
        https://github.com/Yangxinsix/curator/tree/master
        curator.data.TorchNeighborList
    """

    def __init__(
        self,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        disp_mat = torch.cartesian_prod(
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            torch.arange(-1, 2),
        )
        self.register_buffer("disp_mat", disp_mat, persistent=False)

        self.pairs = torch.jit.annotate(torch.Tensor, torch.empty(1, dtype=torch.long))
        self.buffer_scales = torch.jit.annotate(
            torch.Tensor, torch.empty(1, dtype=torch.long)
        )
        self.ds = torch.jit.annotate(torch.Tensor, torch.empty(1, dtype=torch.float64))
        # self.pairs: torch.Tensor = None
        # self.buffer_scales: torch.Tensor = None
        # self.ds: torch.Tensor = None

    def forward(
        self, positions: torch.Tensor, box: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if box is None:
            pairs = self.forward_obc(positions)
            pbc_flag = False
        else:
            pairs = self.forward_pbc(positions, box)
            pbc_flag = True

        self.pairs = pairs
        self.buffer_scales = self.pairs_buffer_scales(pairs)
        self.ds = self.pairs_ds(positions, pairs, box, pbc_flag)
        return pairs

    def forward_pbc(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        # calculate padding size. It is useful for all kinds of cells
        wrapped_pos = self.wrap_positions(positions, box)
        norm_a = torch.linalg.cross(box[1], box[2]).norm()
        norm_b = torch.linalg.cross(box[2], box[0]).norm()
        norm_c = torch.linalg.cross(box[0], box[1]).norm()
        volume = torch.sum(box[0] * torch.linalg.cross(box[1], box[2]))

        # get padding size and padding matrix to generate padded atoms. Use minimal image convention
        padding_a = torch.ceil(self.cutoff * norm_a / volume).long()
        padding_b = torch.ceil(self.cutoff * norm_b / volume).long()
        padding_c = torch.ceil(self.cutoff * norm_c / volume).long()

        padding_mat = torch.cartesian_prod(
            torch.arange(-padding_a, padding_a + 1, device=padding_a.device),
            torch.arange(-padding_b, padding_b + 1, device=padding_a.device),
            torch.arange(-padding_c, padding_c + 1, device=padding_a.device),
        ).to(box.dtype)
        padding_size = (2 * padding_a + 1) * (2 * padding_b + 1) * (2 * padding_c + 1)

        # padding, calculating box numbers and shapes
        padded_pos = (wrapped_pos.unsqueeze(1) + padding_mat @ box).view(-1, 3)
        padded_cpos = torch.floor(padded_pos / self.cutoff).long()
        corner = torch.min(padded_cpos, dim=0)[0]  # the box at the corner
        padded_cpos -= corner
        c_pos_shap = torch.max(padded_cpos, dim=0)[0] + 1  # c_pos starts from 0
        num_cells = int(torch.prod(c_pos_shap).item())
        count_vec = torch.ones_like(c_pos_shap)
        count_vec[0] = c_pos_shap[1] * c_pos_shap[2]
        count_vec[1] = c_pos_shap[2]

        padded_cind = torch.sum(padded_cpos * count_vec, dim=1)
        padded_gind = (
            torch.arange(padded_cind.shape[0], device=count_vec.device) + 1
        )  # global index of padded atoms, starts from 1
        padded_rind = torch.arange(
            positions.shape[0], device=count_vec.device
        ).repeat_interleave(padding_size)  # local index of padded atoms in the unit box

        # atom box position and index
        atom_cpos = torch.floor(wrapped_pos / self.cutoff).long() - corner
        # atom neighbors' box position and index
        atom_cnpos = atom_cpos.unsqueeze(1) + self.disp_mat
        atom_cnind = torch.sum(atom_cnpos * count_vec, dim=-1)

        # construct a C x N matrix to store the box atom list, this is the most expensive part.
        padded_cind_sorted, padded_cind_args = torch.sort(padded_cind, stable=True)
        cell_ind, cell_atom_num = torch.unique_consecutive(
            padded_cind_sorted, return_counts=True
        )
        max_cell_anum = int(cell_atom_num.max().item())
        global_cell_ind = torch.zeros(
            (num_cells, max_cell_anum, 2),
            dtype=c_pos_shap.dtype,
            device=c_pos_shap.device,
        )
        cell_aind = torch.nonzero(
            torch.arange(max_cell_anum, device=count_vec.device).repeat(
                cell_atom_num.shape[0], 1
            )
            < cell_atom_num.unsqueeze(-1)
        )[:, 1]
        global_cell_ind[padded_cind_sorted, cell_aind, 0] = padded_gind[
            padded_cind_args
        ]
        global_cell_ind[padded_cind_sorted, cell_aind, 1] = padded_rind[
            padded_cind_args
        ]

        # masking
        atom_nind = global_cell_ind[atom_cnind]
        pair_i, neigh, j = torch.where(atom_nind[:, :, :, 0])
        pair_j = atom_nind[pair_i, neigh, j, 1]
        pair_j_padded = (
            atom_nind[pair_i, neigh, j, 0] - 1
        )  # remember global index of padded atoms starts from 1
        pair_diff = padded_pos[pair_j_padded] - wrapped_pos[pair_i]
        pair_dist = torch.norm(pair_diff, dim=1)
        mask = torch.logical_and(
            pair_dist < self.cutoff, pair_dist > 0.01
        )  # 0.01 for numerical stability
        pairs = torch.hstack((pair_i.unsqueeze(-1), pair_j.unsqueeze(-1)))
        return pairs[mask].to(torch.long)

    def wrap_positions(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """Wrap positions into the unit cell"""
        eps = torch.tensor(1e-7, device=positions.device, dtype=positions.dtype)
        # wrap atoms outside of the box
        scaled_pos = (positions @ torch.linalg.inv(box) + eps) % 1.0 - eps
        return scaled_pos @ box

    def forward_obc(self, positions: torch.Tensor) -> torch.Tensor:
        dist_mat = torch.cdist(positions, positions)
        mask = dist_mat < self.cutoff
        mask.fill_diagonal_(False)
        pairs = torch.argwhere(mask)
        return pairs.to(torch.long)

    @staticmethod
    def pairs_buffer_scales(pairs: torch.Tensor) -> torch.Tensor:
        """
        if pair_i < pair_j return 1, else return 0
        exclude repeated pairs and buffer pairs
        """
        dp = pairs[:, 0] - pairs[:, 1]
        return torch.where(
            dp < 0,
            torch.tensor(1, dtype=torch.long, device=pairs.device),
            torch.tensor(0, dtype=torch.long, device=pairs.device),
        )

    @staticmethod
    def pairs_ds(
        positions: torch.Tensor,
        pairs: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        pbc_flag: bool = True,
    ) -> torch.Tensor:
        """
        return the distance between pairs
        """
        ri = positions[pairs[:, 0]]
        rj = positions[pairs[:, 1]]
        if pbc_flag is False:
            dr = rj - ri
        else:
            assert box is not None, "Box should be provided for periodic system."
            dr = pbc_shift(ri - rj, box)
        ds = torch.linalg.vector_norm(dr, dim=1)
        return ds

    def set_pairs(self, pairs: torch.Tensor) -> None:
        self.pairs = pairs

    def set_buffer_scales(self, buffer_scales: torch.Tensor) -> None:
        self.buffer_scales = buffer_scales

    def set_ds(self, ds: torch.Tensor) -> None:
        self.ds = ds

    def get_pairs(self) -> torch.Tensor:
        return self.pairs

    def get_buffer_scales(self) -> torch.Tensor:
        return self.buffer_scales

    def get_ds(self) -> torch.Tensor:
        return self.ds
