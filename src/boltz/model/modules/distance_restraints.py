from __future__ import annotations
import itertools
import numpy as np
import torch
from rdkit import Chem
from scipy import optimize
from .selection import AtomSelector
from boltz.data import const

from pprint import pprint

class DistanceRestraints:

    _instance = None

    @classmethod
    def get_instance(cls) -> DistanceRestraints:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.config = None
        self.verbose = None
        self.start_sigma = None
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.target_distance = None
        self.method = None
        self.max_iter = None
        self.run_restr = None
        self.target_site1 = None
        self.target_site2 = None

    def set_config(self, config: dict) -> None:
        self.config = config
        self.verbose = config.get("verbose", False)
        self.start_sigma = config.get("start_sigma", 1.0)
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.target_distance = config.get("target_distance", None) # angstrom
        self.method = config.get("method", "CG")
        self.max_iter = config.get("max_iter", 100)
        self.run_restr = (self.atom_selection1 is not None) and (self.atom_selection2 is not None) and (self.target_distance is not None)

        if not self.run_restr:
            print("distance restraints not run")

    def set_feats(self, feats) -> None:
        if not self.run_restr:
            print("distance restraints not run")
            return
        asym_id_token = feats['asym_id']
        atom_pad_mask = feats['atom_pad_mask']
        atom_to_token = feats['atom_to_token']
        mol_type = feats["mol_type"]
        record = feats["record"]
        ref_atom_name_chars = feats["ref_atom_name_chars"]
        ref_element = feats["ref_element"]
        ref_space_uid = feats["ref_space_uid"]
        res_type = feats["res_type"]
        asym_id_atom = torch.bmm(atom_to_token.float(), asym_id_token.unsqueeze(-1).float()).squeeze(-1).long()
        if self.verbose:
            pprint(f"{asym_id_atom.shape=}")
            pprint(f"{asym_id_atom.tolist()=}")
        asym_id_atom = asym_id_atom[atom_pad_mask.bool()]
        if self.verbose:
            pprint(f"{asym_id_atom.shape=}")
            pprint(f"{asym_id_atom.tolist()=}")
        atom_to_token_b0 = atom_to_token[0]
        res_type_b0 = res_type[0]
        ref_space_uid_b0 = ref_space_uid[0]

        self.target1_sites = []
        self.target2_sites = []

        atom_selector1 = AtomSelector(self.atom_selection1)
        atom_selector2 = AtomSelector(self.atom_selection2)

        for chain in record[0].chains:
            chain_id = chain.chain_id
            mol_type = chain.mol_type
            chain_type = const.chain_types[mol_type]
            chain_sites = torch.where(asym_id_atom == chain_id)[0].tolist()
            for local_idx, global_padded_idx in enumerate(chain_sites):
                token_idx = torch.argmax(atom_to_token_b0[global_padded_idx, :]).item()
                # res_type_id = torch.argmax(res_type_b0[token_idx, :]).item()
                # res_name = const.tokens[res_type_id] # const.tokens is a list/array
                # uid = ref_space_uid_b0[global_padded_idx].item()
                # atoms_in_same_residue_instance_global_idxs = torch.where(ref_space_uid_b0 == uid)[0]
                # idx_in_residue_atom_list = torch.searchsorted(atoms_in_same_residue_instance_global_idxs, global_padded_idx).item()
                # if atoms_in_same_residue_instance_global_idxs[idx_in_residue_atom_list] != global_padded_idx:
                #     try:
                #         idx_in_residue_atom_list = torch.where(atoms_in_same_residue_instance_global_idxs == global_padded_idx)[0].item()
                #     except: # Handle cases where it might not be found or multiple found
                #         print(f"Error finding {global_padded_idx} in residue list for UID {uid}. Skipping atom.")
                #         continue
                # ref_atom_names_for_res = const.ref_atoms.get(res_name)
                # if ref_atom_names_for_res is None or idx_in_residue_atom_list >= len(ref_atom_names_for_res):
                #     print(f"Warning: Atom index {idx_in_residue_atom_list} out of bounds for {res_name} (len {len(ref_atom_names_for_res if ref_atom_names_for_res else 0)}) or res_name not in const.ref_atoms. Skipping.")
                #     continue
                # atom_name = ref_atom_names_for_res[idx_in_residue_atom_list]

                candidate_atom = {
                    "chain": chain.chain_name,
                    # "resname": res_name, # due to nonpolymer name will be LIG(smiles) or UNK(CCD)
                    "resid": token_idx + 1,
                    # "name": atom_name,
                    "index": global_padded_idx,
                }

                if atom_selector1.eval(candidate_atom):
                    self.target1_sites.append(global_padded_idx)
                if atom_selector2.eval(candidate_atom):
                    self.target2_sites.append(global_padded_idx)

        assert len(self.target1_sites) != 0, "target1_sites is empty"
        assert len(self.target2_sites) != 0, "target2_sites is empty"

        if self.verbose:
            print(f"{len(self.target1_sites)=}")
            print(f"{len(self.target2_sites)=}")

    def minimize(self, batch_crds_in: torch.Tensor, istep: int, sigma_t: float) -> None:
        if not self.run_restr:
            print("distance restraints not run")
            return

        print(f"{sigma_t=}")
        if sigma_t > self.start_sigma:
            return

        device = batch_crds_in.device
        crds_in = batch_crds_in

        crds = crds_in.detach().cpu()
        crds_target1 = crds[:, self.target1_sites, :]
        crds_target2 = crds[:, self.target2_sites, :]
        com_target1 = torch.mean(crds_target1, dim=1, keepdim=True)
        self.nbatch = crds_target2.shape[0]
        self.natoms = crds_target2.shape[1]
        crds_target2_relative = crds_target2 - com_target1
        crds_target2_relative = crds_target2_relative.reshape(-1)

        opt = optimize.minimize(
            self.calc,
            crds_target2_relative,
            jac=self.grad,
            method=self.method,
            options={"maxiter": self.max_iter},
        )

        crds_target2_relative = opt.x.reshape(self.nbatch, self.natoms, 3)
        crds_target2_relative = torch.from_numpy(crds_target2_relative)
        crds_target2 = crds_target2_relative + com_target1
        crds_in[:, self.target2_sites, :] = torch.tensor(crds_target2).to(device)

    def calc(self, crds_in: np.ndarray) -> float:
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        com_per_batch = np.mean(crds, axis=1)
        dist = np.linalg.norm(com_per_batch, axis=1)
        ene = np.sum((dist - self.target_distance) ** 2)

        return ene

    def grad(self, crds_in: np.ndarray) -> np.ndarray:
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        com_per_batch = np.mean(crds, axis=1)
        D = np.linalg.norm(com_per_batch, axis=1)
        # loss = np.sum((D - self.target_distance) ** 2)
        grad_com = 2 * (D - self.target_distance)[:, None] * com_per_batch / (D[:, None] + 1e-8)
        grad_atom = grad_com / self.natoms
        grad = np.tile(grad_atom[:, None, :], (1, self.natoms, 1))
        grad = grad.reshape(-1)

        return grad

