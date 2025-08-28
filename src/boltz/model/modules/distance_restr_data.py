from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
import torch
from rdkit import Chem
from scipy import optimize
from .selection import AtomSelector
from boltz.data import const

from pprint import pprint


@dataclass
class DistanceData:
    atom_selection1: str
    atom_selection2: str
    target_distance: float
    target_distance1: float # used in flat-bottomed, flat-bottomed1
    target_distance2: float # used in flat-bottomed, flat-bottomed2
    restraints_type: str # ["harmonic", "flat-bottomed", "flat-bottomed1", "flat-bottomed2"]
    target_sites1: list
    target_sites2: list
    target_local_sites1: list
    target_local_sites2: list
    calc_method: str # ["unfixed-absolute"], fixed-related can be used in single distance restraints
    # start_sigma: float
    run_restr: bool

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.target_distance = None
        self.target_distance1 = None
        self.target_distance2 = None
        self.restraints_type = None
        self.target_sites1 = None
        self.target_sites2 = None
        self.target_local_sites1 = None
        self.target_local_sites2 = None
        self.calc_method = None
        # self.start_sigma = None
        self.run_restr = None

    def set_config(self, config: dict):
        # self.start_sigma = config.get("start_sigma", 1.0)
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.calc_method = config.get("calc_method", "unfixed-absolute")
        if "harmonic" in config:
            self.target_distance = config["harmonic"].get("target_distance", None)
            if self.target_distance is not None:
                self.distance_restraint_type = "harmonic"
                self.target_distance = float(self.target_distance)
            else:
                print("target_distance is None")
                exit(1)
        elif "flat-bottomed" in config:
            self.target_distance1 = config["flat-bottomed"].get("target_distance1", None)
            self.target_distance2 = config["flat-bottomed"].get("target_distance2", None)
            if self.target_distance1 is not None and self.target_distance2 is not None:
                self.distance_restraint_type = "flat-bottomed"
                self.target_distance1 = float(self.target_distance1)
                self.target_distance2 = float(self.target_distance2)
                if self.target_distance1 > self.target_distance2:
                    print("target_distance1 must be smaller than target_distance2")
                    exit(1)
            else:
                print("target_distance1 or 2 is None")
                exit(1)
        elif "flat-bottomed1" in config:
            self.target_distance1 = config["flat-bottomed1"].get("target_distance1", None)
            if self.target_distance1 is not None:
                self.distance_restraint_type = "flat-bottomed1"
                self.target_distance1 = float(self.target_distance1)
            else:
                print("target_distance1 is None")
                exit(1)
        elif "flat-bottomed2" in config:
            self.target_distance2 = config["flat-bottomed2"].get("target_distance2", None)
            if self.target_distance2 is not None:
                self.distance_restraint_type = "flat-bottomed2"
                self.target_distance2 = float(self.target_distance2)
            else:
                print("target_distance2 is None")
                exit(1)
        self.run_restr = (self.atom_selection1 is not None) and (self.atom_selection2 is not None) and (self.distance_restraint_type is not None)

        if self.calc_method not in ["unfixed-absolute"]:
            print(f"calc_method must be fixed-related or unfixed-absolute")
            exit(1)

        if not self.run_restr:
            print("distance restraints not run")
            exit(1)

        print(f"{self.distance_restraint_type=}")


    def set_feats(self, feats):
        if not self.run_restr:
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
        asym_id_atom = asym_id_atom[atom_pad_mask.bool()]
        atom_to_token_b0 = atom_to_token[0]
        res_type_b0 = res_type[0]
        ref_space_uid_b0 = ref_space_uid[0]

        self.target_sites1 = []
        self.target_sites2 = []

        self.target_local_sites1 = []
        self.target_local_sites2 = []

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
                    self.target_sites1.append(global_padded_idx)
                if atom_selector2.eval(candidate_atom):
                    self.target_sites2.append(global_padded_idx)

        assert len(self.target_sites1) != 0, "target_sites1 is empty"
        assert len(self.target_sites2) != 0, "target_sites2 is empty"

        print(f"{len(self.target_sites1)=}")
        print(f"{self.target_sites1=}")
        print(f"{len(self.target_sites2)=}")
        print(f"{self.target_sites2=}")


    def calc(self, crds_in: np.ndarray) -> float:
        # if self.calc_method == "fixed-related":
        #     com_per_batch = np.mean(crds_in, axis=1)
        if self.calc_method == "unfixed-absolute":
            crds_target1 = crds_in[self.target_local_sites1, :]
            crds_target2 = crds_in[self.target_local_sites2, :]
            com_target1 = np.mean(crds_target1, axis=0, keepdims=True)
            crds_target2_relative = crds_target2 - com_target1
            com_per_batch = np.mean(crds_target2_relative, axis=0)
        else:
            raise NotImplementedError

        dist = np.linalg.norm(com_per_batch)
        ene = 0.0
        if self.distance_restraint_type == "harmonic":
            ene = np.sum((dist - self.target_distance) ** 2)
        elif self.distance_restraint_type == "flat-bottomed":
            ene += np.sum((dist[dist < self.target_distance1] - self.target_distance1) ** 2)
            ene += np.sum((dist[dist > self.target_distance2] - self.target_distance2) ** 2)
        elif self.distance_restraint_type == "flat-bottomed1":
            ene = np.sum((dist[dist < self.target_distance1] - self.target_distance1) ** 2)
        elif self.distance_restraint_type == "flat-bottomed2":
            ene = np.sum((dist[dist > self.target_distance2] - self.target_distance2) ** 2)
        else:
            raise NotImplementedError

        return ene


    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        # if self.calc_method == "fixed-related":
        #     com_per_batch = np.mean(crds, axis=1)
        #     D = np.linalg.norm(com_per_batch, axis=1)
        #     grad_com = np.zeros_like(com_per_batch)
        #     D_safe = D[:, None] + 1e-8
        #
        #     if self.distance_restraint_type == "harmonic":
        #         coeff = 2 * (D - self.target_distance)
        #         grad_com = coeff[:, None] * com_per_batch / D_safe
        #
        #     elif self.distance_restraint_type == "flat-bottomed":
        #         mask1 = D < self.target_distance1
        #         if np.any(mask1):
        #             coeff1 = 2 * (D[mask1] - self.target_distance1)
        #             grad_com[mask1] = coeff1[:, None] * com_per_batch[mask1] / D_safe[mask1]
        #
        #         mask2 = D > self.target_distance2
        #         if np.any(mask2):
        #             coeff2 = 2 * (D[mask2] - self.target_distance2)
        #             grad_com[mask2] = coeff2[:, None] * com_per_batch[mask2] / D_safe[mask2]
        #
        #     elif self.distance_restraint_type == "flat-bottomed1":
        #         mask = D < self.target_distance1
        #         if np.any(mask):
        #             coeff = 2 * (D[mask] - self.target_distance1)
        #             grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]
        #
        #     elif self.distance_restraint_type == "flat-bottomed2":
        #         mask = D > self.target_distance2
        #         if np.any(mask):
        #             coeff = 2 * (D[mask] - self.target_distance2)
        #             grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]
        #     else:
        #         raise NotImplementedError
        #
        #     grad_atom = grad_com / self.natoms
        #     grad[:, self.target_local_sites2, ] += np.tile(grad_atom[:, None, :], (1, len(self.target_local_sites2), 1))

        if self.calc_method == "unfixed-absolute":
            crds_target1 = crds[self.target_local_sites1, :]
            crds_target2 = crds[self.target_local_sites2, :]
            com_target1 = np.mean(crds_target1, axis=0, keepdims=True)
            crds_target2_relative = crds_target2 - com_target1
            com_per_batch = np.mean(crds_target2_relative, axis=0)

            D = np.linalg.norm(com_per_batch)
            grad_com = np.zeros_like(com_per_batch)
            D_safe = D + 1e-8

            if self.distance_restraint_type == "harmonic":
                coeff = 2 * (D - self.target_distance)
                grad_com = coeff * com_per_batch / D_safe

            elif self.distance_restraint_type == "flat-bottomed":
                mask1 = D < self.target_distance1
                if np.any(mask1):
                    coeff1 = 2 * (D[mask1] - self.target_distance1)
                    grad_com[mask1] = coeff1 * com_per_batch[mask1] / D_safe[mask1]

                mask2 = D > self.target_distance2
                if np.any(mask2):
                    coeff2 = 2 * (D[mask2] - self.target_distance2)
                    grad_com[mask2] = coeff2 * com_per_batch[mask2] / D_safe[mask2]

            elif self.distance_restraint_type == "flat-bottomed1":
                mask = D < self.target_distance1
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance1)
                    grad_com[mask] = coeff * com_per_batch[mask] / D_safe[mask]

            elif self.distance_restraint_type == "flat-bottomed2":
                mask = D > self.target_distance2
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance2)
                    grad_com[mask] = coeff * com_per_batch[mask] / D_safe[mask]
            else:
                raise NotImplementedError

            grad_atom1 = -grad_com / len(self.target_sites1)
            grad_atom2 = grad_com / len(self.target_sites2)

            grad[self.target_local_sites1, :] += grad_atom1[np.newaxis, :]
            grad[self.target_local_sites2, :] += grad_atom2[np.newaxis, :]
        else:
            raise NotImplementedError


    def is_valid(self) -> bool:
        return self.run_restr

    def distance(self, crds: np.ndarray) -> float:
        com_1 = np.mean(crds[self.target_local_sites1, :], axis=0)
        com_2 = np.mean(crds[self.target_local_sites2, :], axis=0)
        return np.linalg.norm(com_1 - com_2)

    def print(self, crds: np.ndarray) -> None:
        print(f"COM distance of '{self.atom_selection1}' - '{self.atom_selection2}': {self.distance(crds)}")

    def calc_sd(self, crds: np.ndarray) -> float:
        dist = self.distance(crds)
        return (dist - self.target_distance) ** 2
