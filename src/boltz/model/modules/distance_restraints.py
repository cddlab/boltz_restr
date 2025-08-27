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
        self.distance_restraint_type = None # [harmonic, flat-bottomed, flat-bottomed1, flat-bottomed2]
        self.target_distance = None # used in harmonic
        self.target_distance1 = None # used in flat-bottomed, flat-bottomed1
        self.target_distance2 = None # used in flat-bottomed, flat-bottomed2
        self.method = None
        self.max_iter = None
        self.run_restr = None
        self.target_site1 = None
        self.target_site2 = None
        self.calc_method = None # [fixied-related, unfixed-absolute]
        # fixed-related: fix target_sites1 and move target_sites2
        # unfixed-absolute: move both target_sites1 and target_sites2

    def set_config(self, config: dict) -> None:
        self.config = config
        self.verbose = config.get("verbose", False)
        self.start_sigma = config.get("start_sigma", 1.0)
        if self.start_sigma is not None:
            self.start_sigma = float(self.start_sigma)
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
        self.method = config.get("method", "CG")
        self.max_iter = config.get("max_iter", 100)
        if self.max_iter is not None:
            self.max_iter = int(self.max_iter)
        self.run_restr = (self.atom_selection1 is not None) and (self.atom_selection2 is not None) and (self.distance_restraint_type is not None)

        if not self.run_restr:
            print("distance restraints not run")

        print(f"{self.distance_restraint_type=}")

        if self.calc_method not in ["fixed-related", "unfixed-absolute"]:
            print(f"calc_method must be fixed-related or unfixed-absolute")
            exit(1)

    def set_feats(self, feats) -> None:
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
            print(f"{self.target1_sites=}")
            print(f"{len(self.target2_sites)=}")
            print(f"{self.target2_sites=}")

    def minimize(self, batch_crds_in: torch.Tensor, istep: int, sigma_t: float) -> None:
        if not self.run_restr:
            return

        print(f"{sigma_t=}")
        if sigma_t > self.start_sigma:
            return

        device = batch_crds_in.device
        crds_in = batch_crds_in

        crds = crds_in.detach().cpu()
        crds_target1 = crds[:, self.target1_sites, :]
        crds_target2 = crds[:, self.target2_sites, :]

        if self.calc_method == "fixed-related":
            com_target1 = torch.mean(crds_target1, dim=1, keepdim=True)
            crds_target2_relative = crds_target2 - com_target1
            crds_target2_relative = crds_target2_relative.reshape(-1)
            self.nbatch = crds_target2.shape[0]
            self.natoms = crds_target2.shape[1]

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

        elif self.calc_method == "unfixed-absolute":
            crds_active = crds[:, self.target1_sites + self.target2_sites, :]
            self.nbatch = crds_active.shape[0]
            self.natoms = crds_active.shape[1]
            crds_active = crds_active.reshape(-1)

            opt = optimize.minimize(
                self.calc,
                crds_active,
                jac=self.grad,
                method=self.method,
                options={"maxiter": self.max_iter},
            )

            crds = opt.x.reshape(self.nbatch, self.natoms, 3)
            crds_target1 = crds[:, :len(self.target1_sites), :]
            crds_target2 = crds[:, len(self.target1_sites):, :]
            crds_in[:, self.target1_sites, :] = torch.tensor(crds_target1).to(device)
            crds_in[:, self.target2_sites, :] = torch.tensor(crds_target2).to(device)
        else:
            raise NotImplementedError

    def calc(self, crds_in: np.ndarray) -> float:
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)

        if self.calc_method == "fixed-related":
            com_per_batch = np.mean(crds, axis=1)
        elif self.calc_method == "unfixed-absolute":
            crds_target1 = crds[:, :len(self.target1_sites), :]
            crds_target2 = crds[:, len(self.target1_sites):, :]
            com_target1 = np.mean(crds_target1, axis=1, keepdims=True)
            crds_target2_relative = crds_target2 - com_target1
            com_per_batch = np.mean(crds_target2_relative, axis=1)
        else:
            raise NotImplementedError

        dist = np.linalg.norm(com_per_batch, axis=1)
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

        if self.verbose:
            print(f"{dist=}")
            print(f"{ene=}")
        return ene

    def grad(self, crds_in: np.ndarray) -> np.ndarray:
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)

        if self.calc_method == "fixed-related":
            com_per_batch = np.mean(crds, axis=1)
            D = np.linalg.norm(com_per_batch, axis=1)

            grad_com = np.zeros_like(com_per_batch)

            D_safe = D[:, None] + 1e-8

            if self.distance_restraint_type == "harmonic":
                coeff = 2 * (D - self.target_distance)
                grad_com = coeff[:, None] * com_per_batch / D_safe

            elif self.distance_restraint_type == "flat-bottomed":
                mask1 = D < self.target_distance1
                if np.any(mask1):
                    coeff1 = 2 * (D[mask1] - self.target_distance1)
                    grad_com[mask1] = coeff1[:, None] * com_per_batch[mask1] / D_safe[mask1]

                mask2 = D > self.target_distance2
                if np.any(mask2):
                    coeff2 = 2 * (D[mask2] - self.target_distance2)
                    grad_com[mask2] = coeff2[:, None] * com_per_batch[mask2] / D_safe[mask2]

            elif self.distance_restraint_type == "flat-bottomed1":
                mask = D < self.target_distance1
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance1)
                    grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]

            elif self.distance_restraint_type == "flat-bottomed2":
                mask = D > self.target_distance2
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance2)
                    grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]
            else:
                raise NotImplementedError

            grad_atom = grad_com / self.natoms
            grad = np.tile(grad_atom[:, None, :], (1, self.natoms, 1))
            grad = grad.reshape(-1)
        elif self.calc_method == "unfixed-absolute":
            n1 = len(self.target1_sites)
            n2 = len(self.target2_sites)
            crds_target1 = crds[:, :n1, :]
            crds_target2 = crds[:, n1:, :]
            com_target1 = np.mean(crds_target1, axis=1, keepdims=True)
            crds_target2_relative = crds_target2 - com_target1
            com_per_batch = np.mean(crds_target2_relative, axis=1)

            D = np.linalg.norm(com_per_batch, axis=1)
            grad_com = np.zeros_like(com_per_batch)
            D_safe = D[:, None] + 1e-8

            if self.distance_restraint_type == "harmonic":
                coeff = 2 * (D - self.target_distance)
                grad_com = coeff[:, None] * com_per_batch / D_safe

            elif self.distance_restraint_type == "flat-bottomed":
                mask1 = D < self.target_distance1
                if np.any(mask1):
                    coeff1 = 2 * (D[mask1] - self.target_distance1)
                    grad_com[mask1] = coeff1[:, None] * com_per_batch[mask1] / D_safe[mask1]

                mask2 = D > self.target_distance2
                if np.any(mask2):
                    coeff2 = 2 * (D[mask2] - self.target_distance2)
                    grad_com[mask2] = coeff2[:, None] * com_per_batch[mask2] / D_safe[mask2]

            elif self.distance_restraint_type == "flat-bottomed1":
                mask = D < self.target_distance1
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance1)
                    grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]

            elif self.distance_restraint_type == "flat-bottomed2":
                mask = D > self.target_distance2
                if np.any(mask):
                    coeff = 2 * (D[mask] - self.target_distance2)
                    grad_com[mask] = coeff[:, None] * com_per_batch[mask] / D_safe[mask]
            else:
                raise NotImplementedError

            grad_atom1 = -grad_com / n1
            grad_atom2 = grad_com / n2

            grad = np.zeros_like(crds)
            grad[:, :n1, :] = grad_atom1[:, np.newaxis, :]
            grad[:, n1:, :] = grad_atom2[:, np.newaxis, :]
            grad = grad.reshape(-1)

        else:
            raise NotImplementedError


        return grad
