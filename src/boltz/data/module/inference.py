from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
)


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(
            constraints_dir / f"{record.id}.npz"
        )

    return Input(structure, msas, record, residue_constraints)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "ligand_mols",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        constraints_dir: Optional[Path] = None,
        ccd_path: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        ccd_path : Optional[Path]
            Path to ccd.pkl ({CCD code -> RDKit mol}). When given, ligand mols are
            exposed as ``features['ligand_mols']`` so rgi_toolkit can build ligand
            conformer restraints under boltz1 (which has no mol_dir).

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        # Dir of per-target {id}.pkl holding SMILES-ligand RDKit mols (Target.extra_mols);
        # used as a fallback so conformer restraints work for SMILES ligands (not in CCD).
        self.extra_mols_dir = extra_mols_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        # Load the CCD mol dict once (boltz1 ships the single ccd.pkl rather than a
        # mol_dir). Used only to expose feats['ligand_mols'] for conformer restraints.
        self._ccd = None
        if ccd_path is not None and Path(ccd_path).exists():
            import pickle  # noqa: S403  local cache (boltz ships CCD as a pickle)

            # Same load as main.py for boltz1; trusted local file under the cache dir.
            with ccd_path.open("rb") as f:
                self._ccd = pickle.load(f)  # noqa: S301

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(
                record,
                self.target_dir,
                self.msa_dir,
                self.constraints_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None or len(options.pocket_constraints) == 0:
            binder, pocket = None, None
        else:
            binder, pocket = (
                options.pocket_constraints[0][0],
                options.pocket_constraints[0][1],
            )

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binder,
                inference_pocket=pocket,
                compute_constraint_features=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        features["record"] = record
        # Expose per-ligand RDKit mols for rgi_toolkit conformer restraints
        # (BoltzFeatsAdapter.iter_ligand_confs reads features['ligand_mols']).
        # boltz1 tokens carry no res_name, so resolve each non-polymer chain's CCD
        # code(s) from the residue table (mol.py:666 indexing) and combine per chain.
        if self._ccd is not None:
            from rdkit import Chem

            # SMILES ligands have no CCD code, so their mol isn't in ccd.pkl; it is
            # saved per-target in extra_mols_dir/{id}.pkl (parse_boltz_schema ->
            # Target.extra_mols). Load it once and fall back to it when the residue
            # name (e.g. "LIG0") isn't a CCD code -> conformer restraints then work for
            # SMILES ligands on boltz1, not just CCD ligands.
            extra_mols = {}
            if self.extra_mols_dir is not None:
                emp = Path(self.extra_mols_dir) / f"{record.id}.pkl"
                if emp.exists():
                    import pickle  # noqa: S403  trusted local cache (boltz writes it)

                    with emp.open("rb") as f:
                        extra_mols = pickle.load(f)  # noqa: S301

            nonpoly = const.chain_type_ids["NONPOLYMER"]
            struct = input_data.structure
            ligand_mols = {}
            for chain in struct.chains:
                if int(chain["mol_type"]) != nonpoly:
                    continue
                r0 = int(chain["res_idx"])
                res_mols = []
                for gidx in range(r0, r0 + int(chain["res_num"])):
                    name = str(struct.residues[gidx]["name"])
                    m = self._ccd.get(name)
                    if m is None:
                        m = extra_mols.get(name)  # SMILES ligand (not a CCD code)
                    if m is None:
                        res_mols = None
                        break
                    res_mols.append(m)
                if not res_mols:
                    continue
                chain_mol = res_mols[0]
                for m in res_mols[1:]:
                    chain_mol = Chem.CombineMols(chain_mol, m)
                ligand_mols[int(chain["asym_id"])] = chain_mol
            features["ligand_mols"] = ligand_mols
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        constraints_dir: Optional[Path] = None,
        ccd_path: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.ccd_path = ccd_path
        self.extra_mols_dir = extra_mols_dir

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
            ccd_path=self.ccd_path,
            extra_mols_dir=self.extra_mols_dir,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "ligand_mols",
            ]:
                batch[key] = batch[key].to(device)
        return batch
