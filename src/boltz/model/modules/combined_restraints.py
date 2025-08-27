from __future__ import annotations
import itertools
import numpy as np
import torch
from rdkit import Chem
from scipy import optimize
from pprint import pprint
from .conformer_restraints import ConformerRestraints
from .distance_restraints import DistanceRestraints


class CombinedRestraints:

    _instance = None

    @classmethod
    def get_instance(cls) -> CombinedRestraints:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        pass

    def minimize(self, batch_crds_in: torch.Tensor, istep: int, sigma_t: float) -> None:
        pass

    def calc(self, crds_in: np.ndarray) -> float:
        ene = 0.0
        return ene

    def grad(self, crds_in: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(crds_in)
        return grad
