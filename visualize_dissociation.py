#!/usr/bin/env python3

from pymol import cmd
from pathlib import Path

target_seed = "*"
path_to_cif = "./"

for i in Path(f"{path_to_cif}").glob(f"*_{target_seed}.cif"):
    cmd.load(str(i))
    cmd.save(f"{path_to_cif}/{i.stem}.pdb")
    cmd.delete("all")

for i in sorted(Path(f"{path_to_cif}").glob(f"*_{target_seed}.pdb")):
    print(i)
    cmd.load(str(i), "target")

cmd.intra_fit("chain A")
cmd.color("green", "chain A")
cmd.color("orange", "chain B")
