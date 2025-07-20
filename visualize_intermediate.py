#!/usr/bin/env python3

from pymol import cmd
from pathlib import Path

path_to_cif = "./"
nstep = 200
target = "denoised"

for i in range(nstep):
    cmd.load(f"{path_to_cif}/intermediate_{target}_{i}.cif")
    cmd.save(f"{path_to_cif}/intermediate_{target}_{i}.pdb")
    cmd.delete("all")

for i in range(nstep):
    cmd.load(f"{path_to_cif}/intermediate_{target}_{i}.pdb", f"target_{target}")

cmd.intra_fit("all")
cmd.color("0x0053D6", "b < 100")
cmd.color("0x65CBF3", "b < 90")
cmd.color("0xFFDB13", "b < 70")
cmd.color("0xFF7D45", "b < 50")

