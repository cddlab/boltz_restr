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

cmd.load(f"{path_to_cif}/intermediate_{target}_{nstep-1}.pdb", f"target_{target}")
for i in range(nstep):
    cmd.load(f"{path_to_cif}/intermediate_{target}_{i}.pdb", f"target_{target}")

def delete_state(obj_name, state_to_delete):
    n_states = cmd.count_states(obj_name)

    if n_states <= 1:
        if state_to_delete == 1:
            cmd.delete(obj_name)
        return

    temp_obj = obj_name + "_temp_reserved"

    created = False
    for i in range(1, n_states + 1):
        if i == state_to_delete:
            continue

        if not created:
            cmd.create(temp_obj, obj_name, i, 1)
            created = True
        else:
            cmd.create(temp_obj, obj_name, i, -1)

    cmd.delete(obj_name)
    cmd.set_name(temp_obj, obj_name)

delete_state(f"target_{target}", 1)

cmd.intra_fit("all")
cmd.color("0x0053D6", "b < 100")
cmd.color("0x65CBF3", "b < 90")
cmd.color("0xFFDB13", "b < 70")
cmd.color("0xFF7D45", "b < 50")

