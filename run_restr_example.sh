#!/bin/bash
#SBATCH -J boltz_ex
#SBATCH -o run_restr_example.out
#SBATCH -e run_restr_example.err
#SBATCH -p q1
#SBATCH --gres=gpu:1
# boltz RGI example runner. GPU work must go through sbatch (not the login node).
# Submit from THIS repo directory:  cd boltz_restr && sbatch run_restr_example.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate

rm -rf out_restr_example
# restr_example.yaml sets `msa: empty`, so no MSA server is needed (single-sequence).
# RGI: rgi_utils minimizes the distance + conformer restraints on the x0 prediction
# after each diffusion step (verbose:true logs the built spec + finalize energies).
boltz predict restr_example.yaml \
    --seed 0 --out_dir out_restr_example --model boltz2

CIF=$(find out_restr_example -name '*.cif' | head -1)
echo "prediction: $CIF"
echo done
