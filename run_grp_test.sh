#!/bin/bash
#SBATCH -J boltz_grp
#SBATCH -o run_grp_test.out
#SBATCH -e run_grp_test.err
#SBATCH -p q3
#SBATCH --gres=gpu:1
# boltz group-COM angle/dihedral E2E. GPU work goes through sbatch.
#   cd boltz_restr && sbatch run_grp_test.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate

rm -rf out_grp_test
boltz predict grp_test.yaml --seed 0 --out_dir out_grp_test --model boltz2 \
    > run_grp_test.log 2>&1 || { echo "boltz FAILED:"; tail -n 30 run_grp_test.log; exit 1; }
# decisive signals: built spec counts (group_angle/group_dihedral non-zero) + finalize residuals
grep -iE "built spec|setup:|finalize" run_grp_test.log || true

CIF=$(find out_grp_test -name '*.cif' | head -1)
echo "CIF: $CIF"
# COM angle/dihedral vs target (gemmi via chai venv; boltz venv may lack gemmi)
GP=../chai-lab_restr/.venv/bin/python
"$GP" ../check_angle.py "$CIF" 5-84 90-180 186-224 || true
"$GP" ../check_dihedral.py "$CIF" 5-50 51-100 101-150 151-224 || true
echo done
