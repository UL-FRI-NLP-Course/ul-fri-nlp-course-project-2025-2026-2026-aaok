#!/bin/bash
#SBATCH --job-name=single_question
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j_output.log
#SBATCH --error=logs/%j_error.log
#SBATCH --mem=64G

module load Python/3.10
source "$SLURM_SUBMIT_DIR/legal_rag_env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

pip install bitsandbytes --quiet
pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --quiet

echo "===== Running single question ====="
python gams3_test.py
echo "===== Done ====="