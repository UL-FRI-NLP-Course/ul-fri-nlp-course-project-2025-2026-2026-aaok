#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_output.log
#SBATCH --error=logs/%j_error.log
#SBATCH --mem=64G

# activate environment
module load Python/3.10
source "$SLURM_SUBMIT_DIR/legal_rag_env/bin/activate"

# go to code directory
cd "$SLURM_SUBMIT_DIR"

# create logs dir if not exists
mkdir -p logs

echo "===== Installing torch ====="
pip install bitsandbytes --quiet
pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --quiet

echo "===== Running RAG evaluation ====="
python evaluation_rag.py

python rag_test.py

echo "===== Running Model ====="
python gams3.py

echo "===== Done ====="