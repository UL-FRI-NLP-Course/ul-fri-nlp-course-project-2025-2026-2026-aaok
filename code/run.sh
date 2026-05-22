# ============================================================
# SETUP INSTRUCTIONS (run this before submitting job)
# ============================================================
# 1. Load Python module:
#       module load Python/3.10
#
# 2. Go to code folder:
#       cd /code
#
# 3. Create virtual environment:
#       python -m venv legal_rag_env
#
# 4. Activate it:
#       source legal_rag_env/bin/activate
#
# 5. Install dependencies:
#       pip install -r requirements2.txt
#       pip install torch --index-url https://download.pytorch.org/whl/cu118
#
# 6. Submit job:
#       sbatch run.sh
# ============================================================

#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_output.log
#SBATCH --error=logs/%j_error.log

# activate environment
module load Python/3.10
source "$SLURM_SUBMIT_DIR/legal_rag_env/bin/activate"

# go to code directory
cd "$SLURM_SUBMIT_DIR"

# create logs dir if not exists
mkdir -p logs

echo "===== Running RAG evaluation ====="
python evaluation_rag.py

echo "===== Running GaMS3 generation ====="
python gams3.py

echo "===== Done ====="