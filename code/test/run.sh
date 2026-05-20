#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%J.out
#SBATCH --error=logs/eval-%J.err
#SBATCH --job-name="RAG eval"

module load Python/3.10
source /d/hpc/home/aa1737/test/rag_clean_env/bin/activate #CHANGE THE ENV PATH

#cd /d/hpc/home/aa1737/ul-fri-nlp-course-project-2025-2026-2026-aaok/code/test

echo "Running RAG..."
python rag.py

echo "Running RAG evaluation..."
python evaluationRAG.py

echo "Running Gams3..."
python gams3.py #gams3.py, llama.py, mistral.py, qwen.py, phi.py
echo "Gams3 done"

echo "Running LLaMA..."
python llama.py
echo "Llama done"

echo "Running Mistral..."
python mistral.py
echo "Mistral done"
