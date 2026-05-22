## Setup Instructions

### 1. Load Python module
```bash
module load Python/3.10
```

### 2. Run everything in code folder
```bash
cd code/
```

### 3. Create virtual environment
```bash
python -m venv legal_rag_env
source legal_rag_env/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

# 4. Create retrieval indexes
Instructions can be found inside `code/rag/`.

## Running the evaluation job
```bash
sbatch run_gams3.sh
```
or 
```bash
sbatch run_qwen.sh
```
The results are saved in `results/`

