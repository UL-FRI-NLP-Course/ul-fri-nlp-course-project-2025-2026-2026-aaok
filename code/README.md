## Setup Instructions

Run these steps once before submitting the job.

### 1. Load Python module
```bash
module load Python/3.10
```

### 2. Run everythin in code folder
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

## Submit job
```bash
sbatch run_gams3.sh
```
or 
```bash
sbatch run_qwen.sh
```
The results are saved in results/


To run a model using RAG, first build/unzip the FAISS index (explained in `rag/`), then run:
- `interactive.py`: To run a model with specialed RAG in interactive conversational mode.
- `evaluation.py`: To evaluate a model with specialised RAG on given queries and save the result to `results/`.

