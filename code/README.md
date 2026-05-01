## Set-up
First install the required dependencies for python 3.11.x:

```sh
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

To run a model using RAG, first build/unzip the FAISS index (explained in `rag/`), then run:
- `interactive.py`: To run a model with specialed RAG in interactive conversational mode.
- `evaluation.py`: To evaluate a model with specialised RAG on given queries and save the result to `results/`.

