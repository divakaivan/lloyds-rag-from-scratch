# Basic local RAG from scratch

### Repo structure

```
📁lloyds-rag-from-scratch
└── 📁dev
    └── dev_preprocess_pdf.ipynb -> preprocess pdf using llama-index
    └── dev_rag.ipynb -> runs rag using llama-index preprocessed pdf
└── lbg_relationship_tnc.pdf
└── lbg_relationship_tnc_locked.pdf
└── preprocess_pdf.py -> preprocess pdf using langchain
└── rag.py -> run local rag chat using langchain preprocessed pdf
└── requirements.txt
```

### Setup
1. `git clone https://github.com/divakaivan/pdf-rag-from-scratch.git`
2. `pip install -r requirements.txt`
3. `python preprocess_pdf.py` -> PDF must be saved in the same directory as the file, then it reads and processes the PDF for you, outputs a csv with the embeddings *(Note! use for up to 100k embeddings)*
4. `python rag.py` -> downloads gemma-2b-it, runs the RAG, and lets you have a chat with your PDF
5. (Optional) Run the dev versions (`dev_preprocess_pdf.ipynb` and `dev_rag.ipynb`) which uses llama-index as PDF reader and see the difference in the answer quality

### PDF preprocessing
In the dev folder, I use the files for development, but also am using llama-index, at the time of writing using it requires an API key, which is free, but we do not know in the future~

In preprocess_pdf.py and rag.py I use just local, pip install and run libraries.

### Demo of rag chat

https://github.com/divakaivan/lloyds-rag-from-scratch/assets/54508530/a1701a78-c160-43f6-bee0-d748d7954390

### Using out-of-the box embedding and language models
* embedding model: mixedbread-ai/mxbai-embed-large-v1
* LLM: google/gemma-2b-it

### Any feedback is welcome! ^^
