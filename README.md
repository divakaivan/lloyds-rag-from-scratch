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

### PDF preprocessing
In the dev folder, I use the files for development, but also am using llama-index, at the time of writing using it requires an API key, which is free, but we do not know in the future~

In preprocess_pdf.py and rag.py I use just local, pip install and run libraries.

### Demo of rag chat

https://github.com/divakaivan/lloyds-rag-from-scratch/assets/54508530/a1701a78-c160-43f6-bee0-d748d7954390

### Any feedback is welcome! ^^
