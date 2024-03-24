import re
import os
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

device = 'mps'

# lbg_relationship_tnc.pdf account_bank_tnc.pdf
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_n_texts = []

    for page_n, page in enumerate(doc):
        text = page.get_text()
        text = text.replace('\n', ' ').replace('  ', ' ')

        pages_n_texts.append({
            'page_n': page_n,
            'page_char_count': len(text),
            'page_word_count_raw': len(text.split(' ')),
            'page_sentence_count_raw': len(text.split('. ')),
            'page_token_count': len(text) / 4, # 1 token ~= 4 chars
            'text': text
        })

    return pages_n_texts

def create_text_splitter(chunk_size: int=1500, chunk_overlap: int=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


emb_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1').to(device)
#mixedbread-ai/mxbai-embed-large-v1 all-mpnet-base-v2

if __name__ == "__main__":

    pdf_path = input('Enter PDF name: ')

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")

    pages_n_texts = open_and_read_pdf(pdf_path)

    chunk_size = input('Enter chunk size(int). Default is 1500 (Enter -> skip): ')
    chunk_overlap = input('Enter chunk overlap(int). Default is 0 (Enter -> skip): ')
    if not chunk_size:
        chunk_size = 1500
    if not chunk_overlap:
        chunk_overlap = 0

    text_splitter = create_text_splitter(int(chunk_size), int(chunk_overlap))
    print('Creating sentence chunks ~')
    pages_n_chunks_new = []
    for item in pages_n_texts:
        item['sentence_chunks'] = text_splitter.split_text(item['text'])
        for chunk in item['sentence_chunks']:
            chunk_dict = {}
            chunk_dict['page_n'] = item['page_n']
            joined_sentence_chunk = ''.join(chunk).replace('  ', ' ').strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            joined_sentence_chunk = re.sub(r'\d+(\.\d+)+', '', joined_sentence_chunk)
            chunk_dict['sentence_chunk'] = joined_sentence_chunk

            # # add metadata
            chunk_dict['chunk_chars'] = len(joined_sentence_chunk)
            chunk_dict['chunk_words'] = len([word for word in joined_sentence_chunk.split(' ')])
            chunk_dict['chunk_tokens'] = len(joined_sentence_chunk) / 4

            pages_n_chunks_new.append(chunk_dict)

    text_chunks = [item['sentence_chunk'] for item in pages_n_chunks_new]
    text_chunk_embs = emb_model.encode(text_chunks, batch_size=16, convert_to_tensor=True)

    emb_chunks_df = pd.DataFrame(pages_n_chunks_new)
    emb_chunks_df['embedding'] = text_chunk_embs.cpu().numpy().tolist()
    emb_df_save_path = input('Enter name for embeddings csv. Default is emb_chunks_df (Enter -> skip): ')
    if not emb_df_save_path:
        emb_df_save_path = 'emb_chunks_df.csv'

    emb_chunks_df.to_csv(emb_df_save_path, index=False)
    print(f'File {emb_df_save_path} created!')
