from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from sentence_transformers import util, SentenceTransformer
import pandas as pd
import numpy as np
import torch

device = 'mps'

emb_chunks_df = pd.read_csv('emb_chunks_df.csv')

# convert embeddings back to np.array
emb_chunks_df['embedding'] = emb_chunks_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=', '))
embs = torch.tensor(np.stack(emb_chunks_df['embedding'].tolist(), axis=0), dtype=torch.float32).to(device)

pages_n_chunks = emb_chunks_df.to_dict(orient='records')

emb_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device=device)

def retrieve_relevant_info(query: str, embeddings: torch.tensor, model: SentenceTransformer=emb_model, n_to_retrieve: int=5) -> torch.tensor:
    query_emb = model.encode(query, convert_to_tensor=True)
    dot_scores = util.cos_sim(query_emb, embeddings)[0]
    scores, indices = torch.topk(dot_scores, n_to_retrieve)
    print(scores)
    return scores, indices

model_id = 'google/gemma-2b-it'

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False, attn_implementation='sdpa').to(device)

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = '- ' + '\n- '.join([item['sentence_chunk'] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style, but don't use the below example answers as answers to the query.
\nExample 1:
Query: Who can provide instructions to the bank according to the terms and conditions?
Answer: According to the terms and conditions, only authorized individuals can give instructions to the bank.
\nExample 2:
Query: What are your rights regarding the termination of services as outlined in the terms and conditions?
Answer: The terms and conditions specify the rights granted to you in the event of termination, including any associated procedures or obligations.
\nExample 3:
Query: How does the bank handle refunds for incorrectly executed payment instructions, as per the terms and conditions?
Answer: The terms and conditions detail the process for obtaining refunds in the case of payment instructions being incorrectly executed by the bank.
\nExample 4:
Query: What measures are outlined in the terms and conditions to ensure the security of your accounts and payment instruments?
Answer: The terms and conditions lay out your obligations regarding the security of your accounts, payments, and payment instruments, along with any corresponding measures implemented by the bank.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    base_prompt = base_prompt.format(context=context, query=query)
    
    # make sure the inputs to the model are in the same way that they have been trained
    dialogue_template = [
        {
            'role': 'user',
            'content': base_prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

    return prompt

def ask(query: str, temperature: float=0.2, max_new_tokens: int=256, format_answer_text: bool=True, return_context: bool=False):
    # -------- RETRIEVAL --------
    scores, indices = retrieve_relevant_info(query, embs, n_to_retrieve=10)
    context_items = [pages_n_chunks[i] for i in indices]
    for i, item in enumerate(context_items):
        item['score'] = scores[i].cpu()

    # -------- AUGMENTATION --------
    prompt = prompt_formatter(query, context_items)

    # -------- GENERATION --------
    input_ids = tokenizer(prompt, return_tensors='pt').to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = llm_model.generate(**input_ids, streamer=streamer, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        output_text = output_text.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')

    # if not return_context:
        # return output_text
    
    # return output_text, context_items

if __name__ == "__main__":

    print('Enter a query:\n')
    query = input()
    print('estimating ~ estimating ~')
    ask(query, temperature=0.7, return_context=False)