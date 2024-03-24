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
    base_prompt = """You are a helpful assisstant to customers about a bank's terms and conditions. 
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as clear and concise.
Use the following couple of examples as reference for the ideal answer style, but don't use the below example answers as answers to the query.
\nExample 1:
User query: I'm considering opening a new savings account with a competitive interest rate. However, I noticed a clause regarding minimum balance requirements. Could you elaborate on the potential implications of not maintaining this minimum balance?
AI answer: That's a prudent inquiry!  Many banks offer attractive interest rates on savings accounts, but they may stipulate a minimum balance requirement.  Failing to maintain this minimum can trigger various consequences, including incurring fees or forfeiting the advertised interest rate. Carefully review the minimum balance stipulation within the T&Cs to ensure it aligns with your financial situation.
\nExample 2:
User query: My bank has been sending frequent notifications regarding mobile banking security. While I appreciate the reminder, is utilizing mobile banking inherently risky?
AI answer: Mobile banking offers undeniable convenience but does necessitate vigilance. While not inherently risky, online transactions always carry a certain level of risk.  To mitigate these risks, ensure your mobile device is equipped with a strong password and avoid using public Wi-Fi networks for banking activities. Your bank's security notifications serve as a valuable reminder to prioritize online safety measures.
\nNow based on the following context items:
{context};
\n And answer the user's query:
User query: <start_of_turn>user{query}<end_of_turn>
AI answer:<start_of_turn>model"""

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
    scores, indices = retrieve_relevant_info(query, embs, n_to_retrieve=5)
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

    while True:
        print('Enter a query')
        query = input('(Enter q to quit) ')
        if query == 'q':
            break
        print('estimating ~ estimating ~')
        ask(query, temperature=0.7, return_context=False)