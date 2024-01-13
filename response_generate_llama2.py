import json
from tqdm import  tqdm
import logging
import warnings
warnings.filterwarnings(action='ignore')
logging.disable(logging.INFO) 

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from huggingface_hub import login
login(token='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')
llama2  = "meta-llama/Llama-2-7b-chat-hf"

# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"

model_llama2 = AutoModelForCausalLM.from_pretrained(llama2,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

tokenizer_llama2 = AutoTokenizer.from_pretrained(llama2, use_fast=True)


def generate_llama2(query, max_len):
    prompt_template=f'''{query}

'''
    pipe = pipeline(
        "text-generation",
        model=model_llama2,
        tokenizer=tokenizer_llama2,
        max_new_tokens=max_len,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text = False
    )
    return pipe(prompt_template)[0]['generated_text'].replace('\n',' ')

with open('/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_llama2_finished.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
error_limit =50
for datum in tqdm(data):
    query = datum['question']
    empty_count = 0
    while 1:
        res = generate_llama2(query=query,max_len=256-datum['q_len']).lstrip(' ')
        if len(res)>0:
            break
        else:
            empty_count+=1
            if empty_count==error_limit:
                res = ''
                print(f'empty response! idx: {datum["idx"]}')
                break
    datum['llama2']['response'] = res


with open("/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_llama2_finished.json", "w", encoding='utf-8') as f:
    json.dump(data,f , ensure_ascii=False,indent=4)