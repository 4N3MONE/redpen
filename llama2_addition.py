import json
from tqdm import  tqdm
import logging
import warnings
warnings.filterwarnings(action='ignore')
logging.disable(logging.INFO) 

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os


model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def generate_llama2(query, max_len):
    prompt_template=f'''{query}

'''
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_len,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text = False
    )
    return pipe(prompt_template)[0]['generated_text'].replace('\n',' ')

with open('/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_llama2_finished.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
    
for datum in tqdm(data):
    if len(datum['llama2']['response'])>1:
        continue
    else:
        idx = datum['idx']
        query = datum['question']
        datum['llama2']['response'] = generate_llama2(query,max_len = 256-datum['q_len'])
        print(f'{idx} changed.')
        
with open("/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_llama2_finished_2.json", "w", encoding='utf-8') as f:
    json.dump(data,f , ensure_ascii=False,indent=4)