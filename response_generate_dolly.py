import json
from tqdm import  tqdm
import logging
import warnings
warnings.filterwarnings(action='ignore')
logging.disable(logging.INFO) 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# model_name_or_path = "databricks/dolly-v2-7b"
# generate_text = pipeline(model="databricks/dolly-v2-7b", 
#                          torch_dtype=torch.bfloat16, 
#                          trust_remote_code=True, 
#                          device_map="auto",
#                          max_new_tokens=max_len)
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b",
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", use_fast=True)

def generate_dolly(query, max_len):
    generate_text =pipeline(model=model,
                            tokenizer=tokenizer,
                            task='text-generation',
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True,
                            device_map="auto",
                            temperature=0.7,
                            top_p=0.95,
                            repetition_penalty=1.15,
                            max_new_tokens=max_len,
                            return_full_text = False)
    res = generate_text(query,pad_token_id=tokenizer.eos_token_id)
    return res[0]["generated_text"].replace('\n',' ')

with open('/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_llama2_finished_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
for datum in tqdm(data):
    q_len = 256-datum['q_len']
    query = datum['question']
    datum['dolly_v2']['response'] = generate_dolly(query, max_len=q_len)

    
with open("/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_dolly_finished.json", "w", encoding='utf-8') as f:
    json.dump(data,f , ensure_ascii=False,indent=4)