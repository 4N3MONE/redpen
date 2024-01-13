import json 
from transformers import AutoTokenizer
from tqdm import tqdm
import time

with open('/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
openai2='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'
import openai
openai.api_key = openai2  

def get_completion_gpt3(input_data, model='gpt-3.5-turbo',max_length = 128, split_line=True):
    chat_response = openai.ChatCompletion.create(
    model= model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_data}
    ],
    max_tokens = max_length)
    chat_response = chat_response['choices'][0]['message']['content']
    if split_line:
        chat_response = chat_response.split('\n')
    return chat_response

def get_completion_with_backoff(input_data,max_length, max_retries=10):
    base_delay = 1 # 기본 대기 시간 (초)
    for attempt in range(max_retries + 1):
        try:
            result = get_completion_gpt3(input_data, model='gpt-3.5-turbo',max_length=max_length, split_line=True)
            return result
        except :
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) # 지수 백오프 계산
                print(f"타임아웃 에러 발생, {delay}초 후에 재시도합니다.")
                time.sleep(delay) # 계산된 대기 시간만큼 대기
            else:
                print(f"{topic}에 대한 요청 재시도 횟수를 초과했습니다. 나중에 다시 시도해주세요.")
                return None
            
for datum in tqdm(data):
    query = datum['question']
    mlen = 256-datum['q_len']
    datum['gpt-3.5']['response'] = get_completion_with_backoff(query, mlen,max_retries=10)
    
with open("/home/work/deeptext/yys/redpen/data/data_sampled_10k_only_eng_gpt_finished.json", "w", encoding='utf-8') as f:
    json.dump(data,f , ensure_ascii=False,indent=4)