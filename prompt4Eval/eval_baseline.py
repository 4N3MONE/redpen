import os
import argparse
import pickle
import time
import sys
import shutil
import warnings
from tqdm import tqdm
from datetime import datetime
import torch.cuda
from torch.utils.data import DataLoader
import json
from transformers import AutoTokenizer
from transformers import AdamW
from model import LMPrompt4Eval
from prepro_data import *
#from utils import evaluate
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base')
parser.add_argument('--ability', type=str, default='readability')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_tokens', default=512, type=int, help='max number of tokens')
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--save_path',  type=str, default = './prompt_models/')
parser.add_argument('--patience', type=int, default=5)

args = parser.parse_args()

save_path = args.save_path+args.model_name.split('/')[-1]+'//'+args.ability+'//'

def load_model(model_name, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    answer = ['terrible','poor','average','good','excellent']
    answer_str = ' '.join(answer)  # 리스트를 공백으로 구분된 문자열로 변환
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)

    net = LMPrompt4Eval(model_name, answer_ids, args)
    return net, tokenizer


@torch.no_grad()
def eval(model, data_loader):
    model.to(device)
    model.eval()
    data_loader=tqdm(data_loader)
    val_scores = []
    acc_cnt = torch.zeros(2).to(device)
    labels =[]
    total_loss=0.0
    predictions = {'true_labels': [], 'predicted_labels': []}
    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs = data
        labels = labels + batch_labs.cpu().numpy().tolist()
            
        batch_enc = batch_enc.to(device)
        batch_attn = batch_attn.to(device)
        batch_labs = batch_labs.to(device)

        loss, scores = model(batch_enc, batch_attn, batch_labs)
        total_loss+=loss.item()
        ranking_scores = scores[:, 1].detach()
        val_scores.append(ranking_scores)

        predict = torch.argmax(scores.detach(), dim=1)
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)
        predictions['true_labels'].extend(batch_labs.cpu().numpy().tolist())
        predictions['predicted_labels'].extend(predict.cpu().numpy().tolist())

    acc = acc_cnt[0] / acc_cnt[1]
    val_loss = total_loss / len(data_loader)

    return val_scores, val_loss, acc.item(),predictions, 

#load model
net, tokenizer = load_model(args.model_name, args)
print('model loded.')

#load data
train_dataset = MyDataset(args,tokenizer=tokenizer,path = f'/home/work/deeptext/yys/redpen/data/train/{args.ability}_prompt_train.json')
test_dataset = MyDataset(args,tokenizer=tokenizer,path=f'/home/work/deeptext/yys/redpen/data/test/{args.ability}_prompt_test.json' )
# print(train_dataset[0]['text'])
# print(test_dataset[3]['text'])


# subset_size = 100
# train_dataset = train_dataset[:subset_size]
# test_dataset = test_dataset[:subset_size]

train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': False, 'pin_memory': True, 'collate_fn': train_dataset.collate_fn}
val_kwargs = {'batch_size': args.batch_size, 
                  'shuffle': False, 'pin_memory': True, 'collate_fn': train_dataset.collate_fn}
train_loader = DataLoader(train_dataset, **train_kwargs)
val_loader = DataLoader(test_dataset, **val_kwargs)

print('data loaded.')

# AdamW
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'device : {device}')

best_loss = float('inf')
best_acc = 0
cur_patience = 0
patience_cut = args.patience

losses = []
acc_values = []

torch.cuda.empty_cache()

# #################################  val  ###################################
#best_model = torch.load(save_path+f'model_{args.ability}.pt')
val_scores, val_loss,acc_val,predictions = eval(net, val_loader)

print(f'validation loss = {val_loss}')
print(f'validation acc = {acc_val}')

end_val = time.time()
logdict = {}
logdict['model'] = args.model_name
logdict['ability'] = args.ability
logdict['eval_loss'] = val_loss
logdict['eval_acc'] = acc_val

with open("/home/work/deeptext/yys/redpen/logs/baseline.json", "r") as f:
    log = json.load(f)

#print(f'\n\n\n\n\n{logdict}')
# Append the new log entry
log.append(logdict)

# Open the file in write mode to save the updated content
with open("/home/work/deeptext/yys/redpen/logs/baseline.json", 'w') as f:
    json.dump(log, f, indent=4)

print('log saved.')