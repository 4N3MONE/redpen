import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import logging
import os
import shutil
import json
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
import argparse
import os
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base')
argparser.add_argument('--ability', type=str, default='readability')
argparser.add_argument('--epoch', type=int, default=10)
argparser.add_argument('--lr', type=float, default=5e-5)
argparser.add_argument('--batch_size', type=int, default=16)
#argparser.add_argument('--max_len', type = int, default=256)
args = argparser.parse_args()


train_data_dir = './data/train/'+args.ability+'_train.json'
test_data_dir = './data/test/'+args.ability+'_test.json'

data = load_dataset('json',data_files={'train':train_data_dir,'test':test_data_dir})
# train = data['train']
# test = data['train']
train = data['train'].shuffle(seed=1)
test = data['test'].shuffle(seed=1)
print('dataset loaded.')

model = AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=1,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

print('model and tokenizer loaded.')

train_encoding = tokenizer(
    train['text'],
    return_tensors='pt',
    padding=True,
    truncation=True
)
test_encoding = tokenizer(
    test['text'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

class ScoreDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx],dtype=torch.float32)

        return data

    def __len__(self):
        return len(self.labels)

train_set = ScoreDataset(train_encoding, train[args.ability])
test_set = ScoreDataset(test_encoding, test[args.ability])

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print(f'device : {device}')

train_loader = DataLoader(train_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

from tqdm.notebook import tqdm
from datasets import load_metric

def train(epoch, model, path,dataloader, optimizer, device, patience = 5):
    model.to(device)
    best_loss = float('inf')
    best_rmse = float('inf')
    cur_patience = 0
    m1 = load_metric('mse')
    losses = []
    rmse_values = []
    start_time = time.time()
    for e in range(1, epoch+1):
        total_loss = 0.
        preds = []
        labels = []
        progress_bar = tqdm(dataloader, desc=f'TRAIN - EPOCH {e} |')
        for data in progress_bar:
            data = {k:v.to(device) for k, v in data.items()}
            output = model(**data)
            
            current_loss = output.loss
            total_loss += current_loss.item()  # .item()을 사용하여 스칼라 값을 얻음
            
            preds += list(output.logits.squeeze().detach().cpu().numpy())
            labels += list(data['labels'].detach().cpu().numpy())

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            progress_bar.set_description(f'TRAIN - EPOCH {e} | current-loss: {current_loss:.4f}')
        
        train_rmse = m1.compute(predictions=preds, references=labels, squared=False)
        train_avg = total_loss / len(dataloader.dataset)
        losses.append(train_avg)
        rmse_values.append(train_rmse)
        print('='*64)
        print(f"TRAIN - EPOCH {e} | LOSS: {train_avg:.4f} RMSE: {train_rmse}")
        print('='*64)
        if train_avg<best_loss:
            best_loss = train_avg
            current_patience =0
            print(f'saving model in {path}')
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, path+f'model_{args.ability}.pt') 
        else:
            current_patience+=1
        if train_rmse['mse']<best_rmse:
            best_rmse = train_rmse['mse']
        if current_patience>=patience:
                break
    elapsed_time = round(time.time()-start_time,2)
    return best_loss, best_rmse, losses, rmse_values, elapsed_time

def evaluate(model,path, dataloader, device):
    model = torch.load(path+f'model_{args.ability}.pt')
    model.to(device)

    m1 = load_metric('mse')

    total_loss = 0.
    preds = []
    labels = []
    progress_bar = tqdm(dataloader, desc=f'EVAL |')
    predictions = {'true_score':[],'predicted_score':[]}
    for data in progress_bar:
        data = {k:v.to(device) for k, v in data.items()}
        with torch.no_grad():
            output = model(**data)
        
        current_loss = output.loss
        total_loss += current_loss.item()  # .item()을 사용하여 스칼라 값을 얻음
        
        preds += list(output.logits.squeeze().detach().cpu().numpy())
        labels += list(data['labels'].detach().cpu().numpy())
        # predictions['true_score'].extend(list(output.logits.squeeze().detach().cpu().numpy()))
        # prediections['predicted_score'].extend(list(data['labels'].detach().cpu().numpy()))
        progress_bar.set_description(f'EVAL | current-loss: {current_loss:.4f}')
    
    eval_rmse = m1.compute(predictions=preds, references=labels, squared=False)
    eval_avg = total_loss / len(dataloader.dataset)
    print('='*64)
    print(f"EVAL | LOSS: {eval_avg:.4f} RMSE: {eval_rmse}")
    print('='*64)
    return eval_avg, eval_rmse,preds,labels


optimizer = AdamW(model.parameters(), lr=args.lr)
print('train start.')
path_to_save = './/models//'+args.model_name.split('/')[-1]+'//'+args.ability+'//'
train_avg, train_rmse, losses, rmse_values, elapsed_time = train(args.epoch, model, path_to_save,train_loader, optimizer, device)
print('evaluation start.')
eval_avg, eval_rmse = evaluate(model,path_to_save, test_loader, device)

logdict = {}
logdict['model'] = args.model_name
logdict['ability'] = args.ability
logdict['train_loss'] = train_avg
logdict['train_rmse'] = train_rmse
logdict['eval_loss'] = eval_avg
logdict['eval_rmse'] = eval_rmse
logdict['log_loss'] = losses
logdict['log_rmse'] = rmse_values
logdict['train_time'] = elapsed_time
with open("./logs/regression_log.json", "r") as f:
    log = json.load(f)

print(f'\n\n\n\n\n{logdict}')
# Append the new log entry
log.append(logdict)

# Open the file in write mode to save the updated content
with open("./logs/regression_log.json", 'w') as f:
    json.dump(log, f, indent=4)

print('log saved.')