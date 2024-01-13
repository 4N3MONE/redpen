
import re
import random
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import torch
from datasets import load_dataset


class MyDataset(Dataset):
    def __init__(self, args,tokenizer,path):
        self.tokenizer = tokenizer
        self.data = []
        self.data_path = path
        self.args = args
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


    def calc_len(self,text):
        return len(self.tokenizer.tokenize(text))

    def load_data(self):
        datafile = load_dataset('json',data_files=self.data_path)
        for file in datafile['train']:
            if file['label']=='terrible':
                label = 0
            elif file['label']=='poor':
                label = 1
            elif file['label']=='average':
                label = 2
            elif file['label']=='good':
                label = 3
            elif file['label']=='excellent':
                label = 4


            input_text = 'Question : '+file['question']+self.tokenizer.sep_token+'Response : '+file['response']+\
            self.tokenizer.sep_token+f' Based on this response, the {self.args.ability} level of the response is '+self.tokenizer.mask_token

            #len(input_text) < max_input_size
            self.data.append({'text':input_text, 'label':label})


   

    def collate_fn(self, batch):
        sentences = [x['text'] for x in batch]
        target = [x['label'] for x in batch]

        encode_dict = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.args.max_tokens,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        # target_tokens = self.tokenizer(target, add_special_tokens=False)['input_ids']
        # target = torch.tensor(target_tokens)
        target = torch.LongTensor(target)
        return batch_enc, batch_attn, target





