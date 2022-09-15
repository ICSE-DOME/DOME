import json

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class DOMEDataset(Dataset):
    def __init__(self, tokenizer, dataset, mode, max_code_len=100, max_comment_len=15):
        self.ids = []
        self.code = []
        self.comment = []
        self.exemplar = []
        self.intents = []
        self.bos_id = []

        self.mode = mode
        self.tokenizer = tokenizer
        self.intent2id = {'what': 0, 'why': 1, 'usage': 2, 'done': 3, 'property': 4}
        self.intent2bos_id = {'what': "[WHAT/]", 'why': "[WHY/]", 'usage': "[USAGE/]", 'done': "[DONE/]",
                              'property': "[PROP/]"}
        self.intent2cls_id = {'what': "[/WHAT]", 'why': "[/WHY]", 'usage': "[/USAGE]", 'done': "[/DONE]",
                              'property': "[/PROP]"}

        with open(rf'./dataset/{dataset}/{mode}/code.{mode}', 'r') as f:
            code_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/comment.{mode}', 'r') as f:
            comment_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/comment.similar_{mode}', 'r') as f:
            exemplar_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/label.{mode}', 'r') as f:
            label_lines = f.readlines()

        count_id = 0
        for code_line, comment_line, exemplar_line, label_line in tqdm(
                zip(code_lines, comment_lines, exemplar_lines, label_lines)):
            if code_line.strip() == '':
                continue
            # id & intent
            count_id += 1
            self.ids.append(count_id)
            intent = label_line.strip()
            self.intents.append(self.intent2id[intent])

            # code_token
            self.code.append(self.tokenizer.encode(code_line.strip()).ids[:max_code_len] +
                             [self.tokenizer.token_to_id(self.intent2cls_id[intent])])
            # exemplar
            exemplar_line = json.loads(exemplar_line.strip())[intent]
            self.exemplar.append(self.tokenizer.encode(exemplar_line).ids[:max_comment_len])
            # comment
            if 'test' not in mode:
                self.comment.append(self.tokenizer.encode(comment_line.strip()).ids[:max_comment_len] +
                                    [self.tokenizer.token_to_id('[EOS]')])
            else:
                comment_token_list = comment_line.strip().split(' ')
                self.comment.append(comment_token_list)
            # bos_id
            self.bos_id.append(self.tokenizer.token_to_id(self.intent2bos_id[intent]))

    def __getitem__(self, index):
        return self.code[index], \
               self.exemplar[index], \
               self.comment[index], \
               len(self.code[index]), \
               len(self.exemplar[index]), \
               len(self.comment[index]), \
               self.intents[index], \
               self.bos_id[index], \
               self.ids[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i < 2:
                return_list.append(pad_sequence([torch.tensor(x, dtype=torch.int64) for x in dat[i].tolist()], True))
            elif i == 2:
                if 'test' in self.mode:
                    return_list.append(dat[i].tolist())
                else:
                    return_list.append(pad_sequence([torch.tensor(x) for x in dat[i].tolist()], True))
            elif i < 8:
                return_list.append(torch.tensor(dat[i].tolist()))
            else:
                return_list.append(dat[i].tolist())
        return return_list
