import json
import string
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class commentDataset(Dataset):
    def __init__(self, file_address, tokenizer_address, max_code_length=200, max_comment_length=50):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_address)
        self.max_code_length = max_code_length
        self.max_comment_length = max_comment_length
        with open(file_address, 'r') as r:
            data_dict = json.load(r)

        self.ids = data_dict['id']
        self.codes = data_dict['code']
        self.comments = data_dict['comment']
        self.labels = []  # ['what', 'why', 'usage', 'done', 'property', 'others']
        for l in data_dict['label']:
            if l == 'what':
                self.labels.append(0)
            elif l == 'why':
                self.labels.append(1)
            elif l == 'how-to-use':
                self.labels.append(2)
            elif l.strip() == 'how-it-is-done':
                self.labels.append(3)
            elif l == 'property':
                self.labels.append(4)
            elif l == 'others':
                self.labels.append(5)
            else:
                print(l)
                assert 1 == 2

        self.cc_pairs = []
        self.cc_mask = []
        self.comment_len = []
        self.punc_num = []
        assert len(self.ids) == len(self.codes) == len(self.comments) == len(self.labels)

        # # codeBert tokenization
        # for code, comment in zip(self.codes, self.comments):
        #     code_tokens = self.tokenizer.tokenize(code)[:max_code_length]
        #     comment_tokens = self.tokenizer.tokenizerr(comment)[:max_comment_length]
        #     cc_tokens = [self.tokenizer.cls_token]+comment_tokens+[self.tokenizer.sep_token]+\
        #                 code_tokens+[self.tokenizer.sep_token]
        #     self.cc_pairs.append(self.tokenizer.convert_tokens_to_ids(cc_tokens))
        #     self.cc_mask.append([1] * len(cc_tokens))

        # codeBert tokenization
        for comment in self.comments:
            comment_tokens = self.tokenizer.tokenize(comment)[:max_comment_length]
            cc_tokens = [self.tokenizer.cls_token] + comment_tokens + [self.tokenizer.sep_token]
            self.cc_pairs.append(self.tokenizer.convert_tokens_to_ids(cc_tokens))
            self.cc_mask.append([1] * len(cc_tokens))
            if len(comment.strip().split()) < 3:
                self.comment_len.append(1)
            else:
                self.comment_len.append(0)
            # self.comment_len.append(len(comment.strip().split()) / max_comment_length)
            self.punc_num.append(self.count_punc_num(comment, len(comment.strip().split())))
        # print(self.comment_len)
        # print(self.punc_num)

    def count_punc_num(self, comment, comment_len):
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        punc_num = count(comment, set(string.punctuation))
        digits_num = count(comment, set(string.digits))
        return (punc_num + digits_num) / comment_len

    def __getitem__(self, index):

        return self.cc_pairs[index], \
               self.cc_mask[index], \
               self.labels[index], \
               self.comment_len[index], \
               self.punc_num[index], \
               self.ids[index], \
               self.comments[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i == 0:
                return_list.append(pad_sequence([torch.tensor(x) for x in dat[i].tolist()], batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id))
            elif i == 1:
                return_list.append(
                    pad_sequence([torch.tensor(x) for x in dat[i].tolist()], batch_first=True, padding_value=0))
            elif i < 5:
                return_list.append(torch.tensor(dat[i]))
            else:
                return_list.append(dat[i].tolist())
        return return_list


class predictionDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer_address, max_code_length=200, max_comment_length=50):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_address)

        self.ids = []
        self.codes = []
        self.comments = []
        self.cc_pairs = []
        self.cc_mask = []
        self.comment_len = []
        self.punc_num = []
        with open(f'./dataset/clean/{dataset}/{mode}/{dataset}.{mode}', 'r') as f:
            json_data = f.readlines()

        for json_line in json_data:
            data_dict = json.loads(json_line.strip())
            self.ids.append(data_dict['id'])
            self.codes.append(data_dict['code'].strip())
            self.comments.append(data_dict['comment'].strip())

        assert len(self.ids) == len(self.codes) == len(self.comments)

        # # codeBert tokenization
        # for code, comment in zip(self.codes, self.comments):
        #     code_tokens = self.tokenizer.tokenize(code)[:max_code_length]
        #     comment_tokens = self.tokenizer.tokenize(comment)[:max_comment_length]
        #     cc_tokens = [self.tokenizer.cls_token]+comment_tokens+[self.tokenizer.sep_token]+\
        #                 code_tokens+[self.tokenizer.sep_token]
        #     self.cc_pairs.append(self.tokenizer.convert_tokens_to_ids(cc_tokens))
        #     self.cc_mask.append([1] * len(cc_tokens))

        # codeBert tokenization
        for comment in self.comments:
            comment_tokens = self.tokenizer.tokenize(comment)[:max_comment_length]
            cc_tokens = [self.tokenizer.cls_token] + comment_tokens + [self.tokenizer.sep_token]
            self.cc_pairs.append(self.tokenizer.convert_tokens_to_ids(cc_tokens))
            self.cc_mask.append([1] * len(cc_tokens))
            if len(comment.strip().split()) < 3:
                self.comment_len.append(1)
            else:
                self.comment_len.append(0)
            # self.comment_len.append(len(comment.strip().split()) / max_comment_length)
            self.punc_num.append(self.count_punc_num(comment, len(comment.strip().split())))

    def count_punc_num(self, comment, comment_len):
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        punc_num = count(comment, set(string.punctuation))
        digits_num = count(comment, set(string.digits))
        return (punc_num + digits_num) / comment_len

    def __getitem__(self, index):

        return self.cc_pairs[index], \
               self.cc_mask[index], \
               self.comment_len[index], \
               self.punc_num[index], \
               self.ids[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i == 0:
                return_list.append(pad_sequence([torch.tensor(x) for x in dat[i].tolist()], batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id))
            elif i == 1:
                return_list.append(
                    pad_sequence([torch.tensor(x) for x in dat[i].tolist()], batch_first=True, padding_value=0))
            elif i < 4:
                return_list.append(torch.tensor(dat[i]))
            else:
                return_list.append(dat[i].tolist())
        return return_list
