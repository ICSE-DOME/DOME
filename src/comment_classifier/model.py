import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
import numpy as np


class commentClassifier(nn.Module):
    def __init__(self, pretrained_model, class_num, dropout):
        super(commentClassifier, self).__init__()
        self.codeBert = AutoModel.from_pretrained(pretrained_model)
        self.fc1 = nn.Linear(768 + 1, 768//3)
        self.fc2 = nn.Linear(self.fc1.out_features, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, att_mask, comment_len, punc_num):
        cls_embed = self.codeBert(input_ids=input_ids, attention_mask=att_mask)[1]
        # comment_len/punc_num -> (batch_size, 1)
        comment_len = comment_len.view(-1, 1).float()
        # punc_num = punc_num.view(-1, 1).float()
        cls_embed = torch.cat([cls_embed, comment_len], dim=-1)
        logits = self.fc2(self.dropout(F.relu(self.fc1(self.dropout(cls_embed)))))
        return logits
