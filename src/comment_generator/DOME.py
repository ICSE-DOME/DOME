import math
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn.functional as F
from sparsemax import Sparsemax
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    if X.dim() == 4:
        b_, s_, t_, d_ = X.size()
        X = X.reshape(b_, s_, t_, num_heads, -1)
        X = X.permute(1, 0, 3, 2, 4)
        return X.reshape(s_, b_ * num_heads, t_, -1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # Shape of output `X`:
        # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        X = X.permute(0, 2, 1, 3)

        # Shape of `output`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def masked_sparsemax(X, valid_lens):
    sparsemax = Sparsemax(dim=-1)
    if X.dim() == 3:
        # b_ * num_heads, q_, s_
        shape = X.shape
        # b * num_heads
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return sparsemax(X.reshape(shape))
    elif X.dim() == 4:
        # s_, b_ * num_heads, q_, t_
        shape = X.shape
        # b_ * num_heads, s_
        valid_lens = torch.repeat_interleave(valid_lens.transpose(0, 1).reshape(-1), shape[2])
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return sparsemax(X.reshape(shape))
    else:
        assert 1 == 2


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    if X.dim() == 4:
        maxlen = X.size(2)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, :, None]
        X[~mask] = value
    else:
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
    return X


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, d_model))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, dropout, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, pos):
        # x -> batch * seq * dim
        # pos -> batch * seq
        x = x + self.pos_embedding(pos)
        return self.dropout(x)


class MultiHeadAttentionWithRPR(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, clipping_distance,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttentionWithRPR, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_rpr = DotProductAttentionWithRPR(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.relative_pos_v = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.relative_pos_k = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.clipping_distance = clipping_distance

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # relative position matrix
        range_queries = torch.arange(queries.size(1), device=queries.device)
        range_keys = torch.arange(keys.size(1), device=keys.device)
        distance_mat = range_keys[None, :] - range_queries[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_distance, self.clipping_distance) + \
                               self.clipping_distance
        # pos_k, pos_v -> seq_q * seq_k * dim
        pos_k = self.relative_pos_k(distance_mat_clipped)
        pos_v = self.relative_pos_v(distance_mat_clipped)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention_rpr(queries, keys, values, pos_k, pos_v, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttentionWithRPR(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttentionWithRPR, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, pos_k, pos_v, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores_pos = torch.bmm(queries.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
        scores = (scores + scores_pos) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        output = torch.bmm(self.dropout(self.attention_weights), values)
        output_pos = torch.bmm(self.dropout(self.attention_weights.transpose(0, 1)), pos_v).transpose(0, 1)
        return output + output_pos


class MultiHeadSelectiveAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, stat_k, token_k, bias=False, **kwargs):
        super(MultiHeadSelectiveAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q_stat = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_q_token = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k_stat = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_k_token = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.stat_k = stat_k
        self.token_k = token_k

    def forward(self, queries, stat_keys, token_keys, values, stat_valid_lens):
        # batch * num_heads, query_num, d_
        query_stat = transpose_qkv(self.W_q_stat(queries), self.num_heads)
        query_token = transpose_qkv(self.W_q_token(queries), self.num_heads)
        # batch * num_heads, stat_num, d_
        key_stat = transpose_qkv(self.W_k_stat(stat_keys), self.num_heads)
        # batch * stat_num * num_heads, token_num, d_
        key_token = transpose_qkv(self.W_k_token(token_keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        b_h_, query_num, d_ = query_stat.size()
        stat_num = key_stat.size(1)
        token_num = key_token.size(1)
        # === stat_weights ===
        # stat_scores -> batch * num_heads, query_num, stat_num
        stat_scores = torch.bmm(query_stat, key_stat.transpose(1, 2)) / math.sqrt(d_)
        if stat_valid_lens.dim() == 1:
            stat_valid_lens = torch.repeat_interleave(stat_valid_lens, self.num_heads * query_num)
            stat_scores = sequence_mask(stat_scores.reshape(-1, stat_num), stat_valid_lens, value=-1e6)
            stat_scores = stat_scores.reshape(b_h_, query_num, stat_num)
        else:
            assert 1 == 2
        stat_k_weights, stat_k_indices = torch.topk(stat_scores, k=min(stat_num - 1, self.stat_k), dim=-1)
        # print(stat_k_indices)
        stat_weights = torch.full_like(stat_scores, -1e6)

        # stat_weights[:, :, :min(stat_num - 1, self.stat_k)] = stat_scores[:, :, :min(stat_num - 1, self.stat_k)]

        stat_weights = stat_weights.scatter(-1, stat_k_indices, stat_k_weights)
        # b_ * num_heads, query_num, stat_num
        stat_weights = torch.softmax(stat_weights, dim=-1)

        # === token_weights ===
        # batch * num_heads * stat_num, query_num, d_
        query_token = torch.repeat_interleave(query_token, stat_num, dim=0)
        # batch * num_heads * stat_num, token_num, d_
        key_token = key_token.reshape(-1, stat_num, self.num_heads, token_num, d_).transpose(1, 2).\
            reshape(b_h_ * stat_num, token_num, d_)
        # token_scores -> batch * num_heads * stat_num, query_num, token_num
        token_scores = torch.bmm(query_token, key_token.transpose(1, 2)) / math.sqrt(d_)
        token_k_weights, token_k_indices = torch.topk(token_scores, k=min(token_num - 1, self.token_k), dim=-1)

        # print(token_k_indices)
        token_weights = torch.full_like(token_scores, -1e6)

        token_weights[:, :, -min(token_num - 1, self.token_k):] = token_scores[:, :, -min(token_num - 1, self.token_k):]
        # token_weights = token_weights.scatter(-1, token_k_indices, token_k_weights)
        # batch * num_heads * stat_num, query_num, token_num
        token_weights = torch.softmax(token_weights, dim=-1)
        # batch * num_heads, query_num, stat_num, token_num
        token_weights = token_weights.reshape(b_h_, stat_num, query_num, token_num).transpose(1, 2)
        # print(token_weights.size(), 'token_weights')

        # batch * num_heads, query_num, stat_num * token_num
        combine_weights = torch.mul(stat_weights.unsqueeze(-1), token_weights).reshape(b_h_, query_num, stat_num * token_num)
        values = values.reshape(-1, stat_num, self.num_heads, token_num, d_).transpose(1, 2).reshape(b_h_, stat_num * token_num, d_)

        output_concat = transpose_output(torch.bmm(combine_weights, values), self.num_heads)
        return self.W_o(output_concat)


class SelectiveAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, stat_k, token_k, bias=False, **kwargs):
        super(SelectiveAttention, self).__init__(**kwargs)
        self.W_q_stat = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_q_token = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k_stat = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_k_token = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.stat_k = stat_k
        self.token_k = token_k

    def forward(self, queries, stat_keys, token_keys, values, stat_valid_lens):
        # batch, query_num, d_
        query_stat = self.W_q_stat(queries)
        query_token = self.W_q_token(queries)
        # batch, stat_num, d_
        key_stat = self.W_k_stat(stat_keys)
        # batch * stat_num, token_num, d_
        key_token = self.W_k_token(token_keys)
        values = self.W_v(values)

        b_, query_num, d_ = query_stat.size()
        stat_num = key_stat.size(1)
        token_num = key_token.size(1)
        # === stat_weights ===
        # stat_scores -> batch, query_num, stat_num
        stat_scores = torch.bmm(query_stat, key_stat.transpose(1, 2)) / math.sqrt(d_)
        if stat_valid_lens.dim() == 1:
            stat_valid_lens = torch.repeat_interleave(stat_valid_lens, query_num)
            stat_scores = sequence_mask(stat_scores.reshape(-1, stat_num), stat_valid_lens, value=-1e6)
            stat_scores = stat_scores.reshape(b_, query_num, stat_num)
        else:
            assert 1 == 2
        stat_k_weights, stat_k_indices = torch.topk(stat_scores, k=min(stat_num - 1, self.stat_k), dim=-1)
        # print(stat_k_indices)
        stat_weights = torch.full_like(stat_scores, -1e6)
        stat_weights = stat_weights.scatter(-1, stat_k_indices, stat_k_weights)
        # b_, query_num, stat_num
        stat_weights = torch.softmax(stat_weights, dim=-1)

        # === token_weights ===
        # batch * stat_num, query_num, d_
        query_token = torch.repeat_interleave(query_token, stat_num, dim=0)
        # token_scores -> batch * num_heads * stat_num, query_num, token_num
        token_scores = torch.bmm(query_token, key_token.transpose(1, 2)) / math.sqrt(d_)
        token_k_weights, token_k_indices = torch.topk(token_scores, k=min(token_num - 1, self.token_k), dim=-1)
        # print(token_k_indices)
        token_weights = torch.full_like(token_scores, -1e6)
        token_weights = token_weights.scatter(-1, token_k_indices, token_k_weights)
        # batch * stat_num, query_num, token_num
        token_weights = torch.softmax(token_weights, dim=-1)
        # batch, query_num, stat_num, token_num
        token_weights = token_weights.reshape(b_, stat_num, query_num, token_num).transpose(1, 2)
        # print(token_weights.size(), 'token_weights')
        # batch, query_num, stat_num * token_num
        combine_weights = torch.mul(stat_weights.unsqueeze(-1), token_weights).reshape(b_, query_num, stat_num * token_num)
        values = values.reshape(b_, stat_num, token_num, d_).reshape(b_, stat_num * token_num, d_)
        return self.W_o(torch.bmm(combine_weights, values))


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        super(EncoderBlockWithRPR, self).__init__()
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class EncoderWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, N=6, dropout=0.1):
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class DecoderBlock_RL(nn.Module):
    def __init__(self, i, d_model, d_intent, d_ff, head_num, stat_k, token_k, dropout=0.1):
        super(DecoderBlock_RL, self).__init__()
        self.i = i
        self.masked_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.selective_attention = SelectiveAttention(d_model + d_intent, d_model, d_model, d_model,
                                                      stat_k, token_k)
        self.cross_attention_exemplar = MultiHeadAttention(d_model + d_intent, d_model, d_model, d_model, head_num, dropout)
        self.gate = nn.Linear(d_model + d_model, 1, bias=False)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, x, state):
        stat_enc, stat_valid_len = state[0], state[1]
        exemplar_enc, example_valid_len = state[3], state[4]
        stat_feature, intent_embed = state[-2], state[-1]
        if state[2][self.i] is None:
            # 训练阶段
            key_values = x
        else:
            # 预测阶段，需要把新预测的词与之前的词拼接
            key_values = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_values
        if self.training:
            # 训练阶段，需要把还未预测到的地方mask
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        x2 = self.masked_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x2)
        y_intent = torch.cat([y, intent_embed.repeat(1, y.size(1), 1)], dim=-1)
        y2_select = self.selective_attention(y_intent, stat_feature, stat_enc, stat_enc, stat_valid_len)
        y2_exemplar = self.cross_attention_exemplar(y_intent, exemplar_enc, exemplar_enc, example_valid_len)
        gate_weight = torch.sigmoid(self.gate(torch.cat([y2_select, y2_exemplar], dim=-1)))
        y2 = gate_weight * y2_select + (1. - gate_weight) * y2_exemplar
        # print(torch.sum(y2_select+y2_exemplar + y))
        # y2 = y2_select + y2_exemplar
        z = self.add_norm2(y, y2 * 2)
        return self.add_norm3(z, self.feedForward(z)), state


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_intent, d_ff, head_num, stat_k, token_k, N=6, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = N
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderBlock_RL(i, d_model, d_intent, d_ff, head_num, stat_k, token_k, dropout) for i in
             range(self.num_layers)])
        self.dense = nn.Linear(d_model + d_intent, vocab_size)

    def init_state(self, stat_enc, stat_valid_len, exemplar_enc, example_valid_len, stat_feature, intent_embed):
        return [stat_enc, stat_valid_len, [None] * self.num_layers, exemplar_enc, example_valid_len,
                stat_feature, intent_embed]

    def forward(self, x, state, intent_embed):
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x, state = layer(x, state)
        return self.dense(torch.cat([x, intent_embed.repeat(1, x.size(1), 1)], dim=-1)), state


class Generator(nn.Module):
    def __init__(self, d_model, d_intent, d_ff, head_num, enc_layer_num, dec_layer_num, vocab_size, max_comment_len,
                 clip_dist_code, eos_token, intent_num, stat_k, token_k, dropout=0.1, beam_width=None):
        super(Generator, self).__init__()
        self.share_embedding = nn.Embedding(vocab_size, d_model)
        self.intent_embedding = nn.Embedding(intent_num, d_intent)
        self.comment_pos_embedding = nn.Embedding(max_comment_len + 2, d_model)

        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clip_dist_code, enc_layer_num, dropout)
        self.exemplar_encoder = Encoder(d_model, d_ff, head_num, enc_layer_num, dropout)
        self.decoder = Decoder(vocab_size, d_model, d_intent, d_ff, head_num, stat_k, token_k, dec_layer_num, dropout)

        self.dropout = nn.Dropout(dropout)
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.max_comment_len = max_comment_len
        self.dec_layer_num = dec_layer_num

    def forward(self, code, exemplar, comment, stat_valid_len, exemplar_valid_len, intent):
        """
        :param code: (batch, token_num)
        :param exemplar:
        :param comment:
        :param stat_valid_len: (batch, )
        :param exemplar_valid_len:
        :param intent:
        :return:
        """
        code_embed = self.dropout(self.share_embedding(code))
        # code_enc -> batch, token_num, d_model
        code_enc = self.code_encoder(code_embed, None)[:, 1:, :]
        # print(code_enc.size())
        b_, _, d_ = code_enc.size()
        # stat_enc -> batch, stat_num, token_num, d_model
        stat_num_inbatch = max(stat_valid_len)
        stat_enc = code_enc.reshape(b_, stat_num_inbatch, -1, d_)
        # stat_feature -> batch, stat_num, d_model
        stat_feature = torch.max(stat_enc, dim=2)[0]

        intent_embed = self.intent_embedding(intent.view(-1, 1))
        b_, r_exemplar = exemplar.size()
        exemplar_pos = torch.arange(1, r_exemplar + 1, device=exemplar.device).repeat(b_, 1)
        exemplar_embed = self.dropout(self.share_embedding(exemplar) + self.comment_pos_embedding(exemplar_pos))
        exemplar_enc = self.exemplar_encoder(exemplar_embed, exemplar_valid_len)

        dec_state = self.decoder.init_state(stat_enc.reshape(b_ * stat_num_inbatch, -1, d_), stat_valid_len,
                                            exemplar_enc, exemplar_valid_len, stat_feature, intent_embed)

        if self.training:
            r_comment = comment.size(1)
            comment_pos = torch.arange(r_comment, device=comment.device).repeat(b_, 1)
            comment_embed = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(comment_pos))
            comment_pred, state = self.decoder(comment_embed, dec_state, intent_embed)
            return comment_pred
        else:
            if self.beam_width is None:
                return self.greed_search(b_, comment, dec_state)
            else:
                return self.beam_search(b_, comment, dec_state, self.beam_width, stat_num_inbatch)

    def greed_search(self, batch_size, comment, dec_state):
        comment_pred = [[-1] for _ in range(batch_size)]
        intent_embed = dec_state[-1]
        for pos_idx in range(self.max_comment_len):
            pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
            tensor, dec_state = self.decoder(comment_embed, dec_state, intent_embed)
            comment = torch.argmax(tensor, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:] for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width, stat_num_inbatch):
        # comment -> batch * 1
        # first node
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}  # 每个时间步都只保留beam_width个node
        # initialization
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            node_dec_state = [dec_state[0][batch_idx * stat_num_inbatch:(batch_idx+1) * stat_num_inbatch],
                              dec_state[1][batch_idx].unsqueeze(0),
                              [None] * self.dec_layer_num,
                              dec_state[3][batch_idx].unsqueeze(0), dec_state[4][batch_idx].unsqueeze(0),
                              dec_state[5][batch_idx].unsqueeze(0), dec_state[6][batch_idx].unsqueeze(0)]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNode(node_list)

        # start beam search
        pos_idx = 0
        while pos_idx < self.max_comment_len:
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                # comment -> batch * 1
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()

                # decode for one step using decoder
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                # comment -> batch * d_model
                comment = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
                tensor, dec_state = self.decoder(comment, dec_state, dec_state[-1])
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prob, comment_candidates -> batch * beam_width
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    node_dec_state = [dec_state[0][batch_idx * stat_num_inbatch:(batch_idx+1) * stat_num_inbatch],
                                      dec_state[1][batch_idx].unsqueeze(0),
                                      [l[batch_idx].unsqueeze(0) for l in dec_state[2]],
                                      dec_state[3][batch_idx].unsqueeze(0), dec_state[4][batch_idx].unsqueeze(0),
                                      dec_state[5][batch_idx].unsqueeze(0), dec_state[6][batch_idx].unsqueeze(0)]
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        # check
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue

                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = log_prob[batch_idx][beam_idx].item()
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1

            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNode(node_list)

            pos_idx += 1
        # the first batchNode in batchNode_dict is the best node
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred

    # batch should have only one example
    def beam_search_oneExample(self, batch_size, comment, dec_state, beam_width):
        assert batch_size == 1
        node_count = 0
        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(dec_state, None, comment, 0, 0)
        nodes_list = [(-node.score, node_count, node)]
        node_count += 1

        pos_idx = 0
        while pos_idx < self.max_comment_len:

            all_nodes_queue = PriorityQueue()

            for idx in range(len(nodes_list)):
                pre_score, _, pre_node = nodes_list[idx]
                comment = pre_node.commentID
                dec_state = pre_node.dec_state
                if pre_node.history_word[-1] == self.eos_token:
                    all_nodes_queue.put((pre_score, _, pre_node))
                    continue

                # decode for one step using decoder
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                # comment -> batch * d_model
                comment = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
                tensor, dec_state = self.decoder(comment, dec_state, dec_state[-1])
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for beam_idx in range(beam_width):
                    node_log_prob = log_prob[0][beam_idx].item()
                    node_comment = comment_candidates[0][beam_idx].view(1, -1)
                    new_node = BeamSearchNode(dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                              pre_node.leng + 1)
                    all_nodes_queue.put((-new_node.score, node_count, new_node))
                    node_count += 1

            range_num = min(beam_width, all_nodes_queue.qsize())
            temp_nodes_list = []
            Flag = True
            for _ in range(range_num):
                cur_score, cur_count, cur_node = all_nodes_queue.get()
                if cur_node.history_word[-1] != self.eos_token:
                    Flag = False
                temp_nodes_list.append((cur_score, cur_count, cur_node))
            nodes_list = temp_nodes_list
            pos_idx += 1
            if Flag:
                break

        # the first node is the best node
        best_score, _, best_node = nodes_list[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred


class BatchNode(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0)]
        if dec_state_list[0][2][0] is None:
            batch_dec_state.append(dec_state_list[0][2])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][2])):
                state_3.append(torch.cat([batch_state[2][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][2])
            batch_dec_state.append(state_3)

        batch_dec_state.extend([torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[5] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[6] for batch_state in dec_state_list], dim=0)])
        return batch_dec_state

    def if_allEOS(self, eos_token):
        for node in self.list_node:
            if node.history_word[-1] != eos_token:
                return False
        return True


class BeamSearchNode(object):
    def __init__(self, dec_state, previousNode, commentID, logProb, length, length_penalty=0.75):
        '''
        :param dec_state:
        :param previousNode:
        :param commentID:
        :param logProb:
        :param length:
        '''
        self.dec_state = dec_state
        self.prevNode = previousNode
        self.commentID = commentID
        self.logp = logProb
        self.leng = length
        self.length_penalty = length_penalty
        if self.prevNode is None:
            self.history_word = [int(commentID)]
            self.score = -100
        else:
            self.history_word = previousNode.history_word + [int(commentID)]
            self.score = self.eval()

    def eval(self):
        return self.logp / self.leng ** self.length_penalty
