import random

import torch
import time
from torch import nn
from collections import OrderedDict, Counter
from src.comment_generator.eval.bleu import corpus_bleu
from src.comment_generator.eval.rouge import Rouge
from src.comment_generator.eval.meteor import Meteor
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math
import numpy as np


def get_statement_mask(token_num_batch, indices):
    """
    :param token_num_batch: list of the length of each statement
    :param indices: list of lists of statement indices
    :return:
    """
    statement_mask = []
    for x, y in zip(token_num_batch, indices):
        y = y.tolist()
        y.append(len(x) - 1)  # the last special tag is seen as the selected statement
        temp_mask = []
        for idx, xx in enumerate(x):
            if idx in y:
                temp_mask += [1] * xx
            else:
                temp_mask += [0] * xx

        statement_mask.append(torch.tensor(temp_mask))

    return pad_sequence(statement_mask, True)


def get_statement(statement_list, indices, flag):
    """
    :param statement_list: list of lists of the statement tokens
    :param indices: list of lists of statement indices
    :return:
    """
    # print(statement_list, indices)
    statement, statement_lens = [], []
    if flag == 'sample':
        for stat_each_code, idx_each_code in zip(statement_list, indices):
            idx_each_code = idx_each_code.tolist()
            temp_statement = []
            if idx_each_code:
                for idx in range(len(stat_each_code)):
                    if idx in idx_each_code:
                        temp_statement += stat_each_code[idx]
            else:
                temp_statement = stat_each_code[-1]

            statement.append(temp_statement)
            statement_lens.append(len(statement[-1]))
    elif flag == 'select':
        for stat_each_code, idx_each_code in zip(statement_list, indices):
            temp_statement = []
            for idx in range(len(stat_each_code)):
                if idx_each_code[idx] == 1:
                    # print(idx, idx_each_code)
                    temp_statement += stat_each_code[idx]
            if not temp_statement:
                temp_statement = stat_each_code[-1]

            statement.append(temp_statement)
            statement_lens.append(len(statement[-1]))
    else:
        assert 1 == 2
    # print(statement, statement_lens)
    statement = pad_sequence([torch.tensor(i) for i in statement], True).cuda()
    statement_lens = torch.tensor(statement_lens).cuda()

    return statement, statement_lens


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def get_bleu_score(ids, comment_pred, comment):
    """An unofficial evalutation helper.
     Arguments:
        ids: list: list of id for the reference comments
        comment_pred: list: list of tokens for the prediction comments
        comment: list: list of tokens for the reference comments
    """
    assert len(ids) == len(comment_pred) == len(comment)

    _, bleu, ind_bleu = corpus_bleu(ids, comment_pred, comment)

    return bleu


def eval_bleu_rouge_meteor(ids, comment_pred, comment):
    """An unofficial evalutation helper.
     Arguments:
        ids: list: list of id for the reference comments
        comment_pred: list: list of tokens for the prediction comments
        comment: list: list of tokens for the reference comments
    """
    assert len(ids) == len(comment_pred) == len(comment)

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(ids, comment_pred, comment)
    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(ids, comment_pred, comment)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(ids, comment_pred, comment)

    return bleu * 100, rouge_l * 100, meteor * 100, ind_bleu, ind_rouge


def bleu_score_sentence1(select_pred, gold, eos_id):
    """
    :param select_pred: list of lists
    :param gold: list of lists
    :param eos_id: eos index in vocab
    :return:
    """

    def clean_up_sentence(sent):
        if eos_id in sent:
            sent = sent[:sent.index(eos_id)]
        if not sent:
            sent = [0]
        return sent

    bleu_score_diff = []
    # min_value = 100
    for select_pp, gg in zip(select_pred, gold):
        select_pp = clean_up_sentence(select_pp)
        gg = clean_up_sentence(gg)
        # print(gg)
        # print(select_pp, sample_pp)
        select_score = get_bleu_score([0], [select_pp], [[gg]])
        score = select_score
        # min_value = min(score, min_value)
        bleu_score_diff.append(score)

    bleu_score_diff = torch.tensor(bleu_score_diff).cuda()
    # bleu_score_diff = (bleu_score_diff - bleu_score_diff.mean()) / (
    #         bleu_score_diff.std() + np.finfo(np.float32).eps.item())
    return bleu_score_diff


def bleu_score_sentence(select_pred, sample_pred, gold, eos_id):
    """
    :param select_pred: list of lists
    :param sample_pred: list of lists
    :param gold: list of lists
    :param eos_id: eos index in vocab
    :return:
    """

    def clean_up_sentence(sent):
        if eos_id in sent:
            sent = sent[:sent.index(eos_id)]
        if not sent:
            sent = [0]
        return sent

    bleu_score_diff = []
    # max_value, min_value = -100, 100
    for select_pp, sample_pp, gg in zip(select_pred, sample_pred, gold):
        select_pp = clean_up_sentence(select_pp)
        sample_pp = clean_up_sentence(sample_pp)
        gg = clean_up_sentence(gg)
        # print(gg)
        # print(select_pp, sample_pp)
        select_score = get_bleu_score([0], [select_pp], [[gg]])
        sample_score = get_bleu_score([0], [sample_pp], [[gg]])
        score = sample_score - select_score
        # max_value = max(score, max_value)
        # min_value = min(score, min_value)
        bleu_score_diff.append(score)
    # print(bleu_score_diff)
    # as
    # bleu_score_diff = [(i - min_value) / (max_value - min_value + 1e-8) - 0.5 for i in bleu_score_diff]
    bleu_score_diff = torch.tensor(bleu_score_diff).cuda()
    # bleu_score_diff = (bleu_score_diff - bleu_score_diff.mean()) / (bleu_score_diff.std() + np.finfo(np.float32).eps.item())
    return bleu_score_diff


def defined_reward_diff(pred_logits, comment, comment_valid_len):
    """
    :param pred_logits: batch, num_steps, vocab_num
    :param comment: batch, num_steps
    :param comment_valid_len: batch,
    :return: batch,
    """
    pred_logits_copy = pred_logits.clone().detach()
    pred_logprobs = torch.log_softmax(pred_logits_copy, -1)
    # batch, num_steps
    comment_logprobs = torch.gather(pred_logprobs, -1, comment.unsqueeze(-1)).squeeze(-1)
    top2_logprobs, top2_indices = torch.topk(pred_logprobs, 2, dim=-1)
    max_logprobs, max2_logprobs = top2_logprobs[:, :, 0], top2_logprobs[:, :, 1]
    first_item = max_logprobs - comment_logprobs
    second_item = max_logprobs / (torch.exp(max_logprobs) - torch.exp(max2_logprobs))
    reward = first_item - second_item
    reward = sequence_mask(reward, comment_valid_len)
    reward = torch.exp(-torch.mean(reward, dim=-1))
    return reward


# implement by "Towards automatically generating block comments for code snippets"
def score_sentence(pred, gold, ngrams, smooth=1e-5):
    scores = []
    # Get ngrams count for gold.
    count_gold = defaultdict(int)
    _update_ngrams_count(gold, ngrams, count_gold)
    # Init ngrams count for pred to 0.
    count_pred = defaultdict(int)
    # p[n][0] stores the number of overlapped n-grams.
    # p[n][1] is total # of n-grams in pred.
    p = []
    for n in range(ngrams + 1):
        p.append([0, 0])
    for i in range(len(pred)):
        for n in range(1, ngrams + 1):
            if i - n + 1 < 0:
                continue
            # n-gram is from i - n + 1 to i.
            ngram = tuple(pred[(i - n + 1): (i + 1)])
            # Update n-gram count.
            count_pred[ngram] += 1
            # Update p[n].
            p[n][1] += 1
            if count_pred[ngram] <= count_gold[ngram]:
                p[n][0] += 1
        scores.append(_compute_bleu(p, i + 1, len(gold), smooth))
    return scores


def score_corpus(preds, golds, ngrams, smooth=1e-5):
    golds = [ref for refs in golds for ref in refs]
    assert len(preds) == len(golds)
    p = []
    for n in range(ngrams + 1):
        p.append([0, 0])
    len_pred = len_gold = 0
    for pred, gold in zip(preds, golds):
        len_gold += len(gold)
        count_gold = defaultdict(int)
        _update_ngrams_count(gold, ngrams, count_gold)

        len_pred += len(pred)
        count_pred = defaultdict(int)
        _update_ngrams_count(pred, ngrams, count_pred)

        for k, v in count_pred.items():
            n = len(k)
            p[n][0] += min(v, count_gold[k])
            p[n][1] += v

    return _compute_bleu(p, len_pred, len_gold, smooth)


def _update_ngrams_count(sent, ngrams, count):
    length = len(sent)
    for n in range(1, ngrams + 1):
        for i in range(length - n + 1):
            ngram = tuple(sent[i: (i + n)])
            count[ngram] += 1


def _compute_bleu(p, len_pred, len_gold, smooth):
    # Brevity penalty.
    log_brevity = 1 - max(1, (len_gold + smooth) / (len_pred + smooth))
    log_score = 0
    ngrams = len(p) - 1
    for n in range(1, ngrams + 1):
        if p[n][1] > 0:
            if p[n][0] == 0:
                p[n][0] = 1e-16
            log_precision = math.log((p[n][0] + smooth) / (p[n][1] + smooth))
            log_score += log_precision
    log_score /= ngrams
    return math.exp(log_score + log_brevity)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len, average=True):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss.mean() if average else weighted_loss.sum()


class MaskedBCELoss(nn.BCELoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""

    # `pred` shape: (`batch_size`, `stat_num`)
    # `label` shape: (`batch_size`, `stat_num`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len, average=True):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, label.float())
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss.mean() if average else weighted_loss.sum()


def selectorReinforcedLoss(reward, statement_probs, sample_indices, sample_valid_num):
    """
    :param reward: (batch, )
    :param statement_probs: (batch, statement_num)
    :param sample_indices: tensor
    :param sample_valid_num: tensor
    :param average: bool
    :return:
    """
    sample_indices = pad_sequence(sample_indices, batch_first=True)
    mask = torch.ones_like(sample_indices)
    mask = sequence_mask(mask, sample_valid_num)
    log_dist = torch.log(statement_probs)
    losses = -log_dist.gather(1, sample_indices) * mask
    losses = torch.sum(losses, dim=-1)
    losses = (losses * reward).mean()
    # entropy bonus
    H = statement_probs * log_dist
    entropy = H.gather(1, sample_indices) * mask
    entropy = -1 * torch.mean(entropy)
    entropy_bonus = -1 * 0.1 * entropy
    print(losses)
    print(entropy_bonus)
    return losses + entropy_bonus


# def selectorReinforcedLoss(reward, statement_probs, sample_indices, sample_valid_num):
#     """
#     :param reward: (batch, )
#     :param statement_probs: (batch, statement_num)
#     :param sample_indices: tensor
#     :param sample_valid_num: tensor
#     :param average: bool
#     :return:
#     """
#     sample_indices = pad_sequence(sample_indices, batch_first=True)
#     mask = torch.ones_like(sample_indices)
#     mask = sequence_mask(mask, sample_valid_num)
#     log_dist = torch.log(statement_probs)
#     losses = -log_dist.gather(1, sample_indices) * mask
#     # print(losses)
#     # assert 1==2
#     losses = torch.sum(losses, dim=-1)
#     losses = (losses * reward).mean()
#     # entropy bonus
#     H = statement_probs * log_dist
#     entropy = H.gather(1, sample_indices) * mask
#     # print(H)
#     # print(H.size())
#     # assert 1==2
#     # # log_p = log_softmax(statement_probs, dim=-1)
#     entropy = torch.sum(entropy, dim=-1) # * mask
#     entropy = -1 * torch.mean(entropy)
#     entropy_bonus = -1 * 0.1 * entropy
#     print(losses)
#     print(entropy_bonus)
#     return losses + entropy_bonus
#
#     # # ================================== 1 ==================
#     # # sample_indices -> batch, statement_num
#     # # sample_indices = pad_sequence(sample_indices, batch_first=True)
#     # # # print(sample_indices)
#     # # # sample_valid_num -> batch,
#     # # # sample_valid_num = torch.tensor(sample_valid_num, device=sample_indices.device)
#     # # mask = torch.ones_like(sample_indices)
#     # # mask = sequence_mask(mask, sample_valid_num)
#     # log_dist = torch.log(statement_probs)
#     # # print(list(-log_dist[:10]))
#     # losses = -log_dist.gather(1, sample_indices) #* mask
#     # losses = torch.sum(losses, dim=-1)
#     # # print(losses)
#     # # print(reward)
#     # losses = losses * reward
#     # # print(losses)
#     # # assert 1==2
#     # return losses.mean() if average else losses.sum()


def generatorReinforcedLoss(reward, comment_logits, comment_pred, comment_valid_num, average=True):
    """
    :param reward: (batch, )
    :param comment_logits: (batch, num_steps, vocab)
    :param comment_pred: (batch, num_steps)
    :param comment_valid_num: (batch, )
    :param average: bool
    :return:
    """
    mask = torch.ones_like(comment_pred)
    mask = sequence_mask(mask, comment_valid_num)

    log_dist = F.log_softmax(comment_logits, -1)
    losses = -log_dist.gather(2, comment_pred.unsqueeze(2)).squeeze(2) * mask
    losses = torch.sum(losses, dim=-1)
    losses = losses * reward
    return losses.mean() if average else losses.sum()


if __name__ == '__main__':
    # hypothesis = [292, 255, 255, 676, 12384, 891, 2727, 593, 255, 3564, 926, 292, 168, 255, 733, 168, 12]
    # reference = [292, 255, 733, 719, 2084, 280, 3617, 2465, 255, 2467, 258, 7170, 940, 1670, 1249, 168, 12]
    # references = [reference]  # list of references for 1 sentence.
    # # list_of_references = [references]  # list of references for all sentences in corpus.
    # # list_of_hypotheses = [hypothesis]  # list of hypotheses that corresponds to list of references.
    # # score = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
    # # print(score)
    # score = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.8, 0.2, 0, 0))
    # print(score)
    import json
    from tqdm import tqdm

    # dataset = 'funcom'
    # file = 'test'
    with open(f'./nl.original', 'r') as f:
        comment = f.readlines()
    # with open(f'./dataset/{dataset}/{file}/label.{file}', 'r') as f:
    #     label = f.readlines()
    # with open(f'./dataset/{dataset}/{file}/comment.similar_{file}', 'r') as f:
    #     similar_comment = f.readlines()
    with open(f'./pred.txt', 'r') as f:
        results = f.readlines()

    ids, reference, prediction = [], [], []
    count = 0
    for ref, pred in zip(comment, results):
        ids.append(count)
        count += 1
        reference.append([ref.strip().split()])
        prediction.append(pred.strip().split())
        print(prediction[-1])

    # ids = []
    # reference = []
    # prediction = []
    # count = 0
    # for com, sim_com, lab, result in tqdm(zip(comment, similar_comment, label, results)):
    #     lab = lab.strip()
    #     sim_com = json.loads(sim_com.strip())[lab]
    #     if len(sim_com.split()) == 0:
    #         continue
    #     reference.append([sim_com.split()])
    #     ids.append(count)
    #     count += 1
    #     # com = com.strip()
    #     # reference.append([com.split()])
    #     result = result.strip()
    #     prediction.append(result.split())

    print(eval_bleu_rouge_meteor(ids, prediction, reference)[:2])
