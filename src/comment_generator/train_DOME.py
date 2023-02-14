import random
import time
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from DOME_dataloader import DOMEDataset
from DOME import Generator
from utils import MaskedSoftmaxCELoss, eval_bleu_rouge_meteor
from tokenizers import Tokenizer
seed = 12345


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_loaders(tokenizer, dataset, batch_size, max_token_inline, max_line_num, max_comment_len,
                num_workers=2, pin_memory=False):
    train_set = DOMEDataset(tokenizer=tokenizer, dataset=dataset, mode='train',
                            max_token_inline=max_token_inline, max_line_num=max_line_num, max_comment_len=max_comment_len)

    test_set = DOMEDataset(tokenizer=tokenizer, dataset=dataset, mode='test',
                           max_token_inline=max_token_inline, max_line_num=max_line_num, max_comment_len=max_comment_len)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_model(model, loss_func, dataloader, optimizer, epoch, cuda):
    losses, ids = [], []
    model.train()

    seed_everything(seed + epoch)
    for data in tqdm(dataloader):
        # clear the grad
        optimizer.zero_grad()

        code, code_valid_len, exemplar, comment, exemplar_valid_len, comment_valid_len, intent, bos = \
            [d.cuda() for d in data[:8]] if cuda else data[:8]
        # code_id = data[-1]

        comment_input = torch.cat([bos.reshape(-1, 1), comment[:, :-1]], 1)
        comment_pred = model(code, exemplar, comment_input, code_valid_len, exemplar_valid_len, intent)
        loss = loss_func(comment_pred, comment, comment_valid_len)
        losses.append(loss.item())
        # ids += code_id
        # accumulate the grad
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        # optimizer the parameters
        optimizer.step()

    avg_loss = round(np.sum(losses) / len(losses), 4)

    return avg_loss


def evaluate_model(model, dataloader, tokenizer, cuda):
    losses, comment_prediction, comment_reference, ids, intents = [], [], [], [], []
    id2intent = ['what', 'why', 'usage', 'done', 'property']
    intent_id = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    intent_pred = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    intent_ref = {'what': [], 'why': [], "usage": [], "done": [], "property": []}
    model.eval()
    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            code, code_valid_len, exemplar = data[0].cuda(), data[1].cuda(), data[2].cuda()
            exemplar_valid_len, comment_valid_len, intent, bos = [data[i].cuda() for i in range(4, 8)]
            comment, code_id = data[3], data[-1]

            bos = bos.reshape(-1, 1)
            comment_pred = model(code, exemplar, bos, code_valid_len, exemplar_valid_len, intent)

            for i in range(len(comment)):
                ref = comment[i]
                comment_reference.append([ref])
                pre = tokenizer.decode(comment_pred[i]).split()
                if not pre:
                    comment_prediction.append(['1'])
                    # print(ref, comment_pred[i])
                else:
                    comment_prediction.append(pre)
            ids += code_id
            intents += intent.tolist()

    assert len(ids) == len(comment_prediction) == len(comment_reference) == len(intents)
    for ii, pp, rr, ll in zip(ids, comment_prediction, comment_reference, intents):
        ll = id2intent[ll]
        intent_id[ll].append(ii)
        intent_pred[ll].append(pp)
        intent_ref[ll].append(rr)
    what_bleu, what_rouge, what_meteor = eval_bleu_rouge_meteor(intent_id['what'], intent_pred['what'], intent_ref['what'])[:3]
    why_bleu, why_rouge, why_meteor = eval_bleu_rouge_meteor(intent_id['why'], intent_pred['why'], intent_ref['why'])[:3]
    usage_bleu, usage_rouge, usage_meteor = eval_bleu_rouge_meteor(intent_id['usage'], intent_pred['usage'], intent_ref['usage'])[:3]
    done_bleu, done_rouge, done_meteor = eval_bleu_rouge_meteor(intent_id['done'], intent_pred['done'], intent_ref['done'])[:3]
    property_bleu, property_rouge, property_meteor = eval_bleu_rouge_meteor(intent_id['property'], intent_pred['property'], intent_ref['property'])[:3]

    return what_bleu, what_rouge, what_meteor, why_bleu, why_rouge, why_meteor, \
           usage_bleu, usage_rouge, usage_meteor, done_bleu, done_rouge, done_meteor, \
           property_bleu, property_rouge, property_meteor, comment_prediction


class Config(object):
    def __init__(self):
        self.cuda = True
        self.dataset = 'tlcodesum'
        self.bpe_model = f'./dataset/{self.dataset}/bpe_tokenizer_all_token.json'
        self.tokenizer = Tokenizer.from_file(self.bpe_model)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_token = self.tokenizer.token_to_id('[EOS]')

        self.d_model = 512
        self.d_intent = 128
        self.d_ff = 2048
        self.head_num = 8
        self.enc_layer_num = 6
        self.dec_layer_num = 6
        # self.max_code_num = 100
        self.max_token_inline = 25
        self.max_line_num = 15
        self.max_comment_len = 30
        self.clip_dist_code = 8
        self.intent_num = 5
        self.stat_k = 5
        self.token_k = 10
        self.beam_width = 5
        self.lr = 1e-4
        self.batch_size = 64
        self.dropout = 0.2
        self.epochs = 100


if __name__ == '__main__':
    config = Config()
    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    seed_everything(seed)

    model = Generator(config.d_model, config.d_intent, config.d_ff, config.head_num, config.enc_layer_num, config.dec_layer_num, config.vocab_size,
                      config.max_comment_len, config.clip_dist_code, config.eos_token,
                      config.intent_num, config.stat_k, config.token_k, config.dropout, None)

    # 模型装载至cuda
    if cuda:
        model.cuda()

    loss_func = MaskedSoftmaxCELoss()

    print(get_parameter_number(model))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    train_loader, test_loader = get_loaders(config.tokenizer, config.dataset, config.batch_size, config.max_token_inline,
                                            config.max_line_num, config.max_comment_len)
    # training stage
    last_improve = 0
    best_test_bleu = 0
    for e in range(config.epochs):
        start_time = time.time()

        train_loss = train_model(model, loss_func, train_loader, optimizer, e, cuda)
        print('epoch:{},train_loss:{},time:{}sec'.format(e + 1, train_loss, round(time.time() - start_time, 2)))

        what_bleu, what_rouge, what_meteor, why_bleu, why_rouge, why_meteor, \
        usage_bleu, usage_rouge, usage_meteor, done_bleu, done_rouge, done_meteor, \
        property_bleu, property_rouge, property_meteor, test_prediction = \
            evaluate_model(model, test_loader, config.tokenizer, cuda)

        avg_bleu, avg_rouge, avg_meteor = (what_bleu+why_bleu+usage_bleu+done_bleu+property_bleu)/5, \
            (what_rouge+why_rouge+usage_rouge+done_rouge+property_rouge)/5,  (what_meteor+why_meteor+usage_meteor+done_meteor+property_meteor)/5
        print('final_results: what_bleu:{},what_rouge:{},what_meteor:{}'.format(what_bleu, what_rouge, what_meteor))
        print('final_results: why_bleu:{},why_rouge:{},why_meteor:{}'.format(why_bleu, why_rouge, why_meteor))
        print('final_results: usage_bleu:{},usage_rouge:{},usage_meteor:{}'.format(usage_bleu, usage_rouge, usage_meteor))
        print('final_results: done_bleu:{},done_rouge:{},done_meteor:{}'.format(done_bleu, done_rouge, done_meteor))
        print('final_results: property_bleu:{},property_rouge:{},property_meteor:{}'.format(property_bleu, property_rouge, property_meteor))
        print('macro_avg_bleu:{}, macro_avg_rouge:{}, macro_avg_meteor:{}'.format(avg_bleu, avg_rouge, avg_meteor))

        if avg_bleu > best_test_bleu:
            best_test_bleu = avg_bleu
            last_improve = e
            # save the best model parameters
            torch.save(model.state_dict(), f"./saved_model/{config.dataset}/comment_generator.pkl")
        print("=========================================================================")

    print("finish!!!")
    print("best_test_bleu:", best_test_bleu)


