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
from dataloader import DOMEDataset
from model import Generator
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


def get_loaders(tokenizer, dataset, batch_size=32, num_workers=2, pin_memory=False):
    train_set = DOMEDataset(tokenizer=tokenizer, dataset=dataset, mode='valid')

    test_set = DOMEDataset(tokenizer=tokenizer, dataset=dataset, mode='test')

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

        code, exemplar, comment, code_valid_len, exemplar_valid_len, comment_valid_len, intent, bos = \
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
            code, exemplar = data[0].cuda(), data[1].cuda()
            code_valid_len, exemplar_valid_len, comment_valid_len, intent, bos = [data[i].cuda() for i in range(3, 8)]
            comment, code_id = data[2], data[-1]

            bos = bos.reshape(-1, 1)
            comment_pred = model(code, exemplar, bos, code_valid_len, exemplar_valid_len, intent)

            for i in range(len(comment)):
                ref = comment[i]
                comment_reference.append([ref])
                pre = tokenizer.decode(comment_pred[i]).split()
                if not pre:
                    comment_prediction.append(['1'])
                    print(ref, comment_pred[i])
                else:
                    comment_prediction.append(pre)
            ids += code_id
            intents += intent.tolist()

    assert len(ids) == len(comment_prediction) == len(comment_reference) == len(intents)
    avg_bleu, avg_rouge, avg_meteor = eval_bleu_rouge_meteor(ids, comment_prediction, comment_reference)[:3]
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

    return avg_bleu, avg_rouge, avg_meteor, what_bleu, what_rouge, what_meteor, why_bleu, why_rouge, why_meteor, \
           usage_bleu, usage_rouge, usage_meteor, done_bleu, done_rouge, done_meteor, \
           property_bleu, property_rouge, property_meteor, comment_prediction


class Config(object):
    def __init__(self):
        self.cuda = True
        self.dataset = 'funcom'
        self.bpe_model = f'./dataset/{self.dataset}/bpe_tokenizer_all_token.json'
        self.tokenizer = Tokenizer.from_file(self.bpe_model)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_token = self.tokenizer.token_to_id('[EOS]')

        self.d_model = 512
        self.d_intent = 256
        self.d_ff = 2048
        self.head_num = 8
        self.layer_num = 4
        self.max_code_num = 100
        self.max_token_num = 20
        self.max_stat_num = 10
        self.max_comment_len = 15
        self.clip_dist_code = 8
        self.clip_dist_stat = 8
        self.intent_num = 5
        self.beam_width = 4
        self.lr = 1e-4
        self.batch_size = 128
        self.dropout = 0.2
        self.epochs = 20


if __name__ == '__main__':
    config = Config()
    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    seed_everything(seed)

    model = Generator(config.d_model, config.d_intent, config.d_ff, config.head_num, config.layer_num, config.vocab_size,
                      config.max_stat_num, config.max_comment_len, config.clip_dist_code, config.clip_dist_stat,
                      config.eos_token, config.intent_num, config.dropout, None)

    print("load the parameters of the pretrained generator!")
    model.load_state_dict(torch.load(f"./saved_model/{config.dataset}/dome_parameters.pkl"))

    # 模型装载至cuda
    if cuda:
        model.cuda()

    loss_func = MaskedSoftmaxCELoss()

    print(get_parameter_number(model))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    train_loader, test_loader = get_loaders(config.tokenizer, config.dataset, config.batch_size)
    # training stage
    last_improve = 0
    best_test_bleu = 0
    for e in range(config.epochs):
        start_time = time.time()

        # train_loss = train_model(model, loss_func, train_loader, optimizer, e, cuda)
        # print('epoch:{},train_loss:{},time:{}sec'.format(e + 1, train_loss, round(time.time() - start_time, 2)))

        if (e+1) % 5 == 0 or e >= 0:
            test_bleu, test_rouge, test_meteor, what_bleu, what_rouge, what_meteor, why_bleu, why_rouge, why_meteor, \
            usage_bleu, usage_rouge, usage_meteor, done_bleu, done_rouge, done_meteor, \
            property_bleu, property_rouge, property_meteor, test_prediction = \
                evaluate_model(model, test_loader, config.tokenizer, cuda)

            print('final_results: avg_bleu:{},avg_rouge:{},avg_meteor:{}'.format(test_bleu, test_rouge, test_meteor))
            print('final_results: what_bleu:{},what_rouge:{},what_meteor:{}'.format(what_bleu, what_rouge, what_meteor))
            print('final_results: why_bleu:{},why_rouge:{},why_meteor:{}'.format(why_bleu, why_rouge, why_meteor))
            print('final_results: usage_bleu:{},usage_rouge:{},usage_meteor:{}'.format(usage_bleu, usage_rouge, usage_meteor))
            print('final_results: done_bleu:{},done_rouge:{},done_meteor:{}'.format(done_bleu, done_rouge, done_meteor))
            print('final_results: property_bleu:{},property_rouge:{},property_meteor:{}'.format(property_bleu, property_rouge, property_meteor))
            assert 1==2
            if test_bleu > best_test_bleu:
                best_test_bleu = test_bleu
                last_improve = e
                # save the best model parameters
                torch.save(model.state_dict(), f"./saved_model/{config.dataset}/dome_parameters.pkl")
                with open(f'./results/{config.dataset}/pretrain_gen', 'w') as w:
                    for comment_list in test_prediction:
                        comment = ' '.join(comment_list)
                        w.write(comment + '\n')
        print("=========================================================================")

        # if e - last_improve >= 20:
        #     print("No optimization for 20 epochs, auto-stopping and save model parameters")
        #     break

    print("finish!!!")
    # print("best_valid_bleu:", best_valid_bleu)
    print("best_test_bleu:", best_test_bleu)


