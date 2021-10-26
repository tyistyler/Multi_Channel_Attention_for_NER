from models.SANER import SAModel
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler, SequentialSampler
from fastNLP.embeddings import StaticEmbedding, BertEmbedding, StackEmbedding
from modules.pipe import CNNERPipe
from fastNLP.io.pipe.cws import CWSPipe
from fastNLP.core.losses import LossInForward

from run_token_level_classification import BertTokenizer, MSRNgramDict, ZenForTokenClassification, load_examples
from utils_token_level_task import PeopledailyProcessor, CwsmsraProcessor
from ngram_helper import NgramProcessor


import os
import argparse
from modules.callbacks import EvaluateCallback

from datetime import datetime

import random
import torch
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'resume', 'ontonotes', 'msra'])
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--tencent_path', type=str)
parser.add_argument('--bert_model', type=str, required=True)
parser.add_argument('--zen_model', type=str, default="")
parser.add_argument('--pool_method', type=str, default="first", choices=["first", "last", "avg", "max"])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--trans_dropout', type=float, default=0.3)
parser.add_argument('--fc_dropout', type=float, default=0.3)
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dims', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
#---------------new add---------------------------
parser.add_argument("--use_ngram", action='store_true', help="Whether to add ngram.")
parser.add_argument("--cat_type", type=str, default="length", choices=["length", "freq"])
parser.add_argument("--cat_num", type=int, default=5)   # 通道种类数量
parser.add_argument("--ngram_length", type=int, default=5)  # 检测的ngram长度
parser.add_argument('--multi_att_dropout', type=float, default=0.3)

#---------------new add end---------------------------
args = parser.parse_args()

dataset = args.dataset
n_heads = args.n_heads
head_dims = args.head_dims
num_layers = args.num_layers

cat_num = args.cat_num
ngram_length = args.ngram_length
multi_att_dropout = args.multi_att_dropout


lr = args.lr
attn_type = 'adatrans'
n_epochs = 50

pos_embed = None

batch_size = args.batch_size
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True

# dropout=0.3
trans_dropout = args.trans_dropout
fc_dropout = args.fc_dropout

# new_add
tencent_path = args.tencent_path

encoding_type = 'bioes'
ner_name = 'caches/{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])

save_path = "ckpt/elmo_{}_{}_{}_{}_{}.pth".format(dataset, num_layers, n_heads, head_dims, batch_size)

logPath = args.log

def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")

def setup_seed(seed):
    torch.manual_seed(seed)				#为cpu分配随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)	#为gpu分配随机种子
        # torch.cuda.manual_seed_all(seed)#若使用多块gpu，使用该命令设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False

setup_seed(args.seed)

@cache_results(ner_name, _refresh=False)
def load_ner_data():
    paths = {'train': 'data/{}/train.txt'.format(dataset),
             'dev':'data/{}/dev.txt'.format(dataset),
             'test':'data/{}/test.txt'.format(dataset)}
    min_freq = 2
    data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)

    # train_list = data_bundle.get_dataset('train')['raw_chars']

    embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name='data/gigaword_chn.all.a2b.uni.ite50.vec',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    bi_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'),
                               model_dir_or_name='data/gigaword_chn.all.a2b.bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=2,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    tencent_embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                                    model_dir_or_name='data/tencent_unigram.txt',
                                    min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01,
                                    dropout=0.3)

    bert_embed = BertEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=True)

    # embed = StackEmbedding([tencent_embed, bert_embed], dropout=0, word_dropout=0.02)
    embed = StackEmbedding([embed, tencent_embed, bert_embed], dropout=0, word_dropout=0.02)
    return data_bundle, embed, bi_embed

data_bundle, embed, ner_bi_embed = load_ner_data()

print(data_bundle)

gram2id = None
ngram_train_examlpes = None
ngram_dev_examlpes = None
ngram_test_examlpes = None
if args.use_ngram:
    print("[Info] Use Ngram !!! ")
    vocab_path = 'memory_lexicon'
    zen_model_path = args.zen_model

    tokenizer = BertTokenizer.from_pretrained(zen_model_path, do_lower_case=False)
    ngram_dict = MSRNgramDict(vocab_path, tokenizer=tokenizer)
    data_dir = os.path.join("data", dataset)
    max_seq_len = 512

    gram2id = ngram_dict.ngram_to_id_dict
    gram2count = ngram_dict.ngram_to_freq_dict

    print(len(gram2id))
    # f = open('word.txt', 'w', encoding='utf-8')
    # for k,v in gram2id.items():
    #     f.write(str(k) +'\n')
    # f.close()



    processor = NgramProcessor(dataset=dataset, cat_type="length", cat_num=cat_num, ngram_length=ngram_length, gram2id=gram2id, use_ngram=args.use_ngram)
    label_list = processor.get_labels()

    ngram_train_examlpes = load_examples(data_dir, max_seq_len, tokenizer, processor, label_list, mode="train")
    ngram_dev_examlpes = load_examples(data_dir, max_seq_len, tokenizer, processor, label_list, mode="dev")
    ngram_test_examlpes = load_examples(data_dir, max_seq_len, tokenizer, processor, label_list, mode="test")
    # ngram_dev_dataset = None
    # ngram_test_dataset = None
    # print(ngram_train_dataset.size())
    
    print("[Info] Ngram dataset loaded ...")


model = SAModel(tag_vocab=data_bundle.get_vocab('target'),
                embed=embed, num_layers=num_layers,
                d_model=d_model, n_head=n_heads,
                feedforward_dim=dim_feedforward, dropout=trans_dropout,
                after_norm=after_norm, attn_type=attn_type,
                bi_embed=None,
                fc_dropout=fc_dropout,  multi_att_dropout=multi_att_dropout,
                pos_embed=pos_embed,
                scale=attn_type=='naive',
                use_knowledge=False,
                # kv_attn_type=kv_attn_type,
                use_ngram=args.use_ngram,
                gram2id=gram2id, device=device,
                cat_num=cat_num
              )

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
cws_optimizer = None
callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=data_bundle.get_dataset('test'),
                                     use_knowledge=False,
                                     use_ngram=args.use_ngram,
                                     zen_model=None,
                                     ngram_test_examlpes=ngram_test_examlpes,
                                     args=args,
                                     gram2id=None,
                                     device=device,
                                     dataset='test'
                                     )

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer,
                  batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=0, n_epochs=n_epochs, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  use_knowledge=False,
                  logger_func=write_log,
                  use_ngram=args.use_ngram,
                  zen_model=None,
                  ngram_train_examlpes=ngram_train_examlpes,
                  ngram_dev_examlpes=ngram_dev_examlpes,
                  gram2id=None, args=args
                  )

trainer.train(load_best_model=False)

