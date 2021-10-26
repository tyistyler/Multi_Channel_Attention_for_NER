import os

# dataset name
dataset = "resume"
attn_type = "dot"
seed = 123
# Path of bert model
bert_model = "data/bert-base-chinese"
# Path of the pre-trained word embeddings for getting similar words for each token
tencent_path = "data/tencent_unigram.txt"
pool_method = "first"
# Path of the ZEN model
zen_model = "./data/ZEN_pretrain_base"

log = "log/{}_bert_{}.txt".format(dataset, pool_method)

#       batch_size-32
#       num_layers = 1
os.system("python3 train_zen_cn.py --dataset {} "
          "--seed 123 --bert_model {} --pool_method first --tencent_path {} --zen_model {} "
          "--lr 0.0001 --trans_dropout 0.3 --fc_dropout 0.3 --log {} --batch_size 32 "
          "--num_layers 1 --n_heads 4 --head_dims 64 --use_ngram "
          "--cat_type length --cat_num 5 --ngram_length 5 --multi_att_dropout 0.5 ".format(dataset, bert_model, tencent_path, zen_model, log))
