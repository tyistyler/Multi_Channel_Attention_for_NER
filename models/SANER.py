
import numpy as np
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, AdaptedTransformerEncoder

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from torch.autograd import Function

class MultiChannelAttention(nn.Module):
    def __init__(self, ngram_size, hidden_size, cat_num):
        print("ngram_size:  ", ngram_size)
        super(MultiChannelAttention, self).__init__()
        self.word_embedding = nn.Embedding(ngram_size, hidden_size, padding_idx=0)
        self.channel_weight = nn.Embedding(cat_num, 1)
        self.temper = hidden_size ** 0.5

    def forward(self, word_seq, hidden_state, char_word_mask_matrix, channel_ids):
        # print("word_seq", word_seq)
        # word_seq: (batch_size, channel, word_seq_len)                       [32, 5, 110]
        # hidden_state: (batch_size, character_seq_len, hidden_size)          [32, 29, 128]
        # mask_matrix: (batch_size, channel, character_seq_len, word_seq_len) [32, 5, 29, 110]
        # print("word_seq:    ", word_seq.size())
        # print("hidden_state:    ", hidden_state.size())
        # print("char_word_mask_matrix:    ", char_word_mask_matrix.size())
        # print("channel_ids:    ", channel_ids.size())

        '''
            word_seq = word_ids = ngram_ids,                                [32, 5, 110]
            matching_matrix = [channel, max_seq_length, max_word_size],     [32, 29, 128]
            word_mask = matching_matrix,
            channel_ids = tensor([0,1,2,3,4,5,6,7,8,9])                     [32, 5]
        '''
        # embedding (batch_size, channel, word_seq_len, word_embedding_dim)
        batch_size, character_seq_len, hidden_size = hidden_state.shape
        channel = char_word_mask_matrix.shape[1]
        word_seq_length = word_seq.shape[2]

        embedding = self.word_embedding(word_seq)   # 给ngram编码  [batch_size, channel, word_seq_len, hideen_size]
        # print("embedding:   ", embedding)    # [32, 5, 110, 128]
        tmp = embedding.permute(0, 1, 3, 2)         # [batch_size, channel, ngram_hideen, word_seq_len]
        # print("tmp:   ", tmp)
        tmp_hidden_state = torch.stack([hidden_state] * channel, 1)     # [batch_size, channel, character_seq_len, hidden_size]

        # u (batch_size, channel, character_seq_len, word_seq_len)
        u = torch.matmul(tmp_hidden_state, tmp) / self.temper           # [batch_size, channel, character_seq_len, word_seq_len]

        # attention (batch_size, channel, character_seq_len, word_seq_len)
        tmp_word_mask_metrix = torch.clamp(char_word_mask_matrix, 0, 1) # [batch_size, channel, character_seq_len, word_seq_len]
        '''
            tmp_word_mask_metrix表示跟当前字符相关的ngram位置
        '''
        # print("tmp_word_mask_metrix:",tmp_word_mask_metrix.size())
        exp_u = torch.exp(u)
        # print("exp_u:   ",exp_u.size())
        # print("tmp_word_mask_metrix:   ", tmp_word_mask_metrix.size())
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)            # 只计算相关ngram的向量，其余位置置零，[batch_size, channel, character_seq,_len, word_seq_len]
        # print("delta_exp_u:", delta_exp_u.size())
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 3)] * delta_exp_u.shape[3], 3)    # [batch_size, channel, character_seq, word_seq_len]
        # print("sum_delta_exp_u:", sum_delta_exp_u.size())
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)     # [batch_size, channel, character_seq, word_seq_len]
        # print("attention:", attention.size())
        attention = attention.view(batch_size * channel, character_seq_len, word_seq_length)
        embedding = embedding.view(batch_size * channel, word_seq_length, hidden_size)

        character_attention = torch.bmm(attention, embedding)

        character_attention = character_attention.view(batch_size, channel, character_seq_len, hidden_size)

        channel_w = self.channel_weight(channel_ids)                    # 初始化每个通道的编码 [batch_size, channel, 1]

        channel_w = nn.Softmax(dim=1)(channel_w)                        # 计算每个通道的权重   [batch_size, channel, 1]

        channel_w = channel_w.view(batch_size, -1, 1, 1)                # [batch_size, channel, 1, 1]

        character_attention = torch.mul(character_attention, channel_w) # [batch_size, channel, character_seq_len, hidden_size], 通过广播的方式，给每个通道的ngram分配不同的值
        character_attention = character_attention.permute(0, 2, 1, 3)   # [batch_size, character_seq_len, channel, hidden_size]
        character_attention = character_attention.flatten(start_dim=2)  # [batch_size, character_seq_len, channel * hidden_size]

        return character_attention


class GateConcMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GateConcMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self): # 作用
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, hidden):
        # input: hidden state from encoder;
        # hidden: hidden state from key value memory network
        # output = [gate * input; (1 - gate) * hidden]
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias  # input*w1 + hidden*w2 + bias
        gate = torch.sigmoid(gated)
        # output = torch.add(input.mul(gate), hidden.mul(1 - gate))
        output = torch.cat([input.mul(gate), hidden.mul(1 - gate)],dim=-1)
        return output




class SAModel(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None,
                 use_knowledge=False, multi_att_dropout=0.3,
                 use_ngram=False,
                 gram2id=None, cat_num=5,
                 device=None
                 ):
        """
        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        :param use_knowledge: 是否使用stanford corenlp的知识
        :param feature2count: 字典, {"gram2count": dict, "pos_tag2count": dict, "chunk_tag2count": dict, "dep_tag2count": dict},
        :param
        """
        super().__init__()
        self.use_knowledge = use_knowledge
        self.use_ngram = use_ngram
        self.gram2id = gram2id
        self.embed = embed

        # new add
        self.cat_num = cat_num
        self.use_attention = use_ngram
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        # self.ngram_embeddings = BertWordEmbeddings(hidden_size=embed_size)

        self.in_fc = nn.Linear(embed_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.hidden_size = d_model

        if self.use_attention:
            print("use multi_attention")
            self.multi_attention = MultiChannelAttention(len(self.gram2id), self.hidden_size, self.cat_num)
            self.attention_fc = nn.Linear(self.hidden_size * self.cat_num, self.hidden_size, bias=False)
            self.multi_att_dropout = nn.Dropout(multi_att_dropout)
            self.out_fc = nn.Linear(self.hidden_size*2, len(tag_vocab), bias=False)

            self.gate = GateConcMechanism(hidden_size=self.hidden_size)
            # self.gete_dropout = nn.Dropout(gate_dropout)


        else:
            self.multi_attention = None
            self.out_fc = nn.Linear(self.hidden_size, len(tag_vocab), bias=False)
        # self.out_fc = nn.Linear(d_model, len(tag_vocab))
        # print("len(tag_vocab):  ", len(tag_vocab))
        self.fc_dropout = nn.Dropout(fc_dropout)

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, target, bigrams=None, word_seq=None, word_mask=None, channel_ids=None):
        # NER transformer + shared transformer
        mask = chars.ne(0)

        hidden = self.embed(chars)
        hidden = self.in_fc(hidden)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            hidden = torch.cat([hidden, bigrams], dim=-1)


        # Transformer
        hidden = self.transformer(hidden, mask)
        # hidden = self.trans_dropout(hidden)
        if self.multi_attention is not None:
            '''
                word_seq = word_ids = ngram_ids,
                matching_matrix = [channel, max_seq_length, max_word_size],
                word_mask = matching_matrix,
                channel_ids = tensor([0,1,2,3,4,5,6,7,8,9])
            '''

            attention_output = self.multi_attention(word_seq, hidden, word_mask, channel_ids)
            attention_output = self.multi_att_dropout(attention_output)
            attention_output = self.attention_fc(attention_output)

            # add-gate
            hidden = self.gate(hidden, attention_output)

            # add-directly
            # hidden = torch.cat([hidden, attention_output], dim=2)
            # print("hidden:  ", hidden.size())

        hidden = self.fc_dropout(hidden)
        encoder_output = self.out_fc(hidden)
        logits = F.log_softmax(encoder_output, dim=-1)

        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}

        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}


    def forward(self, chars, target, bigrams=None, word_seq=None, word_mask=None, channel_ids=None):
        return self._forward(chars, target, bigrams, word_seq, word_mask, channel_ids)

    def predict(self, chars, bigrams=None, word_seq=None, word_mask=None, channel_ids=None):
        return self._forward(chars, target=None, bigrams=bigrams, word_seq=word_seq, word_mask=word_mask, channel_ids=channel_ids)
