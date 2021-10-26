# coding: utf-8
# Name:     ngram_helper
# Author:   dell
# Data:     2021/3/25

import os
import math
import torch
import numpy as np

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file, encoding="utf-8")
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data


class NgramProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    def __init__(self, dataset, cat_type="length", cat_num=5, ngram_length=5, gram2id=None, gram2count=None, use_ngram=False):
        super(NgramProcessor, self).__init__()
        self.dataset = dataset.lower()
        self.cat_type = cat_type
        self.cat_num = cat_num
        self.ngram_length = ngram_length
        self.gram2id = gram2id
        self.gram2count = gram2count
        self.multi_attention = use_ngram

        self.zen = None
        self.zen_ngram_dict = None

    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    #
    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
    #
    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        if "ontonote" in self.dataset:
            return ["O", "I-LOC", "B-ORG", "I-PER", "I-GPE", "B-LOC", "B-GPE", "B-PER", "I-ORG", "[CLS]", "[SEP]"]
        elif "weibo" in self.dataset:
            return ["O", "B-PER.NOM", "I-PER.NOM", "B-LOC.NAM", "I-LOC.NAM", "B-PER.NAM", "I-PER.NAM",
                    "B-GPE.NAM", "I-GPE.NAM", "B-ORG.NAM", "I-ORG.NAM", "B-ORG.NOM", "I-ORG.NOM",
                    "B-LOC.NOM", "I-LOC.NOM", "B-GPE.NOM", "I-GPE.NOM", "[CLS]", "[SEP]"]
        elif "resume" in self.dataset:
            return ["O", "I-ORG", "I-RACE", "I-PRO", "I-NAME", "B-RACE", "B-ORG", "I-LOC", "I-TITLE", "I-EDU", "B-LOC",
                    "B-TITLE", "B-CONT", "B-NAME", "I-CONT", "B-PRO", "B-EDU", "[CLS]", "[SEP]"]
        else:
            raise ValueError("dataset can not be {}".format(self.dataset))

    def load_tsv_data(self, data_path, flag="train"):

        # flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        sentence_list, label_list = self.read_ngram_tsv(data_path)

        data = []
        # print(len(self.gram2id))
        for sentence, label in zip(sentence_list, label_list):
            # print(sentence)
            # print(label)
            if self.multi_attention is not None:    # 为所有ngram分配不同的channel
                ngram_list = []
                matching_position = []
                ngram_list_len = []
                for i in range(self.cat_num):   # categories = 10
                    ngram_list.append([])
                    matching_position.append([])
                    ngram_list_len.append(0)
                for i in range(len(sentence)):
                    for j in range(0, self.ngram_length):
                        if i + j + 1 > len(sentence):
                            break
                        ngram = ''.join(sentence[i: i + j + 1])
                        ngram_ = tuple(ngram)
                        if ngram_ in self.gram2id:
                            channel_index = self._ngram_category(ngram)         # "深圳" len=2, ngram_category=1
                            try:
                                index = ngram_list[channel_index].index(ngram)
                            except ValueError:
                                ngram_list[channel_index].append(ngram)
                                index = len(ngram_list[channel_index]) - 1      # 获得ngram对应的 index
                                ngram_list_len[channel_index] += 1
                            for k in range(j + 1):                              # 为当前ngram,关联所有对应字符 --> i+0-->i+j
                                matching_position[channel_index].append((i + k, index))
                # print("ngram_list:  ", ngram_list)
            else:
                ngram_list = None
                matching_position = None
                ngram_list_len = None
            max_ngram_len = max(ngram_list_len) if ngram_list_len is not None else None     # 10个channel中,哪一个channel的ngram最多
            data.append((sentence, label, ngram_list, matching_position, max_ngram_len))

        examples = []
        for i, (sentence, label, word_list, matching_position, word_list_len) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = word_list
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word, matrix=matching_position,
                             sent_len=len(sentence), word_list_len=word_list_len))
        return examples

    def _ngram_category(self, ngram):
        if self.cat_type == 'length':
            index = int(min(self.cat_num, len(ngram))) - 1
            assert 0 <= index < self.cat_num
            return index
        elif self.cat_type == 'freq':
            index = int(min(self.cat_num, math.log2(self.gram2count[ngram]))) - 1
            assert 0 <= index < self.cat_num
            return index
        else:
            raise ValueError()

    def read_ngram_tsv(self, input_file, quotechar=None):
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file, encoding="utf-8")
        sentence = []
        label = []
        sentence_list = []
        label_list = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(label)
                    sentence = []
                    label = []
                continue
            splits = line.strip('\ufeff\n').split()
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            sentence_list.append(sentence)
            label_list.append(label)
            sentence = []
            label = []
        return sentence_list, label_list

    def convert_ngram_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        '''
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word, matrix=matching_position,
                         sent_len=len(sentence), word_list_len=word_list_len)
        '''
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        # print(label_map)
        if self.multi_attention is not None:
            max_word_size = max(max([e.word_list_len for e in examples]), 1)
        else:
            max_word_size = 1

        features = []

        # tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer
        print("self.gram2id: ", len(self.gram2id))
        max_v = 0
        for (ex_index, example) in enumerate(examples):
            # if ex_index == 5:
            #     break
            textlist = example.text_a.split(' ')
            # print(textlist)
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []
            # print(textlist)
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(label_map["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label = labels[i] if labels[i] in label_map else '<UNK>'    # 此时的label_map中没有UNK，需要注意，遇到问题时，再进行修改
                    label_ids.append(label_map[label])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(label_map["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            if self.multi_attention is not None:
                wordlist = example.word
                matching_position = example.matrix
                channel_ids = []
                word_ids = []
                for i in range(self.cat_num):
                    channel_ids.append(i)       # [0,1,2,3,4,5,6,7,8,9]
                    word_ids.append([])

                matching_matrix = np.zeros((self.cat_num, max_seq_length, max_word_size), dtype=np.int)     #[10, 300, 26], 10个10channel中最多的一个有26个ngram
                for i in range(len(wordlist)):      # word_list = ngram_list
                    if len(wordlist[i]) > max_word_size:
                        wordlist[i] = wordlist[i][:max_word_size]

                for i in range(len(wordlist)):      # word_list = ngram_list, 针对每个channel
                    for word in wordlist[i]:
                        if word == '':
                            continue
                        try:
                            word_ = tuple(word)
                            max_v = max(max_v, self.gram2id[word_])
                            word_ids[i].append(self.gram2id[word_])  # 将ngram转化为id
                        except KeyError:
                            print(word)
                            print(wordlist)
                            print(textlist)
                            raise KeyError()

                for i in range(len(word_ids)):      # padding for ngram_list
                    while len(word_ids[i]) < max_word_size:
                        word_ids[i].append(0)
                # print(wordlist)
                # print(matching_position)

                for i in range(len(matching_position)):     # 第i个通道
                    for position in matching_position[i]:
                        char_p = position[0] + 1            # char 在seq中的位置, +1是因为前边有[CLS]
                        word_p = position[1]                # ngram在word_list的位置
                        if char_p > max_seq_length - 2 or word_p > max_word_size - 1:
                            continue
                        else:
                            matching_matrix[i][char_p][word_p] = 1

                assert len(word_ids) == self.cat_num
                assert len(word_ids[0]) == max_word_size
            else:
                word_ids = None
                matching_matrix = None
                channel_ids = None
            # break
            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                # random.shuffle(ngram_matches)
                ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              matching_matrix=matching_matrix,
                              channel_ids=channel_ids,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        print("max_v:  ", max_v)
        return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None, sent_len=None, word_list_len=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.matrix = matrix
        self.sent_len = sent_len
        self.word_list_len = word_list_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None, channel_ids=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix
        self.channel_ids = channel_ids

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks





