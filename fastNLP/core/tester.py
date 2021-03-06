"""
tester模块实现了 fastNLP 所需的Tester类，能在提供数据、模型以及metric的情况下进行性能测试。

.. code-block::

    import numpy as np
    import torch
    from torch import nn
    from fastNLP import Tester
    from fastNLP import DataSet
    from fastNLP import AccuracyMetric

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)
        def forward(self, a):
            return {'pred': self.fc(a.unsqueeze(1)).squeeze(1)}

    model = Model()

    dataset = DataSet({'a': np.arange(10, dtype=float), 'b':np.arange(10, dtype=float)*2})

    dataset.set_input('a')
    dataset.set_target('b')

    tester = Tester(dataset, model, metrics=AccuracyMetric())
    eval_results = tester.test()

这里Metric的映射规律是和 :class:`fastNLP.Trainer` 中一致的，具体使用请参考 :mod:`trainer 模块<fastNLP.core.trainer>` 的1.3部分。
Tester在验证进行之前会调用model.eval()提示当前进入了evaluation阶段，即会关闭nn.Dropout()等，在验证结束之后会调用model.train()恢复到训练状态。


"""
import time

import torch
import torch.nn as nn
import numpy as np

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm

from .batch import BatchIter, DataSetIter
from .dataset import DataSet
from .metrics import _prepare_metrics
from .sampler import SequentialSampler
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
from functools import partial
from ._logger import logger
import json

__all__ = [
    "Tester"
]


class Tester(object):
    """
    Tester是在提供数据，模型以及metric的情况下进行性能测试的类。需要传入模型，数据以及metric进行验证。
    """
    
    def __init__(self, data, model, metrics, batch_size=32, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 use_knowledge=False,
                 knowledge_type=None,
                 use_ngram=False,
                 zen_model=None,
                 ngram_test_examlpes=None,
                 args=None,
                 gram2id=None,
                 dataset=None
                 # n_device=None
                 ):
        """

        :param ~fastNLP.DataSet data: 需要测试的数据集
        :param torch.nn.module model: 使用的模型
        :param ~fastNLP.core.metrics.MetricBase,List[~fastNLP.core.metrics.MetricBase] metrics: 测试时使用的metrics
        :param int batch_size: evaluation时使用的batch_size有多大。
        :param str,int,torch.device,list(int) device: 将模型load到哪个设备。默认为None，即Trainer不对模型
            的计算位置进行管理。支持以下的输入:

            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中,可见的第一个GPU中,可见的第二个GPU中;

            2. torch.device：将模型装载到torch.device上。

            3. int: 将使用device_id为该值的gpu进行训练

            4. list(int)：如果多于1个device，将使用torch.nn.DataParallel包裹model, 并使用传入的device。

            5. None. 为None则不对模型进行任何处理，如果传入的model为torch.nn.DataParallel该值必须为None。

            如果模型是通过predict()进行预测的话，那么将不能使用多卡(DataParallel)进行验证，只会使用第一张卡上的模型。
        :param int verbose: 如果为0不输出任何信息; 如果为1，打印出验证结果。
        :param bool use_tqdm: 是否使用tqdm来显示测试进度; 如果为False，则不会显示任何内容。
        """
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger

        # new_add
        self.use_knowledge = use_knowledge
        self.knowledge_type = knowledge_type
        self.args = args
        self.gram2id = gram2id
        self.device = device
        self.label_list = ["O", "B", "I", "E", "S", "[CLS]", "[SEP]"]
        self.label_map = {label: i for i, label in enumerate(self.label_list, 1)}
        
        self.dataset = dataset

        # add ZEN
        self.use_ngram = use_ngram
        self.zen_model = zen_model
        self.ngram_test_examlpes = ngram_test_examlpes

        if isinstance(data, DataSet):
            self.data_iterator = DataSetIter(
                dataset=data, batch_size=batch_size, num_workers=num_workers, sampler=SequentialSampler())
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  # 用于匹配参数
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  # 用于调用
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward

    def get_ngram_data(self, indices, max_seq_len):
        batch_word_ids, batch_matching_matrix, batch_channel_ids = [], [], []
        '''
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_word_ids, all_matching_matrix,
                         all_channel_ids, all_valid_ids, all_lmask_ids
        '''
        for index in indices:
            input_ids, input_mask, _, _, word_ids, matching_matrix, channel_ids, _, _ = self.ngram_test_examlpes[index]
            batch_word_ids.append(word_ids.unsqueeze(0))
            batch_matching_matrix.append(matching_matrix.unsqueeze(0))
            batch_channel_ids.append(channel_ids.unsqueeze(0))
        batch_word_ids = torch.cat(batch_word_ids, dim=0).to(self._model_device)
        batch_matching_matrix = torch.cat(batch_matching_matrix, dim=0).to(self._model_device)
        batch_channel_ids = torch.cat(batch_channel_ids, dim=0).to(self._model_device)

        batch_matching_matrix = batch_matching_matrix[:, :, 1: 1 + max_seq_len, :]
        # print('batch_word_ids',batch_word_ids.size())
        # print('batch_matching_matrix',batch_matching_matrix.size())
        # print('batch_channel_ids',batch_channel_ids.size())

        # batch_ngram_ids = batch_ngram_ids[:,1:1+max_seq_len]
        # batch_ngram_positions = batch_ngram_positions[:, 1:1 + max_seq_len]
        # batch_ngram_attention_mask = batch_ngram_attention_mask[:, 1:1 + max_seq_len]

        return batch_word_ids, batch_matching_matrix, batch_channel_ids

    def get_zen_data(self, indices, max_seq_len):
        batch_input_ids, batch_ngram_ids, batch_ngram_positions, batch_attention_mask, batch_ngram_attention_mask = \
            [], [], [], [], []
        for index in indices:
            input_ids, input_mask, _, _, ngram_ids, ngram_positions, _, _, ngram_masks, _, _ = self.zen_dataset[index]
            batch_input_ids.append(input_ids.unsqueeze(0))
            batch_ngram_ids.append(ngram_ids.unsqueeze(0))
            batch_ngram_positions.append(ngram_positions.unsqueeze(0))
            batch_attention_mask.append(input_mask.unsqueeze(0))
            # batch_ngram_attention_mask.append(ngram_masks.unsqueeze(0))

        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(self._model_device)
        batch_ngram_ids = torch.cat(batch_ngram_ids, dim=0).to(self._model_device)
        batch_ngram_positions = torch.cat(batch_ngram_positions, dim=0).to(self._model_device)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0).to(self._model_device)
        # batch_ngram_attention_mask = torch.cat(batch_ngram_attention_mask, dim=0).to(self._model_device)

        with torch.no_grad():
            output, _ = self.zen_model(input_ids=batch_input_ids, input_ngram_ids=batch_ngram_ids,
                                       ngram_position_matrix=batch_ngram_positions, token_type_ids=None,
                                       attention_mask=batch_attention_mask, ngram_attention_mask=None,
                                       output_all_encoded_layers=False, head_mask=None)
        output = output[:, 1: 1 + max_seq_len, :]
        return output

    def test(self, epoch=None):
        r"""开始进行验证，并返回验证结果。

        :return Dict[Dict]: dict的二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
        """
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")
                    start_time = time.time()
                    result = []

                    for indices, batch_x, batch_y in data_iterator:

                        # zen_input = None
                        # if self.use_zen:
                        #     zen_input = self.get_zen_data(indices, max_seq_len=torch.max(batch_x.get("seq_len")).item())
                        # batch_x["zen_input"] = zen_input
                        # ******************************************************************************************************
                        # if self.use_knowledge:
                        #     word_seq = None
                        #     label_value_matrix = None
                        #     word_mask = None
                        #     label_content1 = None
                        # print(batch_x)
                        # ******************************************************************************************************

                        word_seq = None
                        word_mask = None
                        channel_ids = None
                        if self.use_ngram:
                            word_seq, word_mask, channel_ids = self.get_ngram_data(indices, max_seq_len=torch.max(batch_x.get("seq_len")).item())
                        batch_x["word_seq"] = word_seq
                        batch_x["word_mask"] = word_mask
                        batch_x["channel_ids"] = channel_ids

                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)



                        pred_dict = self._data_forward(self._predict_func, batch_x=batch_x)

                        pred = pred_dict["pred"].tolist()
                        target = batch_y["target"].tolist()

                        # new_add for print result
                        if epoch is not None:
                            for i, p, t in zip(indices, pred, target):
                                result.append({"index": i, "pred": p, "target": t})

                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        for metric in self.metrics:
                            metric(pred_dict, batch_y)

                        if self.use_tqdm:
                            pbar.update()

                    # new_add for print result
                    # if epoch is not None:
                    # with open("result/result.txt", "w+", encoding="utf-8") as f:
                        # f.write(json.dumps(result))

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
            

        self._mode(network, is_test=False)
        return eval_results
    
    def _mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()
    
    def _data_forward(self, func, batch_x):
        """A forward pass of the model. """
        # x = _build_args(func, **x)
        y = self._predict_func_wrapper(
            batch_x.get("chars"),
            batch_x.get("bigrams", None),
            batch_x.get("word_seq"),
            batch_x.get("word_mask"),
            batch_x.get("channel_ids")
            # batch_x.get("zen_input")
        )
        return y
    
    def _format_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]
        
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, word_ids=None, matching_matrix=None, label_content=None):
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix
        self.label_content = label_content
        
        
