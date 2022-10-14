from pprint import pprint
from . import models
import os
import os.path as osp
from typing import Callable, Dict, List, Optional, OrderedDict, Sequence, Union
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmocr.utils import register_all_modules
from mmocr.registry import METRICS, DATASETS, TRANSFORMS
from mmocr.datasets import OCRDataset
import time
import torch
import timeit
from .hmean_metric import CustomHmeanMetric

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


@METRICS.register_module()
class TestingPipelineMetric(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu') -> None:
        super().__init__(collect_device=collect_device, prefix='')

        print('Stupid Metric has been initialized')

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        # for data in data_samples:
        #     pprint(data)
        #     break
        # print('Process on TestingPipelineMetric has been called')
        # exit(0)
        self.results.append({})

    def compute_metrics(self, results: List[Dict]) -> Dict:
        print('Compute Metrics on TestingPipelineMetric has been called.')
        return {}


@DATASETS.register_module()
class CustomTestingPipelineDataset(OCRDataset):
    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: dict = ...,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = ...,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000
                 ):
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg,
                         indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


class TestPipeline:
    """
    This Class given a det_model, rec_model and pre/post-processing methods
    will run an OCR Pipeline on the LVDB Dataset and measure some performance metrics.
    """

    def __init__(self,
                 det: str = None,
                 recog: str = None,
                 batch_size: int = 1,
                 launcher: str = 'none',
                 **kwargs) -> None:
        # register all modules in mmocr into the registries
        # do not init the default scope here because it will be init in the runner
        register_all_modules(init_default_scope=False)

        self.config_dir = os.environ['MMOCR_HOME'] + \
            '/configs' if 'MMOCR_HOME' in os.environ else './configs'
        self.launcher = launcher
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        self.timestamp = time.strftime('%Y%m%d_%H%M%S',
                                       time.localtime(timestamp.item()))
        self.det_model_name = det
        self.recog_model_name = recog
        self.batch_size = batch_size

        self.work_dir_base = osp.join(
            './work_dirs', f'{self.det_model_name}___{self.recog_model_name}', self.timestamp)

        # build text detection runner
        if self.det_model_name:
            self.det_runner, self.det_model_config = self._build_from_cfg(det, True)
        
        # build text recognition runner
        if self.recog_model_name:
            self.recog_runner, self.recog_model_config = self._build_from_cfg(recog, False)

        self.metrics = {}
        self.det_metric_output_path = './work_dirs/det_model_metrics.csv'
        self.rec_metric_output_path = './work_dirs/rec_model_metrics.csv'

    def _build_from_cfg(self, model_name: str, det: bool):
        # load config
        model_config = self.__get_model_config(model_name)
        config_path = osp.join(self.config_dir, model_config['config'])
        cfg = Config.fromfile(config_path)
        cfg.launcher = self.launcher

        # work_dir is determined in this priority: segment in file > filename
        cfg.work_dir = osp.join(self.work_dir_base,
                                osp.splitext(osp.basename(model_config['config']))[0])

        cfg.load_from = model_config['ckpt']
        cfg.log_level = 'CRITICAL'

        dump_metric = dict(
            type='DumpResults',
            out_file_path=osp.join(
                cfg.work_dir,
                f'{osp.basename(cfg.load_from)}_predictions.pkl'))

        stupid_metric = dict(type='TestingPipelineMetric')
        custom_hmean_metric = dict(type='CustomHmeanMetric', prefix='LVDB')
                
        cfg.test_dataloader.batch_size = self.batch_size

        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            for eva in cfg.test_evaluator:
                eva['prefix' if det else 'dataset_prefixes']= 'LVDB' if det else ['LVDB']
            cfg.test_evaluator.append(dump_metric)
            cfg.test_evaluator.append(stupid_metric)
            if det:
                # remove old hmean iou metric
                cfg.test_evaluator = [e for e in cfg.test_evaluator if e['type'] != 'HmeanIOUMetric']
                cfg.test_evaluator.append(custom_hmean_metric)
        elif isinstance(cfg.test_evaluator, Dict) and 'metrics' in cfg.test_evaluator and isinstance(cfg.test_evaluator.metrics, list):
            cfg.test_evaluator['prefix' if det else 'dataset_prefixes']= 'LVDB' if det else ['LVDB']
            cfg.test_evaluator.metrics.append(dump_metric)
            cfg.test_evaluator.metrics.append(stupid_metric)
            if det:
                # remove old hmean iou metric
                cfg.test_evaluator.metrics = [e for e in cfg.test_evaluator.metrics if e['type'] != 'HmeanIOUMetric']
                cfg.test_evaluator.metrics.append(custom_hmean_metric)
        else:
            cfg.test_evaluator['prefix' if det else 'dataset_prefixes']= 'LVDB' if det else ['LVDB']
            cfg.test_evaluator = [ cfg.test_evaluator, dump_metric, stupid_metric]
            if det:
                cfg.test_evaluator = [e for e in cfg.test_evaluator if e['type'] != 'HmeanIOUMetric']
                cfg.test_evaluator.append(custom_hmean_metric)

        config = {
            'filepath': config_path,
            'ckpt': model_config['ckpt'],
            'loaded': cfg
        }

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            return Runner.from_cfg(cfg), config
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            return RUNNERS.build(cfg), config

    def __get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        if model_name not in models.model_dict:
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return models.model_dict[model_name]

    def start_test(self):
        if self.det_model_name:
            # run the detection model
            time, metrics = timeit.timeit(self.det_runner.test, number=1)
            self.metrics['det'] = metrics
            self.metrics['det']['run_time'] = time
            print('Finished text detection')
            # print(self.metrics['det'])
        if self.recog_model_name:
            # runt the recognition model on the result of the detection model
            time, metrics = timeit.timeit(self.recog_runner.test, number=1)
            self.metrics['recog'] = metrics
            self.metrics['recog']['run_time'] = time
            print('Finished text recognition')
            # print(self.metrics['recog'])
        return self.metrics

    def write_metrics(self):
        if self.det_model_name:
            with open(self.det_metric_output_path, mode='a+') as det_output:
                header = ''
                if os.stat(self.det_metric_output_path).st_size == 0:
                    header = 'model_name,lvdb_.5_iou_thresh_precision,lvdb_.5_iou_thresh_recall,lvdb_.5_iou_thresh_hmean,lvdb_avg_precision,lvdb_avg_recall,lvdb_avg_hmean,FPS,number_test_samples,run_time,config_path,ckpt,timestamp'
                precision = round(self.metrics['det']['LVDB/0.5_IOU_THRESH_precision'], 4)
                recall = round(self.metrics['det']['LVDB/0.5_IOU_THRESH_recall'], 4)
                hmean = round(self.metrics['det']['LVDB/0.5_IOU_THRESH_hmean'], 4)
                avg_precision = round(self.metrics['det']['LVDB/avg_precision'], 4)
                avg_recall = round(self.metrics['det']['LVDB/avg_recall'], 4)
                avg_hmean = round(self.metrics['det']['LVDB/avg_hmean'], 4)
                run_time = self.metrics['det']['run_time']
                num_samples = len(self.det_runner.test_dataloader.dataset)
                FPS = int(num_samples // run_time)
                config_path = self.det_model_config['filepath']
                ckpt = self.det_model_config['ckpt']
                line = f'{self.det_model_name},{precision},{recall},{hmean},{avg_precision},{avg_recall},{avg_hmean},{FPS},{num_samples},{run_time},{config_path},{ckpt},{self.timestamp}'
                if header: det_output.write(f'{header}\n')
                det_output.write(f'{line}\n')
        if self.recog_model_name:
            with open(self.rec_metric_output_path, mode='a+') as recog_output:
                header = ''
                if os.stat(self.rec_metric_output_path).st_size == 0:
                    header = 'model_name,lvdb_word_acc,lvdb_word_acc_ignore_case,lvdb_word_acc_ignore_case_symbol,lvdb_char_recall,lvdb_char_precision,FPS,number_test_samples,run_time,config_path,ckpt,timestamp'
                word_acc = round(self.metrics['recog']['LVDB/recog/word_acc'], 2)
                word_acc_ignore_case = round(self.metrics['recog']['LVDB/recog/word_acc_ignore_case'], 2)
                word_acc_ignore_case_symbol = round(self.metrics['recog']['LVDB/recog/word_acc_ignore_case_symbol'], 2)
                char_recall = round(self.metrics['recog']['LVDB/recog/char_recall'], 2)
                char_precision = round(self.metrics['recog']['LVDB/recog/char_precision'], 2)
                run_time = self.metrics['recog']['run_time']
                num_samples = len(self.recog_runner.test_dataloader.dataset)
                FPS = int(num_samples // run_time)
                config_path = self.recog_model_config['filepath']
                ckpt = self.recog_model_config['ckpt']
                line = f'{self.recog_model_name},{word_acc},{word_acc_ignore_case},{word_acc_ignore_case_symbol},{char_recall},{char_precision},{FPS},{num_samples},{run_time},{config_path},{ckpt},{self.timestamp}'
                if header: recog_output.write(f'{header}\n')
                recog_output.write(f'{line}\n')
