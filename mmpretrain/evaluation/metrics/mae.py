from typing import List, Optional, Sequence, Union, Dict
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS
import torch

@METRICS.register_module()
class ICNetMAE(BaseMetric):

    def __init__(self):
        super(ICNetMAE, self).__init__()

    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]):
        for data_sample in data_samples:
            result = dict()
            result['pred_score'] = data_sample['pred_score'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List):

        metrics ={}

        predictions = torch.cat([res['pred_score'] for res in results])
        targets = torch.cat([res['gt_label'] for res in results])

        metrics['MAE'] = torch.mean(torch.abs(predictions - targets))
        return metrics
