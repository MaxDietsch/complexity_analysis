from typing import List, Optional, Sequence, Union, Dict
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS

@METRICS.register_module()
class ICNetMAE(BaseMetric):

    def __init__(self):
        super(ICNetMAE, self).__init__()

    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]):
        result['pred_score'] = data_samples['pred_score'].cpu()
        result['gt_label'] = data_samples['gt_label'].cpu()
        self.results.append(result)

    def compute_metrics(self, results: List):
        print(results)
