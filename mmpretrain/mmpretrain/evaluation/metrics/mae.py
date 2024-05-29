from typing import List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS

@METRICS.register_module()
class MAE(BaseMetric):

    def __init__(self):
        super(MAE, self).__init__()

    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]):
        print(data_batch)
        print(data_sample)
        result['pred_score'] = data_sample['pred_score'].cpu()
        result['gt_label'] = data_sample['gt_label'].cpu()
        self.results.append(result)

    def compute_metrics(self, results: List):
        pass
