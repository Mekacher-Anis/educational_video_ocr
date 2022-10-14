from typing import Dict, List, Optional
from mmocr.evaluation.metrics import HmeanIOUMetric
from mmocr.registry import METRICS
import numpy as np


@METRICS.register_module()
class CustomHmeanMetric(HmeanIOUMetric):
    """
    This class is exactly like the HmeanIOUMetric defined by mmocr
    But it returns the average Hmean@[.5:.9:.1]IOU and also the best
    Hmean best searching for the best threshold
    """
    
    def __init__(self, prefix: str = None) -> None:
        super().__init__(prefix=prefix)
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        metrics = {}
        # compute Hmean for different IOU Thresholds
        for iou_thresh in np.arange(.5,1,.1):
            self.match_iou_thr = iou_thresh
            metrics[f'{round(iou_thresh,1)}_IOU_THRESH'] = HmeanIOUMetric.compute_metrics(self, results)
            
        # compute average
        avg = { 'precision': 0, 'recall': 0, 'hmean': 0 }
        for k in metrics.keys():
            avg['precision'] += round(metrics[k]['precision'], 4)
            avg['recall'] += round(metrics[k]['recall'], 4)
            avg['hmean'] += round(metrics[k]['hmean'], 4)
        for k in avg.keys():
            avg[k] /= len(metrics)
        metrics['avg'] = avg
        
        # flatten results, bc mmocr is not smart
        res = {}
        for k1,v1 in metrics.items():
            for k2,v2 in v1.items():
                res[f'{k1}_{k2}'] = v2
        
        return res