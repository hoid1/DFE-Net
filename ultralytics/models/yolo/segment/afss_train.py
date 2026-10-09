# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license 

from __future__ import annotations
    
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer
from ultralytics.models.yolo.segment.afss_val import AFSSSegmentationEvaluator
from ultralytics.models.yolo.segment.train import SegmentationTrainer    
   

class AFSSSegmentationTrainer(AFSSDetectionTrainer, SegmentationTrainer):    
    """Opt-in AFSS segmentation trainer that keeps the default segment path unchanged."""   

    afss_task_name = "segment"
    afss_evaluator_cls = AFSSSegmentationEvaluator
