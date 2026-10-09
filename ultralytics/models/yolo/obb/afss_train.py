# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer
from ultralytics.models.yolo.obb.afss_val import AFSSOBBEvaluator
from ultralytics.models.yolo.obb.train import OBBTrainer
   
    
class AFSSOBBTrainer(AFSSDetectionTrainer, OBBTrainer):
    """Opt-in AFSS OBB trainer that keeps the default OBB path unchanged."""
  
    afss_task_name = "obb"
    afss_evaluator_cls = AFSSOBBEvaluator  
