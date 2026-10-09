# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations
 
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer   
from ultralytics.models.yolo.pose.afss_val import AFSSPoseEvaluator     
from ultralytics.models.yolo.pose.train import PoseTrainer    
  
 
class AFSSPoseTrainer(AFSSDetectionTrainer, PoseTrainer):  
    """Opt-in AFSS pose trainer that keeps the default pose path unchanged."""    

    afss_task_name = "pose" 
    afss_evaluator_cls = AFSSPoseEvaluator
