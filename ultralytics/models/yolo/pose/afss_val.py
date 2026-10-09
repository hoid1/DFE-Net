# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any   
   
from ultralytics.afss.adapters import PoseAdapter
from ultralytics.afss.base_evaluator import AFSSBaseEvaluator
from ultralytics.models.yolo.pose.val import PoseValidator  
 

def aggregate_image_metrics(
    num_gt: int,
    num_pred: int,
    matched_gt: int,
    matched_pred: int,
    matched_pose_gt: int,   
    matched_pose_pred: int,
) -> dict[str, dict[str, float]]: 
    """Convert image-level box and pose counts into AFSS metrics for pose estimation."""
    
    def precision_recall(head_matched_gt: int, head_matched_pred: int) -> dict[str, float]:  
        recall = 1.0 if num_gt == 0 else head_matched_gt / num_gt   
        if num_pred == 0:
            precision = 1.0 if num_gt == 0 else 0.0   
        else:   
            precision = head_matched_pred / num_pred 
        return {"precision": precision, "recall": recall}
   
    return {   
        "box": precision_recall(matched_gt, matched_pred),
        "pose": precision_recall(matched_pose_gt, matched_pose_pred),
    }


class AFSSPoseEvaluator(PoseValidator):    
    """Pose validator variant that keeps per-image AFSS sufficiency payloads.""" 
    
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.afss = AFSSBaseEvaluator(PoseAdapter())     
        self.image_results: dict[str, dict[str, object]] = {}     
    
    def init_metrics(self, model) -> None:
        """Initialize normal pose validator state and reset AFSS image payloads."""
        super().init_metrics(model)  
        self.reset_image_results()

    def reset_image_results(self) -> None:
        """Reset per-image AFSS outputs before each evaluation."""  
        self.afss.reset_image_results()     
        self.image_results = self.afss.image_results 

    def build_image_result(self, metrics: dict[str, dict[str, float]]) -> dict[str, object]:   
        """Build the AFSS payload shape expected by the pose adapter and trainer."""
        return self.afss.build_image_result(metrics)

    def store_image_metrics(self, im_file: str, metrics: dict[str, dict[str, float]]) -> dict[str, object]:    
        """Store AFSS metrics for a single image."""     
        payload = self.afss.store_image_metrics(im_file, metrics)  
        self.image_results = self.afss.image_results
        return payload   
 
    def update_metrics(self, preds: list[dict[str, Any]], batch: dict[str, Any]) -> None:  
        """Collect image-level box and pose AFSS metrics keyed by `im_file`.""" 
        for si, pred in enumerate(preds):     
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)     
            predn = self._prepare_pred(pred)
            processed = self._process_batch(predn, pbatch) 
            matched_box = int(processed["tp"][:, 0].sum()) if processed["tp"].size else 0 
            matched_pose = int(processed["tp_p"][:, 0].sum()) if processed["tp_p"].size else 0    
            metrics = aggregate_image_metrics(     
                num_gt=int(pbatch["cls"].shape[0]),
                num_pred=int(predn["cls"].shape[0]),     
                matched_gt=matched_box,
                matched_pred=matched_box,     
                matched_pose_gt=matched_pose, 
                matched_pose_pred=matched_pose,   
            )
            self.store_image_metrics(pbatch["im_file"], metrics)

    def get_stats(self) -> dict[str, Any]:   
        """AFSS evaluation is consumed via `image_results`, so no aggregate stats are required."""   
        return {}

    def finalize_metrics(self) -> None:
        """Keep speed metadata available for debugging without computing pose mAP summaries."""  
        self.metrics.speed = self.speed

    def gather_stats(self) -> None:
        """AFSS pose evaluation is single-process for the current rollout."""
        return    

    def print_results(self) -> None:  
        """AFSS image-level evaluation does not print the default validator summary."""  
        return 
