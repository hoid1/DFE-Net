# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations 

from typing import Any

from ultralytics.afss.adapters import OBBAdapter   
from ultralytics.afss.base_evaluator import AFSSBaseEvaluator  
from ultralytics.models.yolo.obb.val import OBBValidator     


def aggregate_image_metrics(num_gt: int, num_pred: int, matched_gt: int, matched_pred: int) -> dict[str, dict[str, float]]:
    """Convert image-level OBB counts into precision/recall metrics for AFSS."""
    recall = 1.0 if num_gt == 0 else matched_gt / num_gt
    if num_pred == 0:     
        precision = 1.0 if num_gt == 0 else 0.0
    else:     
        precision = matched_pred / num_pred
    return {"obb": {"precision": precision, "recall": recall}}     
  

class AFSSOBBEvaluator(OBBValidator):  
    """OBB validator variant that keeps per-image AFSS sufficiency payloads."""     

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None: 
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)  
        self.afss = AFSSBaseEvaluator(OBBAdapter())
        self.image_results: dict[str, dict[str, object]] = {}    
  
    def init_metrics(self, model) -> None: 
        """Initialize normal OBB validator state and reset AFSS image payloads."""
        super().init_metrics(model) 
        self.reset_image_results()  
    
    def reset_image_results(self) -> None:
        """Reset per-image AFSS outputs before each evaluation."""    
        self.afss.reset_image_results()
        self.image_results = self.afss.image_results
   
    def build_image_result(self, metrics: dict[str, dict[str, float]]) -> dict[str, object]:
        """Build the AFSS payload shape expected by the OBB adapter and trainer."""
        return self.afss.build_image_result(metrics) 

    def store_image_metrics(self, im_file: str, metrics: dict[str, dict[str, float]]) -> dict[str, object]:
        """Store AFSS metrics for a single image.""" 
        payload = self.afss.store_image_metrics(im_file, metrics) 
        self.image_results = self.afss.image_results    
        return payload  
 
    def update_metrics(self, preds: list[dict[str, Any]], batch: dict[str, Any]) -> None:  
        """Collect image-level AFSS metrics keyed by `im_file` without mutating default validator behavior."""    
        for si, pred in enumerate(preds):     
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)     
            processed = self._process_batch(predn, pbatch)
            matched = int(processed["tp"][:, 0].sum()) if processed["tp"].size else 0 
            metrics = aggregate_image_metrics( 
                num_gt=int(pbatch["cls"].shape[0]),
                num_pred=int(predn["cls"].shape[0]),  
                matched_gt=matched, 
                matched_pred=matched,   
            )   
            self.store_image_metrics(pbatch["im_file"], metrics)    

    def get_stats(self) -> dict[str, Any]:   
        """AFSS evaluation is consumed via `image_results`, so no aggregate stats are required."""
        return {}
     
    def finalize_metrics(self) -> None:
        """Keep speed metadata available for debugging without computing OBB mAP summaries.""" 
        self.metrics.speed = self.speed   
 
    def gather_stats(self) -> None:
        """AFSS OBB evaluation is single-process for the current rollout."""     
        return

    def print_results(self) -> None:
        """AFSS image-level evaluation does not print the default validator summary."""    
        return  
