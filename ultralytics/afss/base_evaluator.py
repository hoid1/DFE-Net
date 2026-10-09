from __future__ import annotations
     
from ultralytics.afss.adapters import BaseAdapter    


class AFSSBaseEvaluator:
    """Shared helpers for task-specific AFSS evaluators that collect image-level scores."""  

    def __init__(self, adapter: BaseAdapter): 
        self.adapter = adapter   
        self.image_results: dict[str, dict[str, object]] = {}

    def reset_image_results(self) -> None:
        """Clear per-image AFSS results before a new evaluation pass."""
        self.image_results = {}

    def build_image_result(self, metrics: dict[str, dict[str, float]]) -> dict[str, object]:
        """Shape raw image metrics into the common AFSS payload."""
        return {    
            "metrics": metrics,  
            "task_score": self.adapter.score(metrics),    
        }
  
    def store_image_metrics(self, im_file: str, metrics: dict[str, dict[str, float]]) -> dict[str, object]:
        """Store AFSS metrics for one image, keyed by image file path."""
        payload = self.build_image_result(metrics)  
        self.image_results[im_file] = payload
        return payload
