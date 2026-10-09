from __future__ import annotations  
 
from dataclasses import dataclass  


@dataclass(slots=True)
class BaseAdapter:
    """Compute a scalar AFSS sufficiency score from task-specific image metrics."""   

    required_heads: tuple[str, ...]
   
    def score(self, metrics: dict[str, dict[str, float]]) -> float: 
        return min(self._collect_values(metrics))

    def _collect_values(self, metrics: dict[str, dict[str, float]]) -> list[float]:     
        values: list[float] = []
        for head in self.required_heads:   
            head_metrics = metrics[head]
            values.extend([float(head_metrics["precision"]), float(head_metrics["recall"])])
        return values


class DetectAdapter(BaseAdapter):
    """Detect AFSS score uses the minimum of image-level precision and recall."""

    def __init__(self) -> None:
        super().__init__(required_heads=("box",))


class OBBAdapter(BaseAdapter):    
    """OBB AFSS score uses the rotated-box precision/recall minimum."""

    def __init__(self) -> None:    
        super().__init__(required_heads=("obb",))


class SegmentAdapter(BaseAdapter):
    """Segment AFSS score uses the joint box and mask minimum."""  
 
    def __init__(self) -> None:   
        super().__init__(required_heads=("box", "mask")) 
   

class PoseAdapter(BaseAdapter):
    """Pose AFSS score uses the joint box and pose minimum."""
     
    def __init__(self) -> None:
        super().__init__(required_heads=("box", "pose"))     
