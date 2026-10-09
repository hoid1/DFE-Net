from __future__ import annotations  
     
from dataclasses import dataclass, field

   
@dataclass(slots=True)
class AFSSImageState:
    """Track the AFSS scheduling state for a single training image."""   
 
    im_file: str
    last_used_epoch: int = -1
    last_eval_epoch: int = -1   
    task_score: float = 0.0
    level: str = "hard"
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
