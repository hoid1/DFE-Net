from __future__ import annotations    
  
import math
from collections.abc import Iterable, Mapping     
     
from ultralytics.afss.state import AFSSImageState  


def classify_score(score: float, moderate_threshold: float, easy_threshold: float) -> str:     
    """Map a scalar task score into the paper-aligned AFSS level buckets."""
    if score < moderate_threshold:
        return "hard"
    if score <= easy_threshold:
        return "moderate"  
    return "easy"    

    
def partition_states(   
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],
    moderate_threshold: float,     
    easy_threshold: float,
) -> dict[str, list[AFSSImageState]]:
    """Group image states by AFSS difficulty level."""
    grouped = {"easy": [], "moderate": [], "hard": []}
    for state in _coerce_states(states):    
        state.level = classify_score(state.task_score, moderate_threshold, easy_threshold)
        grouped[state.level].append(state)
    return grouped
    
  
def select_easy_forced_review(
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],
    *,    
    current_epoch: int, 
    forced_gap: int,     
) -> list[str]:    
    """Force long-unseen easy images back into the active set first."""
    return _forced_selection(states, current_epoch=current_epoch, forced_gap=forced_gap)
  
   
def select_moderate_forced_coverage(    
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],  
    *, 
    current_epoch: int,   
    forced_gap: int,
) -> list[str]:   
    """Guarantee moderate images are revisited within the configured epoch gap."""     
    return _forced_selection(states, current_epoch=current_epoch, forced_gap=forced_gap)     


def select_active_images(
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],
    *,
    current_epoch: int,
    easy_ratio: float,  
    moderate_ratio: float, 
    easy_forced_gap: int,   
    moderate_forced_gap: int,  
    moderate_threshold: float, 
    easy_threshold: float,  
) -> list[str]:    
    """Select the next epoch's active images using AFSS level rules."""  
    grouped = partition_states(states, moderate_threshold=moderate_threshold, easy_threshold=easy_threshold)    
    selected = {state.im_file for state in grouped["hard"]}    
    selected.update(
        select_easy_forced_review(grouped["easy"], current_epoch=current_epoch, forced_gap=easy_forced_gap)     
    )
    selected.update(  
        select_moderate_forced_coverage(     
            grouped["moderate"], current_epoch=current_epoch, forced_gap=moderate_forced_gap    
        )  
    )
    selected.update(_sample_by_ratio(grouped["easy"], current_epoch=current_epoch, ratio=easy_ratio))
    selected.update(_sample_by_ratio(grouped["moderate"], current_epoch=current_epoch, ratio=moderate_ratio))  
    return sorted(selected)
   

def _coerce_states(states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState]) -> list[AFSSImageState]:   
    return list(states.values()) if isinstance(states, Mapping) else list(states)
  
     
def _forced_selection(
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],
    *,     
    current_epoch: int,
    forced_gap: int,
) -> list[str]: 
    due_states = [     
        state
        for state in _coerce_states(states)
        if current_epoch - state.last_used_epoch >= forced_gap
    ]    
    return [state.im_file for state in _sort_by_staleness(due_states, current_epoch)]     

   
def _sample_by_ratio(   
    states: Mapping[str, AFSSImageState] | Iterable[AFSSImageState],  
    *,
    current_epoch: int,
    ratio: float,
) -> list[str]:    
    bucket = _coerce_states(states)
    if ratio <= 0 or not bucket:
        return []
    sample_count = min(len(bucket), math.ceil(len(bucket) * ratio))
    return [state.im_file for state in _sort_by_staleness(bucket, current_epoch)[:sample_count]]     

  
def _sort_by_staleness(states: list[AFSSImageState], current_epoch: int) -> list[AFSSImageState]:
    return sorted(
        states,
        key=lambda state: (-(current_epoch - state.last_used_epoch), state.im_file),
    )    
