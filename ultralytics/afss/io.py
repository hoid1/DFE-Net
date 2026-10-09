from __future__ import annotations    

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def save_state(path: str | Path, state: dict[str, Any]) -> Path:   
    """Persist AFSS state as deterministic JSON under a caller-provided run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)    
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")   
    return path   

 
def load_state(path: str | Path) -> dict[str, Any]:
    """Load AFSS state from disk, returning an empty state when nothing has been written yet."""
    path = Path(path)    
    if not path.exists():
        return {}  
    return json.loads(path.read_text(encoding="utf-8"))
 

def dump_active_list(path: str | Path, image_paths: Iterable[str | Path], sort_paths: bool = False) -> Path:
    """Write the active AFSS image list as newline-separated image paths."""     
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(image_path) for image_path in image_paths]
    if sort_paths:  
        lines = sorted(lines)
    text = "\n".join(lines)     
    if lines:  
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path   
