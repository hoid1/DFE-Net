# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations  
    
from contextlib import suppress
from numbers import Number
  
from ultralytics.utils import LOGGER, RANK, colorstr 
from ultralytics.utils.torch_utils import model_info_for_loggers  

PREFIX = colorstr("SwanLab: ")   

   
def _enabled(trainer) -> bool:  
    """Return whether SwanLab logging should run for this trainer process."""
    return bool(getattr(trainer.args, "swanlab", False)) and RANK in {-1, 0}
  
   
def _to_scalar(value):
    """Convert tensors and numpy scalars to plain Python scalar values for logging."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()     
    if hasattr(value, "item"):
        return value.item() 
    if isinstance(value, Number):   
        return value
    with suppress(TypeError, ValueError): 
        return float(value)  
    return value  
  

def _metric_name(key: str) -> str:
    """Strip an existing logger prefix before adding the SwanLab group prefix."""   
    return key.split("/", 1)[1] if "/" in key else key    

  
def _mode(mode: str | None) -> str:
    """Normalize DEIM-style cloud mode to the current SwanLab API name."""
    return "online" if mode in {None, "", "cloud"} else mode 


def _swanlab_module(trainer):
    """Return the SwanLab module stored on the trainer, if initialized.""" 
    return getattr(trainer, "_swanlab", None)     
 

def on_pretrain_routine_end(trainer) -> None:    
    """Initialize SwanLab after model, optimizer, and resume state are ready."""
    if not _enabled(trainer) or _swanlab_module(trainer):     
        return
    
    try:
        import swanlab
    except ImportError as e:  
        LOGGER.warning(f"{PREFIX}package not found, install 'swanlab' to enable logging. {e}")   
        return
    
    run_id = getattr(trainer, "swanlab_run_id", None)
    config = {    
        "args": vars(trainer.args),   
        "output_dir": str(trainer.save_dir),     
    }
    with suppress(Exception):
        config["model_info"] = model_info_for_loggers(trainer)

    try: 
        run = swanlab.init(   
            project=getattr(trainer.args, "swanlab_project", None) or "Ultralytics",    
            name=getattr(trainer.args, "swanlab_run_name", None) or getattr(trainer.args, "name", None),
            id=run_id,   
            resume="must" if run_id else None,
            mode=_mode(getattr(trainer.args, "swanlab_mode", "cloud")), 
            config=config,
            log_dir=str(trainer.save_dir),    
        )     
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to initialize, not tracking this run. {e}")
        return   

    trainer._swanlab = swanlab     
    trainer.swanlab_run_id = getattr(run, "id", run_id)
    LOGGER.info(f"{PREFIX}logging run_id({trainer.swanlab_run_id})")     
     
     
def on_train_batch_end(trainer) -> None:     
    """Log step-level loss and learning-rate metrics."""
    swanlab = _swanlab_module(trainer)
    interval = int(getattr(trainer.args, "swanlab_log_interval", 10) or 0)
    global_step = getattr(trainer, "global_step", None)
    if not swanlab or interval <= 0 or global_step is None or global_step % interval:    
        return 

    payload = {   
        "train/loss/total": _to_scalar(getattr(trainer, "loss", 0.0)),
        "train/epoch": getattr(trainer, "epoch", 0),  
        "train/step": getattr(trainer, "batch_i", 0),
    }
    for idx, param_group in enumerate(getattr(trainer.optimizer, "param_groups", [])):  
        payload[f"train/lr/pg_{idx}"] = _to_scalar(param_group["lr"])
    for name, value in trainer.label_loss_items(getattr(trainer, "loss_items", None), prefix="train").items(): 
        payload[f"train/loss/{_metric_name(name)}"] = _to_scalar(value)   
  
    swanlab.log(payload, step=global_step)   
    

def on_train_epoch_end(trainer) -> None:
    """Log epoch-level training loss and learning-rate summaries."""   
    swanlab = _swanlab_module(trainer)
    if not swanlab:
        return     

    step = trainer.epoch + 1
    payload = {"epoch/meta/epoch": step}
    for name, value in trainer.label_loss_items(trainer.tloss, prefix="train").items():
        payload[f"epoch/loss/{_metric_name(name)}"] = _to_scalar(value)
    for name, value in getattr(trainer, "lr", {}).items():  
        payload[f"epoch/lr/{_metric_name(name)}"] = _to_scalar(value) 
    swanlab.log(payload, step=step) 

 
def _epoch_group(key: str) -> str: 
    """Map metric keys into SwanLab-friendly epoch groups."""  
    if key in {"epoch", "n_parameters"}:     
        return "meta"    
    if "loss" in key:
        return "loss"
    return "metric"
 
  
def on_fit_epoch_end(trainer) -> None:
    """Log validation metrics at the end of a fit epoch."""
    swanlab = _swanlab_module(trainer)
    if not swanlab:  
        return
 
    step = trainer.epoch + 1
    payload = {f"epoch/{_epoch_group(k)}/{k}": _to_scalar(v) for k, v in (trainer.metrics or {}).items()}
    if payload: 
        swanlab.log(payload, step=step)
  

def on_train_end(trainer) -> None:
    """Finish the active SwanLab run."""     
    swanlab = _swanlab_module(trainer)     
    if swanlab:     
        swanlab.finish()  
        trainer._swanlab = None     
     

callbacks = {    
    "on_pretrain_routine_end": on_pretrain_routine_end,    
    "on_train_batch_end": on_train_batch_end,    
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,    
    "on_train_end": on_train_end,     
}
