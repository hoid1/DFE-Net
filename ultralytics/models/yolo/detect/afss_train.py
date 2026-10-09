# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  
 
from __future__ import annotations 

from copy import copy, deepcopy
from dataclasses import asdict
from datetime import datetime     
from pathlib import Path
import re   
from typing import Any

import torch

from ultralytics import __version__
from ultralytics.afss.io import dump_active_list, load_state, save_state     
from ultralytics.afss.scheduler import classify_score, select_active_images
from ultralytics.afss.state import AFSSImageState
from ultralytics.models.yolo.detect.afss_val import AFSSDetectionEvaluator    
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, GIT, LOCAL_RANK, LOGGER, RANK, colorstr  
from ultralytics.utils.torch_utils import unwrap_model


class AFSSDetectionTrainer(DetectionTrainer):     
    """Opt-in AFSS detection trainer that keeps the default detect path unchanged."""    
   
    afss_task_name = "detect"
    afss_evaluator_cls = None

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        overrides = {} if overrides is None else overrides 
        super().__init__(cfg, overrides, _callbacks)
        self.afss_enabled = bool(getattr(self.args, "afss", False))    
        self.afss_dir = self.save_dir / "afss"
        self.afss_state_path = self.afss_dir / "state.json" 
        self.afss_state: dict[str, AFSSImageState] = {}     
        self.afss_active_list_path: Path | None = None
        self.afss_active_list_epoch: int | None = None    
        self.afss_evaluator = None     
        self.afss_startup_summary_logged = False
        if self.afss_enabled and RANK in {-1, 0}: 
            self.afss_dir.mkdir(parents=True, exist_ok=True)
            self._load_afss_state()
            self.add_callback("on_train_start", self._on_afss_train_start)
            self.add_callback("on_train_epoch_start", self._on_afss_train_epoch_start)
    
    def _use_full_dataset(self, epoch: int) -> bool:
        """Keep the full train split active during the warmup period."""
        return epoch < self.args.afss_warmup_epochs

    def _should_refresh_afss(self, epoch: int) -> bool:
        """Refresh AFSS image scores on the configured warmup boundary and interval."""  
        if not self.afss_enabled or self._use_full_dataset(epoch):   
            return False
        return (epoch - self.args.afss_warmup_epochs) % self.args.afss_update_interval == 0
   
    def _write_active_train_list(self, image_paths: list[str], epoch: int) -> Path:
        """Persist the active train split under the run-local AFSS directory."""
        path = self.afss_dir / f"train_epoch{epoch:04d}.txt"
        self.afss_active_list_path = path   
        self.afss_active_list_epoch = epoch     
        return dump_active_list(path, image_paths, sort_paths=True)    
  
    def _serialize_afss_state(self) -> dict[str, dict[str, Any]]:
        """Convert in-memory AFSS state into JSON-compatible dictionaries."""     
        return {im_file: asdict(state) for im_file, state in self.afss_state.items()}
 
    def _load_afss_state(self) -> dict[str, AFSSImageState]:
        """Restore AFSS state from the run-local JSON snapshot if present."""    
        raw_state = load_state(self.afss_state_path)
        self.afss_state = {im_file: AFSSImageState(**payload) for im_file, payload in raw_state.items()}  
        return self.afss_state
     
    def _save_afss_state(self) -> Path:
        """Persist AFSS state under the run-local AFSS directory."""
        self.afss_dir.mkdir(parents=True, exist_ok=True) 
        path = self.afss_state_path
        save_state(path, self._serialize_afss_state())     
        return path

    def _should_save_afss_refresh_json(self) -> bool:
        """Return whether per-refresh AFSS debug snapshots are enabled."""
        return bool(getattr(self.args, "afss_save_refresh_json", False))
     
    def _get_afss_resume_metadata(self) -> dict[str, Any]:     
        """Build AFSS metadata that lets resumed training restore the active loader."""
        return {
            "active_list_epoch": self.afss_active_list_epoch,
            "active_list_name": self.afss_active_list_path.name if self.afss_active_list_path else None,
            "state": self._serialize_afss_state(),
        } 
   
    def _load_afss_state_from_payload(self, payload: dict[str, Any] | None) -> dict[str, AFSSImageState]:  
        """Restore AFSS state from checkpoint metadata, falling back to run-local JSON."""
        state_payload = (payload or {}).get("state")
        if state_payload:
            self.afss_state = {im_file: AFSSImageState(**item) for im_file, item in state_payload.items()}
            return self.afss_state 
        return self._load_afss_state()
     
    def _default_resume_active_list_epoch(self) -> int | None:
        """Infer which AFSS active list should still be in force for the resumed epoch."""
        completed_epoch = self.start_epoch - 1
        if completed_epoch < self.args.afss_warmup_epochs:
            return None
        offset = completed_epoch - self.args.afss_warmup_epochs
        return self.args.afss_warmup_epochs + (offset // self.args.afss_update_interval) * self.args.afss_update_interval

    def _get_resume_active_list_path(self, payload: dict[str, Any] | None) -> Path | None:
        """Resolve the AFSS active list to restore for a resumed post-warmup epoch."""   
        payload = payload or {}  
        active_list_name = payload.get("active_list_name")
        if active_list_name:
            candidate = self.afss_dir / active_list_name
            if candidate.exists():   
                return candidate 
        active_list_epoch = payload.get("active_list_epoch")  
        if active_list_epoch is None:   
            active_list_epoch = self._default_resume_active_list_epoch()
        if active_list_epoch is None:     
            return None
        candidate = self.afss_dir / f"train_epoch{active_list_epoch:04d}.txt"  
        return candidate if candidate.exists() else None

    def _restore_afss_resume_state(self, ckpt: dict[str, Any] | None) -> None: 
        """Restore AFSS state and the active train loader for resumed post-warmup training."""
        if not self.resume or not self.afss_enabled:
            return
        payload = (ckpt or {}).get("afss_resume") or {}
        self._load_afss_state_from_payload(payload)
        if self.start_epoch <= self.args.afss_warmup_epochs:   
            return  
        active_list_path = self._get_resume_active_list_path(payload)
        if active_list_path is None and self.afss_state:    
            active_images = self._select_active_train_images(self.start_epoch - 1)
            active_list_path = self._write_active_train_list(active_images, self.start_epoch) 
        if active_list_path is not None:   
            self._rebuild_train_loader_from_list(active_list_path)
   
    def _initialize_afss_state(self, image_files: list[str]) -> None:     
        """Ensure every train image has an AFSS state entry."""    
        for im_file in image_files:     
            self.afss_state.setdefault(im_file, AFSSImageState(im_file=im_file))
     
    def _mark_last_used_epoch(self, image_files: list[str], epoch: int) -> None:
        """Track the most recent epoch when an image is scheduled into training."""
        for im_file in image_files: 
            self.afss_state.setdefault(im_file, AFSSImageState(im_file=im_file)).last_used_epoch = epoch
 
    def _get_afss_level_counts(self) -> dict[str, int]:
        """Count how many images currently fall into each AFSS difficulty bucket.""" 
        counts = {"easy": 0, "moderate": 0, "hard": 0}
        for state in self.afss_state.values():    
            level = state.level if state.level in counts else "hard"
            counts[level] += 1     
        return counts

    @staticmethod   
    def _format_afss_delta(current: int, previous: int | None) -> str:
        """Format count changes against the previous AFSS update."""   
        if previous is None:
            return "(n/a)" 
        return f"({current - previous:+d})"
   
    def _get_train_loader_image_count(self) -> int | None:
        """Return the number of images behind the current train loader when available."""
        dataset = getattr(self.train_loader, "dataset", None)
        image_files = getattr(dataset, "im_files", None)
        return len(image_files) if image_files is not None else None 

    def _log_afss_update_summary(
        self, 
        epoch: int,    
        previous_counts: dict[str, int] | None,
        previous_active_count: int | None, 
        active_images: list[str],
    ) -> None:     
        """Print AFSS bucket counts and active-set deltas for the latest refresh.""" 
        counts = self._get_afss_level_counts()    
        total_count = sum(counts.values())    
        active_count = len(active_images) 
        previous_counts = previous_counts or {}
        LOGGER.info(
            colorstr(    
                "bold",
                "bright_yellow",     
                "AFSS update "
                f"epoch={epoch} total={total_count} "
                f"active={active_count} {self._format_afss_delta(active_count, previous_active_count)}, "     
                f"easy={counts['easy']} {self._format_afss_delta(counts['easy'], previous_counts.get('easy'))}, " 
                f"middle={counts['moderate']} "     
                f"{self._format_afss_delta(counts['moderate'], previous_counts.get('moderate'))}, "
                f"hard={counts['hard']} {self._format_afss_delta(counts['hard'], previous_counts.get('hard'))}",  
            )
        )  
 
    def _format_afss_startup_summary(self) -> str: 
        """Build a colored one-line summary of the active AFSS settings."""
        fields = (
            "afss",
            "afss_warmup_epochs",
            "afss_update_interval",
            "afss_conf",   
            "afss_easy_ratio",
            "afss_moderate_ratio",    
            "afss_easy_forced_gap",
            "afss_moderate_forced_gap",
            "afss_save_refresh_json",     
            "afss_thresholds",
        )
        summary = ", ".join(f"{field}={getattr(self.args, field)!r}" for field in fields) 
        return f"{colorstr('bold', 'magenta', 'AFSS Summary')} {colorstr('cyan', summary)}" 
  
    def _on_afss_train_start(self, trainer) -> None:  
        """Log the AFSS config once when the owning AFSS trainer starts training."""  
        if trainer is not self or not self.afss_enabled or self.afss_startup_summary_logged:
            return
        LOGGER.info(self._format_afss_startup_summary())
        self.afss_startup_summary_logged = True 

    def _get_full_train_eval_loader(self):
        """Build a non-shuffled loader over the full train split for AFSS scoring."""
        batch_size = self.batch_size // max(self.world_size, 1) 
        return self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="val")   

    def _get_afss_evaluator(self):
        """Create the detect AFSS evaluator bound to the full train split."""
        args = copy(self.args)  
        args.split = "train"    
        args.conf = self.args.afss_conf   
        args.plots = False
        args.save_json = False
        args.save_txt = False 
        evaluator_cls = self.afss_evaluator_cls or AFSSDetectionEvaluator   
        return evaluator_cls( 
            dataloader=self._get_full_train_eval_loader(),
            save_dir=self.afss_dir / "eval",    
            args=args,     
            _callbacks=self.callbacks,   
        )   

    def _evaluate_full_train_split(self) -> dict[str, dict[str, object]]: 
        """Run AFSS detect evaluation over the full train split and return image-level payloads."""  
        if getattr(self, "loss_items", None) is None:   
            self.loss_items = torch.zeros(len(getattr(self, "loss_names", ("box_loss", "cls_loss", "dfl_loss"))))    
        evaluator = self._get_afss_evaluator()
        evaluator(trainer=self)
        self.afss_evaluator = evaluator    
        return evaluator.image_results   

    def _update_afss_state_from_results(self, image_results: dict[str, dict[str, object]], epoch: int) -> None: 
        """Refresh AFSS state using the latest image-level sufficiency payloads."""
        moderate_threshold, easy_threshold = self.args.afss_thresholds[self.afss_task_name] 
        for im_file, payload in image_results.items():
            state = self.afss_state.setdefault(im_file, AFSSImageState(im_file=im_file))    
            state.metrics = payload["metrics"]
            state.task_score = float(payload["task_score"])  
            state.level = classify_score(state.task_score, moderate_threshold, easy_threshold)    
            state.last_eval_epoch = epoch   

    def _select_active_train_images(self, epoch: int) -> list[str]: 
        """Select the next AFSS-managed active image subset for detection training."""
        moderate_threshold, easy_threshold = self.args.afss_thresholds[self.afss_task_name]  
        return select_active_images( 
            self.afss_state,
            current_epoch=epoch,     
            easy_ratio=self.args.afss_easy_ratio,
            moderate_ratio=self.args.afss_moderate_ratio,
            easy_forced_gap=self.args.afss_easy_forced_gap,     
            moderate_forced_gap=self.args.afss_moderate_forced_gap,
            moderate_threshold=moderate_threshold,
            easy_threshold=easy_threshold,  
        )
   
    def _serialize_sorted_afss_images(self) -> list[dict[str, Any]]:   
        """Return AFSS state entries sorted from easy to hard for debug snapshots."""
        return [
            asdict(state)
            for state in sorted(
                self.afss_state.values(),     
                key=lambda state: (-state.task_score, state.im_file),
            ) 
        ]    
  
    def _write_afss_refresh_snapshot(self, epoch: int, active_images: list[str]) -> Path | None:
        """Persist a complete AFSS refresh snapshot when debug JSON export is enabled.""" 
        if not self._should_save_afss_refresh_json():  
            return None
        moderate_threshold, easy_threshold = self.args.afss_thresholds[self.afss_task_name] 
        path = self.afss_dir / f"refresh_epoch{epoch:04d}.json"     
        payload = {    
            "epoch": epoch,
            "task": self.afss_task_name,
            "thresholds": {"moderate": moderate_threshold, "easy": easy_threshold}, 
            "counts": self._get_afss_level_counts(),
            "active_count": len(active_images),
            "active_images": sorted(active_images),
            "images": self._serialize_sorted_afss_images(), 
        }
        return save_state(path, payload)   
    
    def _rebuild_train_loader_from_list(self, list_path: str | Path):
        """Rebuild only this trainer's train loader from the generated AFSS active list."""
        list_path = Path(list_path)
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(str(list_path), batch_size=batch_size, rank=LOCAL_RANK, mode="train")    
        self.afss_active_list_path = list_path     
        match = re.search(r"train_epoch(\d+)$", list_path.stem)
        if match:
            self.afss_active_list_epoch = int(match.group(1))
        return self.train_loader     

    def save_model(self):
        """Save model checkpoints with AFSS resume metadata included."""
        import io

        buffer = io.BytesIO() 
        torch.save(  
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,    
                "model": None,     
                "ema": deepcopy(unwrap_model(self.ema.ema)),     
                "updates": self.ema.updates, 
                "optimizer": deepcopy(self.optimizer.state_dict()),     
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),     
                "afss_resume": self._get_afss_resume_metadata() if self.afss_enabled else None,
                **self._swanlab_checkpoint_metadata(),   
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,  
                },     
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",    
            },
            buffer,   
        ) 
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)  
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:  
            self.best.write_bytes(serialized_ckpt)    
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)    
     
    def resume_training(self, ckpt):
        """Resume standard training state first, then restore AFSS-specific loader state."""   
        super().resume_training(ckpt)
        self._restore_afss_resume_state(ckpt)
     
    def _refresh_afss_epoch(self, epoch: int) -> None:     
        """Evaluate the full train split, select active images, and rebuild the AFSS train loader."""
        previous_counts = self._get_afss_level_counts()   
        previous_active_count = self._get_train_loader_image_count() 
        image_results = self._evaluate_full_train_split()
        self._update_afss_state_from_results(image_results, epoch)  
        active_images = self._select_active_train_images(epoch)
        self._log_afss_update_summary(epoch, previous_counts, previous_active_count, active_images)
        active_list = self._write_active_train_list(active_images, epoch)
        self._rebuild_train_loader_from_list(active_list)    
        self._mark_last_used_epoch(active_images, epoch)    
        self._write_afss_refresh_snapshot(epoch, active_images)   
        self._save_afss_state()     
   
    def _on_afss_train_epoch_start(self, trainer) -> None:
        """Keep AFSS scheduling isolated to the explicit AFSS trainer path."""
        if trainer is not self or not self.afss_enabled:
            return
        image_files = list(getattr(self.train_loader.dataset, "im_files", []))
        self._initialize_afss_state(image_files)
        if self._use_full_dataset(self.epoch):
            self._mark_last_used_epoch(image_files, self.epoch)     
            return
        if self._should_refresh_afss(self.epoch):
            self._refresh_afss_epoch(self.epoch)
            return     
        current_files = list(getattr(self.train_loader.dataset, "im_files", []))   
        self._mark_last_used_epoch(current_files, self.epoch)
