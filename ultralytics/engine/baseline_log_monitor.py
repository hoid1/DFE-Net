# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license    

from __future__ import annotations  
  
import csv  
import math 
from dataclasses import dataclass 
from pathlib import Path
from typing import Any 
    
   
@dataclass     
class BaselineLogDecision:
    active: bool
    should_stop: bool    
    epoch: int     
    metric: str
    current: float | None = None
    baseline: float | None = None
    diff: float | None = None
    bad_epochs: int = 0
    patience: int = 1
    reason: str = ""    


class BaselineLogMonitor:
    def __init__(    
        self,
        baseline_by_epoch: dict[int, dict[str, float]],     
        metric: str,
        start_epoch: int,
        min_gap: float,
        patience: int,
    ):
        self.baseline_by_epoch = baseline_by_epoch  
        self.metric = metric
        self.start_epoch = max(1, int(start_epoch))
        self.min_gap = float(min_gap)   
        self.patience = max(1, int(patience))  
        self.bad_epochs = 0

    @classmethod
    def from_args(
        cls,     
        enabled: bool,
        path: str | None,
        metric: str,
        total_epochs: int,   
        start_ratio: float,
        min_gap: float,    
        patience: int,
    ) -> BaselineLogMonitor | None:
        if not enabled: 
            return None    
        if not path:     
            raise ValueError("baseline_path is required when baseline_monitor=True")

        start_epoch = max(1, int(math.floor(int(total_epochs) * float(start_ratio))))     
        return cls(    
            baseline_by_epoch=cls._load_results_csv(path, metric),
            metric=metric,    
            start_epoch=start_epoch,   
            min_gap=min_gap,     
            patience=patience,
        )
   
    @staticmethod
    def _load_results_csv(path: str | Path, metric: str) -> dict[int, dict[str, float]]:   
        log_path = Path(path)
        if not log_path.exists():     
            raise FileNotFoundError(f"Baseline results.csv does not exist: {log_path}")   

        baseline_by_epoch: dict[int, dict[str, float]] = {}
        with log_path.open(newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)     
            if reader.fieldnames:
                reader.fieldnames = [field.strip() for field in reader.fieldnames]
            for row in reader:
                normalized = {key.strip(): value for key, value in row.items() if key is not None}
                epoch = BaselineLogMonitor._to_int(normalized.get("epoch"))  
                if epoch is None:
                    continue 
                value = BaselineLogMonitor._to_float(normalized.get(metric)) 
                if value is not None:
                    baseline_by_epoch[epoch] = {metric: value}
                else:   
                    baseline_by_epoch[epoch] = {}
        return baseline_by_epoch    
    
    def check(self, metrics: dict[str, Any] | None, epoch: int) -> BaselineLogDecision:     
        epoch = int(epoch)
        metrics = metrics or {}
        baseline_record = self.baseline_by_epoch.get(epoch)
        current = self._to_float(metrics.get(self.metric))   
        baseline = baseline_record.get(self.metric) if baseline_record is not None else None
        diff = current - baseline if current is not None and baseline is not None else None

        if epoch < self.start_epoch:
            return BaselineLogDecision(
                active=False,
                should_stop=False,   
                epoch=epoch,
                metric=self.metric,
                current=current,
                baseline=baseline,   
                diff=diff,    
                bad_epochs=self.bad_epochs,   
                patience=self.patience,
                reason=f"waiting for start_epoch={self.start_epoch}",    
            )
  
        if baseline_record is None:  
            return BaselineLogDecision(   
                active=True,
                should_stop=False,
                epoch=epoch,
                metric=self.metric,
                current=current,     
                bad_epochs=self.bad_epochs,
                patience=self.patience,     
                reason="baseline epoch is missing",
            ) 
   
        if current is None or baseline is None:   
            return BaselineLogDecision(
                active=True, 
                should_stop=False,
                epoch=epoch,    
                metric=self.metric,
                current=current,
                baseline=baseline,    
                bad_epochs=self.bad_epochs, 
                patience=self.patience,
                reason="metric is missing",
            )     
   
        if diff < -self.min_gap:
            self.bad_epochs += 1
        else:  
            self.bad_epochs = 0
 
        should_stop = self.bad_epochs >= self.patience    
        reason = ""
        if should_stop:
            reason = (
                f"current {self.metric} is lower than baseline by more than {self.min_gap:.4f} "  
                f"for {self.bad_epochs} consecutive epochs"
            )

        return BaselineLogDecision(    
            active=True,
            should_stop=should_stop,
            epoch=epoch,     
            metric=self.metric,
            current=current,  
            baseline=baseline,     
            diff=diff, 
            bad_epochs=self.bad_epochs,    
            patience=self.patience,   
            reason=reason,
        )   

    @staticmethod     
    def _to_float(value: Any) -> float | None:
        try:    
            return float(value)  
        except (TypeError, ValueError):
            return None    

    @staticmethod 
    def _to_int(value: Any) -> int | None:  
        try:
            return int(float(value))     
        except (TypeError, ValueError):
            return None  
