









import types
import warnings

import numpy as np
import sklearn.metrics as metrics

from ..abc.abstract_metric import OmniGenomeMetric


class RankingMetric(OmniGenomeMetric):
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            
            def wrapper(y_true, y_score, *args, **kwargs):
                
                y_true, y_score = RankingMetric.flatten(y_true, y_score)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_score = y_score[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))

                return {name: self.compute(y_true, y_score, *args, **kwargs)}

            return wrapper
        raise AttributeError(f"'CustomMetrics' object has no attribute '{name}'")

    def compute(self, y_true, y_score, *args, **kwargs):
        
        raise NotImplementedError(
            "Method compute() is not implemented in the child class."
        )
