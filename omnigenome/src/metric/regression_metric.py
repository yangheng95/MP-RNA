









import types
import warnings

import numpy as np
import sklearn.metrics as metrics

from ..abc.abstract_metric import OmniGenomeMetric


class RegressionMetric(OmniGenomeMetric):
    

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        super().__init__(metric_func, ignore_y, *args, **kwargs)
        self.kwargs = kwargs

    def __getattribute__(self, name):
        
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            setattr(self, "compute", metric_func)
            

            def wrapper(y_true, y_score, *args, **kwargs):
                
                y_true, y_score = RegressionMetric.flatten(y_true, y_score)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_score = y_score[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))
                kwargs.update(self.kwargs)

                return {name: self.compute(y_true, y_score, *args, **kwargs)}

            return wrapper
        else:
            return super().__getattribute__(name)

    def compute(self, y_true, y_score, *args, **kwargs):
        
        if self.metric_func is not None:
            kwargs.update(self.kwargs)
            return self.metric_func(y_true, y_score, *args, **kwargs)

        else:
            raise NotImplementedError(
                "Method compute() is not implemented in the child class."
            )
