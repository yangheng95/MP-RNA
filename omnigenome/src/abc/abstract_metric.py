







import numpy as np
import sklearn.metrics as metrics

from ..misc.utils import env_meta_info


class OmniGenomeMetric:
    

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        self.metric_func = metric_func
        self.ignore_y = ignore_y

        for metric in metrics.__dict__.keys():
            setattr(self, metric, metrics.__dict__[metric])

        self.metadata = env_meta_info()

    def compute(self, y_true, y_pred) -> dict:
        
        raise NotImplementedError(
            "Method compute() is not implemented in the child class. "
            "This function returns a dict containing the metric name and value."
            "e.g. {'accuracy': 0.9}"
        )

    @staticmethod
    def flatten(y_true, y_pred):
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return y_true, y_pred
