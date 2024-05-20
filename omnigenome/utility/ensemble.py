







from typing import List

import numpy as np


class VoteEnsemblePredictor:
    def __init__(
        self,
        predictors: [List, dict],
        weights: [List, dict] = None,
        numeric_agg="average",
        str_agg="max_vote",
    ):
        
        if weights is not None:
            assert len(predictors) == len(
                weights
            ), "Checkpoints and weights should have the same length"
            assert type(predictors) == type(
                weights
            ), "Checkpoints and weights should have the same type"

        assert len(predictors) > 0, "Checkpoints should not be empty"

        self.numeric_agg_methods = {
            "average": np.mean,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "mode": lambda x: max(set(x), key=x.count),
            "sum": np.sum,
        }
        self.str_agg_methods = {
            "max_vote": lambda x: max(set(x), key=x.count),
            "min_vote": lambda x: min(set(x), key=x.count),
            "vote": lambda x: max(set(x), key=x.count),
            "mode": lambda x: max(set(x), key=x.count),
        }
        assert (
            numeric_agg in self.numeric_agg_methods
        ), "numeric_agg should be either: " + str(self.numeric_agg_methods.keys())
        assert (
            str_agg in self.str_agg_methods
        ), "str_agg should be either max or vote" + str(self.str_agg_methods.keys())

        self.numeric_agg_func = numeric_agg
        self.str_agg = self.str_agg_methods[str_agg]

        if isinstance(predictors, dict):
            self.checkpoints = list(predictors.keys())
            self.predictors = predictors
            self.weights = (
                list(weights.values()) if weights else [1] * len(self.checkpoints)
            )
        else:
            raise NotImplementedError(
                "Only support dict type for checkpoints and weights"
            )

    def numeric_agg(self, result: list):
        
        res = np.stack([np.array(x) for x in result])
        return self.numeric_agg_methods[self.numeric_agg_func](res, axis=0)

    def __ensemble(self, result: dict):
        
        if isinstance(result, dict):
            return self.__dict_aggregate(result)
        elif isinstance(result, list):
            return self.__list_aggregate(result)
        else:
            return result

    def __dict_aggregate(self, result: dict):
        
        ensemble_result = {}
        for k, v in result.items():
            if isinstance(result[k], list):
                ensemble_result[k] = self.__list_aggregate(result[k])
            elif isinstance(result[k], dict):
                ensemble_result[k] = self.__dict_aggregate(result[k])
            else:
                ensemble_result[k] = result[k]
        return ensemble_result

    def __list_aggregate(self, result: list):
        if not isinstance(result, list):
            result = [result]

        assert all(
            isinstance(x, (type(result[0]))) for x in result
        ), "all type of result should be the same"

        if isinstance(result[0], list):
            for i, k in enumerate(result):
                result[i] = self.__list_aggregate(k)
            
            try:
                new_result = self.numeric_agg(result)
            except Exception as e:
                try:
                    new_result = self.str_agg(result)
                except Exception as e:
                    new_result = result
            return [new_result]

        elif isinstance(result[0], dict):
            for k in result:
                result[k] = self.__dict_aggregate(result[k])
            return result

        
        try:
            new_result = self.numeric_agg(result)
        except Exception as e:
            try:
                new_result = self.str_agg(result)
            except Exception as e:
                new_result = result

        return new_result

    def predict(self, text, ignore_error=False, print_result=False):
        
        
        result = {}
        
        for ckpt, predictor in self.predictors.items():
            
            raw_result = predictor.inference(
                text, ignore_error=ignore_error, print_result=print_result
            )
            
            for key, value in raw_result.items():
                
                if key not in result:
                    
                    result[key] = []
                
                for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                    result[key].append(value)
        
        return self.__ensemble(result)

    def batch_predict(self, texts, ignore_error=False, print_result=False):
        
        batch_raw_results = []
        for ckpt, predictor in self.predictors.items():
            if hasattr(predictor, "inference"):
                raw_results = predictor.inference(
                    texts,
                    ignore_error=ignore_error,
                    print_result=print_result,
                    merge_results=False,
                )
            else:
                raw_results = predictor.inference(
                    texts, ignore_error=ignore_error, print_result=print_result
                )
            batch_raw_results.append(raw_results)

        batch_results = []
        for raw_result in batch_raw_results:
            for i, result in enumerate(raw_result):
                if i >= len(batch_results):
                    batch_results.append({})
                for key, value in result.items():
                    if key not in batch_results[i]:
                        batch_results[i][key] = []
                    for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                        batch_results[i][key].append(value)

        ensemble_results = []
        for result in batch_results:
            ensemble_results.append(self.__ensemble(result))
        return ensemble_results

    
    
    
    
    
    
