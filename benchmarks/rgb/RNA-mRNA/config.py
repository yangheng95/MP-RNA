









import os

import numpy as np
import torch

from omnigenome import (
    AutoBenchConfig,
    OmniGenomeDatasetForTokenRegression,
    OmniGenomeModelForTokenRegressionWith2DStructure,
    OmniGenomeModelForTokenRegression,
)


class Dataset(OmniGenomeDatasetForTokenRegression):
    def __init__(self, data_source, tokenizer, max_length, **kwargs):
        super().__init__(data_source, tokenizer, max_length, **kwargs)

    def prepare_input(self, instance, **kwargs):
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        instance["sequence"] = f'{instance["sequence"]}'
        tokenized_inputs = self.tokenizer(
            instance["sequence"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = [instance[target_col] for target_col in target_cols]
        labels = np.concatenate(
            [
                np.array(labels),
                np.array(
                    [
                        [-100]
                        * (len(tokenized_inputs["input_ids"][0]) - len(labels[0])),
                        [-100]
                        * (len(tokenized_inputs["input_ids"][0]) - len(labels[0])),
                        [-100]
                        * (len(tokenized_inputs["input_ids"][0]) - len(labels[0])),
                    ]
                ),
            ],
            axis=1,
        ).T
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()
        return tokenized_inputs


def compute_metrics(labels, logits):
    mask = labels != -100
    filtered_logits = logits[mask]
    filtered_labels = labels[mask]
    mse_columns = []
    for i in range(labels.shape[2]):
        mse = np.mean(
            (
                filtered_logits.reshape(labels.shape[0], -1, labels.shape[2])[:, :, i]
                - filtered_labels.reshape(labels.shape[0], -1, labels.shape[2])[:, :, i]
            )
            ** 2
        )
        mse_columns.append(mse**0.5)

    mcrmse = np.mean(mse_columns, axis=0)
    metrics = {"mc_rmse": mcrmse}
    return metrics



config_dict = {
    "task_name": "RNA-mRNA",
    "task_type": "token_regression",
    "label2id": None,  
    "num_labels": 3,  
    "epochs": 20,
    "learning_rate": 2e-5,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "max_length": 512,  
    "seeds": [45, 46, 47],
    "compute_metrics": compute_metrics,
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json")
    else None,
    
    "dataset_cls": Dataset,
    "model_cls": OmniGenomeModelForTokenRegression,
}

bench_config = AutoBenchConfig(config_dict)
