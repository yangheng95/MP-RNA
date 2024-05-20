









import os

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig,
    OmniGenomeDatasetForTokenClassification,
    OmniGenomeModelForTokenClassificationWith2DStructure,
    OmniGenomeModelForTokenClassification,
)

label2id = {"(": 0, ")": 1, ".": 2}


config_dict = {
    "task_name": "RNA-SSP-bpRNA",
    "task_type": "token_classification",
    "label2id": label2id,  
    "num_labels": None,  
    "epochs": 10,
    "learning_rate": 2e-5,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "max_length": 512,  
    "seeds": [45, 46, 47],
    "compute_metrics": ClassificationMetric(ignore_y=-100, average="macro").f1_score,
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json")
    else None,
    
    "dataset_cls": OmniGenomeDatasetForTokenClassification,
    "model_cls": OmniGenomeModelForTokenClassification,
}

bench_config = AutoBenchConfig(config_dict)
