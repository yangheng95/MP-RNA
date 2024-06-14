import random

import autocuda
import torch
from transformers import AutoTokenizer

from omnigenome.src.dataset.omnigenome_dataset import (
    OmniGenomeDatasetForTokenClassification,
)
from omnigenome.src.metric.classification_metric import ClassificationMetric
from omnigenome.src.model.classiifcation.model import (
    OmniGenomeModelForTokenClassification,
    OmniGenomeModelForTokenClassificationWith2DStructure,
)
from omnigenome import OmniGenomeTokenizer, ModelHub
from omnigenome.src.trainer.trainer import Trainer

label2id = {"(": 0, ")": 1, ".": 2}

model_name_or_path = "anonymous8/MP-RNA-186M"

SN_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# The OmniGenome+ in the paper
model = OmniGenomeModelForTokenClassificationWith2DStructure(
    model_name_or_path,
    tokenizer=SN_tokenizer,
    label2id=label2id
)
# The OmniGenome in the paper
# model = OmniGenomeModelForTokenClassification(
#     model_name_or_path,
#     tokenizer=SN_tokenizer,
#     label2id=label2id
# )

epochs = 10
learning_rate = 2e-5
weight_decay = 1e-5
batch_size = 16
seed = random.randint(0, 1000)

train_file = "../benchmarks/rgb/RNA-SSP-Archive2/train.json"
test_file = "../benchmarks/rgb/RNA-SSP-Archive2/test.json"
valid_file = "../benchmarks/rgb/RNA-SSP-Archive2/valid.json"

train_set = OmniGenomeDatasetForTokenClassification(
    data_source=train_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)
test_set = OmniGenomeDatasetForTokenClassification(
    data_source=test_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)
valid_set = OmniGenomeDatasetForTokenClassification(
    data_source=valid_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

compute_metrics = ClassificationMetric(ignore_y=-100, average="macro").f1_score

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    eval_loader=valid_loader,
    test_loader=test_loader,
    batch_size=batch_size,
    epochs=epochs,
    patience=3,
    optimizer=optimizer,
    compute_metrics=compute_metrics,
    seed=seed,
    device=autocuda.auto_cuda(),
)

metrics = trainer.train()
model.save("MP-RNA-186M", overwrite=True)
model.load("MP-RNA-186M")



model = ModelHub.load("MP-RNA-186M")
output = model.inference(
    [
        "GCCCGAAUAGCUCAGCCGGUUAGAGCACUUGACUGUUAAUCAGGGGGUCGUUGGUUCGAGUCCAACUUCGGGCGCCA",
        "GCCCGAAUAGCUCAGCCGGUUAGAGCACUUGACUGUUAAUCAGGGGGUCGUUGGUUCGAGUCCAACUUCGGGCGCCA",
    ]
)
print(output["predictions"])
