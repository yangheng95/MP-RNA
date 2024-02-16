# -*- coding: utf-8 -*-
# file: run_mutation_detection.py
# time: 00:57 21/01/2024


import json
import os
import pickle
import random

import autocuda
import findfile
import numpy as np
import sklearn.metrics as metrics
import torch
from metric_visualizer import MetricVisualizer
from tqdm import tqdm

from benchmark_V2.benchmark_utils import seed_everything, RNADataset


seed_everything()


def convert_rna_structure(structure):
    """
    将RNA结构表示转换为只包含'(', ')', '.'的格式。

    参数:
    structure (str): 原始RNA结构表示字符串，可能包含多种符号。

    返回:
    str: 转换后的RNA结构表示。
    """
    converted = ""
    for char in structure:
        if char in "().":
            converted += char
        else:
            converted += "."
    return converted


class RNAClassificationDataset(RNADataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None, **kwargs):
        self.structure2id = {
            "(": 0,
            ")": 1,
            ".": 2,
            # "<": 3,
            # ">": 4,
            # "[": 5,
            # "]": 6,
            # "{": 7,
            # "}": 8,
            # "B": 9,
            # "C": 10,
            # "D": 11,
            # "b": 12,
            # "c": 13,
            # "d": 14,
            # "A": 15,
            # "a": 16,
            # "E": 17,
            # "F": 18,
            # "e": 19,
            # "f": 20,
            # "G": 21,
            # "g": 22,
            # "H": 23,
            # "h": 24,
            # "I": 25,
            # "i": 26,
            # "J": 27,
            # "K": 28,
            # "j": 29,
            # "k": 30,
            # "O": 31,
            # "Q": 32,
        }
        super().__init__(data_file, tokenizer, max_seq_len, device, **kwargs)

    def encode(self, example):
        example["seq"] = example["seq"]
        # example["label"] = convert_rna_structure(example["label"])
        tokenized_seq = self.tokenize(example["seq"], add_special_tokens=False)
        label = [self.structure2id.get(x, -100) for x in example["label"]]

        if "attention_mask" in tokenized_seq:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "attention_mask": tokenized_seq["attention_mask"],
                "label": torch.tensor(label),
            }
        else:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "label": torch.tensor(label),
            }
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if "attention_mask" in self.data[idx]:
            return {
                "input_ids": self.data[idx]["input_ids"].to(self.device),
                "attention_mask": self.data[idx]["attention_mask"].to(self.device),
                "label": self.data[idx]["label"].to(self.device),
            }
        else:
            return {
                "input_ids": self.data[idx]["input_ids"].to(self.device),
                "label": self.data[idx]["label"].to(self.device),
            }


if __name__ == "__main__":

    epochs = 20
    learning_rate = 2e-5
    weight_decay = 1e-5
    batch_size = 8
    seeds = [42, 43, 44]
    device = autocuda.auto_cuda()

    for dataset_path in findfile.find_dirs('dataset', [''], exclude_key=['rfam', 'bprna'], rescursive=1):
        mv = MetricVisualizer('run_structure_prediction_strand2/' + dataset_path.split('/')[-1])

        for model_name in [
            # 'DNABERT-2-117M',
            # 'nucleotide-transformer-500m-human-ref',
            # 'nucleotide-transformer-v2-50m-multi-species',
            # 'nucleotide-transformer-v2-100m-multi-species',
            # 'hyenadna-small-32k-seqlen-hf',
            # # 'hyenadna-medium-160k-seqlen-hf',
            # 'hyenadna-medium-450k-seqlen-hf',
            # 'hyenadna-large-1m-seqlen-hf',
            # # 'RNAFM',
            # 'splicebert-510nt',
            # 'codonbert',
            # 'cdsbert',
            # '3utrbert',
            # 'mlm_checkpoint',
            "mprna_small",
            # # "esm2_rna_checkpoint",
            # 'agro-nucleotide-transformer-1b'
        ]:
            if '1b' in model_name or '500m' in model_name:
                batch_size = 4
                epochs = 10

            model_path = findfile.find_dir('../', model_name)
            dataset_name = dataset_path.split('/')[-1]
            for seed in seeds:
                seed_everything(seed)
                from benchmark_V2.benchmark_utils import get_token_classification_model

                model_path = findfile.find_dir('../', [model_name, 'pretrained_models'])
                model, tokenizer = get_token_classification_model(model_path, num_labels=3)

                # write a function to calculate trainable parameters
                def count_parameters(model):
                    return sum(p.numel() for p in model.parameters() if p.requires_grad)


                print(f'{model_name} has {count_parameters(model) / 1e6:.2f}M trainable parameters')

                base_dir = "./"
                # 1. 加载数据集
                data_dir = findfile.find_cwd_dir(['dataset'] + dataset_name.split('/'))
                # 1. 加载数据集
                train_set = RNAClassificationDataset(base_dir + data_dir + "/train.json", tokenizer, device=device)
                valid_set = RNAClassificationDataset(base_dir + data_dir + "/valid.json", tokenizer, device=device)
                test_set = RNAClassificationDataset(base_dir + data_dir + "/test.json", tokenizer, device=device)

                model.config.id2label = {v: k for k, v in train_set.structure2id.items()}


                def compute_metrics(p):
                    predictions, labels = p

                    # accuracy_score = metrics.accuracy_score([str(x) for x in labels], [str(x) for x in predictions])
                    # results = {
                    #     "accuracy": accuracy_score,
                    # }
                    # Remove ignored index (special tokens)
                    true_predictions = [
                        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)
                    ]
                    true_labels = [
                        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)
                    ]

                    def flatten(l):
                        return [item for sublist in l for item in sublist]

                    true_labels = flatten(true_labels)
                    true_predictions = flatten(true_predictions)

                    classification_report = metrics.classification_report(true_labels, true_predictions, digits=4)
                    print(classification_report)
                    results = {
                        "accuracy": metrics.accuracy_score(true_labels, true_predictions),
                        "precision": metrics.precision_score(true_labels, true_predictions, average="macro"),
                        "recall": metrics.recall_score(true_labels, true_predictions, average="macro"),
                        "f1": metrics.f1_score(true_labels, true_predictions, average="macro"),
                    }
                    print(results)
                    return results


                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
                valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)

                # 4. 训练参数设置
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                loss_fn = torch.nn.CrossEntropyLoss()
                model.to(device)
                global_acc = -np.inf
                for epoch in range(epochs):
                    model.train()
                    train_loss = []
                    train_it = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Loss:")
                    for batch in train_it:
                        outputs = model(batch)
                        outputs = torch.nn.functional.softmax(outputs, dim=-1)
                        loss = loss_fn(outputs.view(-1, len(model.config.id2label)), batch['label'].view(-1))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        train_loss.append(loss.item())
                        train_it.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {np.average(train_loss):.4f}")
                    with torch.no_grad():
                        model.eval()
                        val_truth = []
                        val_preds = []
                        for batch in valid_loader:
                            outputs = model(batch)
                            outputs = torch.nn.functional.softmax(outputs, dim=-1)
                            val_truth.append(batch['label'].detach().cpu().numpy())
                            val_preds.append(outputs.detach().cpu().argmax(-1).numpy())

                        val_truth = np.concatenate(val_truth)
                        val_preds = np.concatenate(val_preds)
                        valid_metrics = compute_metrics((val_preds, val_truth))
                        if valid_metrics['f1'] > global_acc:
                            torch.save(model.state_dict(), "model.pt")
                            global_acc = valid_metrics['f1']

                with torch.no_grad():
                    # Testing
                    model.load_state_dict(torch.load("model.pt"))
                    model.eval()
                    preds = []
                    truth = []
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
                    it = tqdm(test_loader, desc="Testing")
                    for batch in it:
                        outputs = model(batch)
                        outputs = torch.nn.functional.softmax(outputs, dim=-1)
                        preds.append(outputs.detach().cpu().argmax(dim=-1).numpy())
                        truth.append(batch["label"].detach().cpu().numpy())

                    preds = np.concatenate(preds)
                    truth = np.concatenate(truth)
                    test_metrics = compute_metrics((preds, truth))

                    # mv.log_metric(model_name, 'accuracy', test_metrics['accuracy'])
                    # mv.log_metric(model_name, 'precision', test_metrics['precision'])
                    # mv.log_metric(model_name, 'recall', test_metrics['recall'])
                    mv.log_metric(model_name, 'f1', test_metrics['f1'])

                    os.makedirs('run_structure_prediction_strand2/', exist_ok=True)
                    mv.dump('run_structure_prediction_strand2/' + dataset_path.split('/')[-1] + '.mv')
                    mv.raw_summary(round=4)
                    mv.summary(round=4)
                    mv.to_txt('run_structure_prediction_strand2/' + dataset_path.split('/')[-1] + '.txt')
