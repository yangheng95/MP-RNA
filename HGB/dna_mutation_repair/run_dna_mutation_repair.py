# -*- coding: utf-8 -*-
# file: run_mutation_detection.py
# time: 00:57 21/01/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.


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


def synthesize_mutation_sequence(seq, ops=None, noise_ratio=None):
    """
    向文本添加噪声。
    可以通过删除、替换或插入单词来引入噪声。
    """
    if ops is None:
        ops = ["delete", "replace", "insert"]

    n_tokens = len(seq)
    if noise_ratio is not None:
        n_noise = int(random.random() * noise_ratio * n_tokens)
    else:
        n_noise = 1
    mutation_tokens = list(seq)
    noise_indices = random.sample(range(n_tokens), n_noise)

    for i in noise_indices:
        noise_type = random.choice(ops)
        if noise_type == "delete":
            mutation_tokens[i] = ""
        elif noise_type == "replace":
            mutation_tokens[i] = random.choice(["A", "C", "G", "T"])
        elif noise_type == "insert":
            mutation_tokens.insert(i, random.choice(["A", "C", "G", "T"]))

    mutation_seq = "".join(mutation_tokens)
    return mutation_seq


class RNAClassificationDataset(RNADataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None, **kwargs):
        self.token2id = {
            k: i for i, k in enumerate(["A", "C", "G", "T", "N"])
        }
        super().__init__(data_file, tokenizer, max_seq_len, device, **kwargs)

    def encode(self, example):
        # mutate RNA sequence
        example["mut"] = example["mut"].replace("U", "T")
        # example["mut"] = synthesize_mutation_sequence(example["seq"], noise_ratio=None)
        # example["mut"] = synthesize_mutation_sequence(example["seq"], ops=['replace'], noise_ratio=0.05)
        tokenized_seq = self.tokenize(example["seq"], add_special_tokens=False)
        tokenized_mut_seq = self.tokenize(example["mut"], add_special_tokens=False)

        ori_tokens = list(example["seq"])
        mut_tokens = list(example["mut"])
        repair_labels = np.array(
            [self.token2id[ori] if ori != mut else -100 for (ori, mut) in zip(ori_tokens, mut_tokens)])

        if "attention_mask" in tokenized_seq:
            data = {
                "input_ids": tokenized_mut_seq["input_ids"],
                "attention_mask": tokenized_seq["attention_mask"],
                "label": torch.tensor(repair_labels),
            }
        else:
            data = {
                "input_ids": tokenized_mut_seq["input_ids"],
                "label": torch.tensor(repair_labels),
            }
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if "attention_mask" in self.data[idx]:
            return {
                "input_ids": self.data[idx]["input_ids"].to(self.device),
                "attention_mask": self.data[idx]["attention_mask"].to(self.device),
                "label": self.data[idx]["label"].long().to(self.device),
            }
        else:
            return {
                "input_ids": self.data[idx]["input_ids"].to(self.device),
                "label": self.data[idx]["label"].long().to(self.device),
            }


if __name__ == "__main__":

    mv = MetricVisualizer('run_mutation_repair')
    epochs = 10
    learning_rate = 2e-5
    weight_decay = 1e-5
    batch_size = 32
    device = autocuda.auto_cuda()

    for model_name in [
        # 'DNABERT-2-117M',
        # 'nucleotide-transformer-500m-human-ref',
        # 'nucleotide-transformer-v2-50m-multi-species',
        # 'nucleotide-transformer-v2-100m-multi-species',
        # 'hyenadna-small-32k-seqlen-hf',
        # # 'hyenadna-medium-160k-seqlen-hf',
        # 'hyenadna-medium-450k-seqlen-hf',
        # 'hyenadna-large-1m-seqlen-hf',
        # 'RNAFM',
        'splicebert-510nt',
        # 'codonbert',
        # 'cdsbert',
        # '3utrbert',
        # 'mlm_checkpoint',
        "mprna_small",
        # "esm2_rna_checkpoint",
        # 'agro-nucleotide-transformer-1b'
    ]:
        if '1b' in model_name or '500m' in model_name:
            batch_size = 4

        model_path = findfile.find_dir('../', model_name)

        for seed in [42, 43, 44]:
            seed_everything(seed)

            from benchmark_V2.benchmark_utils import get_token_classification_model

            model_path = findfile.find_dir('../', [model_name, 'pretrained_models'])
            model, tokenizer = get_token_classification_model(model_path, num_labels=5)

            # write a function to calculate trainable parameters
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)


            print(f'{model_name} has {count_parameters(model) / 1e6:.2f}M trainable parameters')

            base_dir = "./"
            # 1. 加载数据集
            train_set = RNAClassificationDataset(base_dir + "dataset/train.json", tokenizer, device=device)
            valid_set = RNAClassificationDataset(base_dir + "dataset/valid.json", tokenizer, device=device)
            test_set = RNAClassificationDataset(base_dir + "dataset/test.json", tokenizer, device=device)
            model.config.id2label = {v: k for k, v in train_set.token2id.items()}


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
                    loss = loss_fn(outputs.view(-1, 5), batch['label'].view(-1))
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
                    if valid_metrics['accuracy'] > global_acc:
                        torch.save(model.state_dict(), "model.pt")
                        global_acc = valid_metrics['accuracy']

            # Testing
            model.load_state_dict(torch.load("model.pt"))
            model.eval()
            preds = []
            truth = []
            with torch.no_grad():
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

                mv.dump('result.mv')
                mv.raw_summary(round=4)
                mv.summary(round=4)
                mv.to_txt('result.txt')
