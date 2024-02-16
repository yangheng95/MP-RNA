# -*- coding: utf-8 -*-
# file: run_mutation_fix.py
# time: 00:32 13/01/2024
import json
import os
import pickle

import autocuda
import findfile
import numpy as np
import sklearn.metrics as metrics
import torch
from metric_visualizer import MetricVisualizer
from tqdm import tqdm

from benchmark_V2.benchmark_utils import seed_everything, RNADataset



seed_everything()
class RNAClassificationDataset(RNADataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None):
        super().__init__(data_file, tokenizer, max_seq_len, device)

    def encode(self, example):
        tokenized_seq = self.tokenize(example["seq"])
        if "attention_mask" in tokenized_seq:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "attention_mask": tokenized_seq["attention_mask"],
                "label": torch.tensor(example["label"]),
            }
        else:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "label": torch.tensor(example["label"]),
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
    mv = MetricVisualizer('run_promoter_detect')
    epochs = 10
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 1e-5
    # weight_decay = 0
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
        # # 'RNAFM',
        # 'splicebert-510nt',
        # 'codonbert',
        # 'cdsbert',
        # '3utrbert',
        # 'mlm_checkpoint',
        "mprna_small",
        # "esm2_rna_checkpoint",
        # 'agro-nucleotide-transformer-1b'
    ]:
        if model_name == 'agro-nucleotide-transformer-1b':
            batch_size = 4

        model_path = findfile.find_dir('../', model_name)

        for seed in [42, 43, 44]:
            seed_everything(seed)
            from benchmark_V2.benchmark_utils import get_sequence_classification_model

            model_path = findfile.find_dir('../', [model_name, 'pretrained_models'])
            model, tokenizer = get_sequence_classification_model(model_path, num_labels=2)

            # write a function to calculate trainable parameters
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)


            print(f'{model_name} has {count_parameters(model) / 1e6:.2f}M trainable parameters')

            base_dir = "./"
            # 1. 加载数据集
            train_set = RNAClassificationDataset(base_dir + "dataset/train.json", tokenizer, device=device)
            valid_set = RNAClassificationDataset(base_dir + "dataset/valid.json", tokenizer, device=device)
            test_set = RNAClassificationDataset(base_dir + "dataset/test.json", tokenizer, device=device)
            model.config.id2label = {0: 'Non-promoter', 1: 'Promoter'}

            def compute_metrics(p):
                true_predictions, true_labels = p

                classification_report = metrics.classification_report(true_labels, true_predictions, digits=4)
                print(classification_report)
                results = {
                    "accuracy": metrics.accuracy_score(true_labels, true_predictions),
                    "precision": metrics.precision_score(true_labels, true_predictions, average="macro"),
                    "recall": metrics.recall_score(true_labels, true_predictions, average="macro"),
                    "f1": metrics.f1_score(true_labels, true_predictions, average="macro"),
                    "sensitivity": metrics.recall_score(true_labels, true_predictions, average="binary", pos_label=1),
                    'specificity': metrics.recall_score(true_labels, true_predictions, average="binary", pos_label=0),
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
                    loss = loss_fn(outputs.view(-1, 2), batch['label'].view(-1).long())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss.append(loss.item())
                    train_it.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {np.average(train_loss):.4f}")

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
            # mv.log_metric(model_name, 'sensitivity', test_metrics['sensitivity'])
            # mv.log_metric(model_name, 'specificity', test_metrics['specificity'])

            mv.dump('result.mv')
            mv.raw_summary(round=4)
            mv.summary(round=4)
            # mv.to_txt('result.txt')
