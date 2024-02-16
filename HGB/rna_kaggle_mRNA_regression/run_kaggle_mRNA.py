# -*- coding: utf-8 -*-
# file: kaggle_mRNA.py
# time: 19:34 19/01/2024
import os
import random

import autocuda
import findfile
import numpy
import numpy as np
import pandas as pd
from metric_visualizer import MetricVisualizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark_V2.benchmark_utils import seed_everything, RNADataset

seed_everything()


# Define a custom dataset for regression
class RegressionDataset(RNADataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None):
        self.inp_seq_cols = ['sequence', 'structure', 'predicted_loop_type']
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        super().__init__(data_file, tokenizer, max_seq_len, device)

    def encode(self, example):
        tokenized_seq = self.tokenize(example["sequence"], add_special_tokens=False)
        labels = [example[target_col] for target_col in self.target_cols]
        labels = np.concatenate([
            np.array(labels),
            np.array([[-100] * (self.max_seq_len - len(labels[0])),
                      [-100] * (self.max_seq_len - len(labels[0])),
                      [-100] * (self.max_seq_len - len(labels[0]))]
                     )], axis=1).T
        if 'attention_mask' in tokenized_seq:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "attention_mask": tokenized_seq["attention_mask"],
                "label": torch.tensor(
                    labels,
                    dtype=torch.float32),
            }
        else:
            data = {
                "input_ids": tokenized_seq["input_ids"],
                "label": torch.tensor(
                    labels,
                    dtype=torch.float32),
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


TRAIN = "dataset/train.json"
TEST = "dataset/test.json"
SS = "dataset/sample_submission.csv"
sample_sub = pd.read_csv(SS)

if __name__ == "__main__":
    mv = MetricVisualizer('run_kaggle_mRNA')
    epochs = 10
    learning_rate = 2e-5
    batch_size = 16
    weight_decay = 1e-5
    device = autocuda.auto_cuda()
    seeds = [random.randint(0, 100) for _ in range(5)]

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

        for seed in seeds:
            seed_everything(seed)

            from benchmark_V2.benchmark_utils import get_token_classification_model, RNADataset

            model_path = findfile.find_dir('../', [model_name, 'pretrained_models'])
            model, tokenizer = get_token_classification_model(model_path, num_labels=3)


            # write a function to calculate trainable parameters
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)


            print(f'{model_name} has {count_parameters(model) / 1e6:.2f}M trainable parameters')


            train_set = RegressionDataset(TRAIN, tokenizer, device=device)
            test_set = RegressionDataset(TEST, tokenizer,  device=device)

            # Training
            train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42, shuffle=True)
            train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
            val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
            test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

            print(f"Using {device}")
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model.to(device)


            def compute_metrics(predictions, labels):
                mask = labels != -100

                # 使用掩码过滤logits和targets
                filtered_logits = predictions[mask]
                filtered_targets = labels[mask]

                # 计算每一列（任务）的均方误差
                mse_columns = []
                for i in range(labels.shape[2]):
                    mse = np.mean(
                        (filtered_targets.reshape(labels.shape[0], -1, labels.shape[2])[:, :, i]
                         - filtered_logits.reshape(labels.shape[0], -1, labels.shape[2])[:, :, i]) ** 2)
                    mse_columns.append(mse ** 0.5)

                # 计算所有列的均方根误差的平均值
                mcrmse = np.mean(mse_columns, axis=0)
                metrics = {
                    'mc_rmse': mcrmse
                }
                print(metrics)
                return metrics


            def loss_fn(predictions, labels):
                padding_value = -100.

                mask = labels != padding_value

                # 使用掩码过滤logits和targets
                filtered_logits = predictions[mask]
                filtered_targets = labels[mask]

                # 对过滤后的值计算MSE损失
                loss = torch.nn.functional.mse_loss(filtered_logits, filtered_targets, reduction='mean')
                return loss ** 0.5


            global_mc_rmse = np.inf
            for epoch in range(epochs):
                model.train()
                train_loss = []
                train_it = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Loss:")
                for batch in train_it:
                    outputs = model(batch)
                    loss = loss_fn(outputs.view(-1), batch['label'].view(-1))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss.append(loss.item())
                    train_it.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {np.average(train_loss):.4f}")

                model.eval()
                val_truth = []
                val_preds = []
                for batch in val_loader:
                    outputs = model(batch)
                    val_truth.append(batch['label'].detach().cpu().numpy())
                    val_preds.append(outputs.detach().cpu().numpy())

                metrics = compute_metrics(np.concatenate(val_preds), np.concatenate(val_truth))
                if metrics['mc_rmse'] < global_mc_rmse:
                    torch.save(model.state_dict(), "model.pt")
                    global_mc_rmse = metrics['mc_rmse']

            # Testing
            model.load_state_dict(torch.load("model.pt"))
            model.eval()
            preds = []
            truth = []
            it = tqdm(test_loader, desc="Testing")
            for batch in it:
                outputs = model(batch)
                truth.append(batch["label"].detach().cpu().numpy())
                preds.append(outputs.detach().cpu().numpy())

            truth = np.concatenate(truth)
            preds = np.concatenate(preds)

            metrics = compute_metrics(preds, truth)
            print(f"TEST MCRMSE: {metrics['mc_rmse']:.4f}")
            mv.log_metric(model_name, 'MCRMSE', metrics['mc_rmse'])

            mv.dump('result.mv')
            mv.summary(round=4)
            # mv.to_txt('result.txt')
