# -*- coding: utf-8 -*-
# file: run_mutation_detection.py
# time: 00:57 21/01/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.



import autocuda
import findfile
import numpy as np
import torch
from metric_visualizer import MetricVisualizer
from sklearn import metrics
from tqdm import tqdm

from benchmark_V2.benchmark_utils import seed_everything, RNADataset


class RNAClassificationDataset(RNADataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None, **kwargs):
        super().__init__(data_file, tokenizer, max_seq_len, device, **kwargs)

    def encode(self, example):
        mut_tokenized_seq = self.tokenize(example["mut"], add_special_tokens=False)
        ori_tokens = list(example["seq"])
        mut_tokens = list(example["mut"])
        label = [0 if ori_tokens[i] == mut_tokens[i] else 1 for i in range(len(ori_tokens))]
        label = torch.tensor(label)
        if "attention_mask" in mut_tokenized_seq:
            data = {
                "input_ids": mut_tokenized_seq["input_ids"],
                "attention_mask": mut_tokenized_seq["attention_mask"],
                "label": label,
            }
        else:
            data = {
                "input_ids": mut_tokenized_seq["input_ids"],
                "label": label,
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
    mv = MetricVisualizer('run_nucleotide_frequency_prediction')
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
        # # 'RNAFM',
        # 'splicebert-510nt',
        # 'codonbert',
        # 'cdsbert',
        # '3utrbert',
        # 'mlm_checkpoint',
        "mprna_small",
        # # "esm2_rdna_checkpoint",
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
            model, tokenizer = get_token_classification_model(model_path, num_labels=2)

            # write a function to calculate trainable parameters
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)


            print(f'{model_name} has {count_parameters(model) / 1e6:.2f}M trainable parameters')

            base_dir = "./"
            # 1. 加载数据集
            train_set = RNAClassificationDataset(base_dir + "dataset/train.json", tokenizer, device=device)
            valid_set = RNAClassificationDataset(base_dir + "dataset/valid.json", tokenizer, device=device)
            test_set = RNAClassificationDataset(base_dir + "dataset/test.json", tokenizer, device=device)


            class FocalLoss(torch.nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, inputs, targets):
                    inputs = torch.nn.functional.softmax(inputs, dim=-1)
                    BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-BCE_loss)
                    F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                    return F_loss.mean()


            def compute_metrics(p):
                def flatten(l):
                    return [item for sublist in l for item in sublist]

                predictions, labels = p
                true_predictions = [
                    [p for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = [
                    [l for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = flatten(true_labels)
                true_predictions = flatten(true_predictions)

                try:
                    auc = metrics.roc_auc_score(true_labels, true_predictions)
                except Exception as e:
                    print(e)
                    auc = 0.5

                classification_report = metrics.classification_report(true_labels, true_predictions, digits=4)
                print(classification_report)
                results = {
                    "auc": auc,
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
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 100.]).to(device))
            # loss_fn = FocalLoss()
            model.to(device)
            global_acc = -np.inf
            for epoch in range(epochs):
                model.train()
                train_loss = []
                train_it = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Loss:")
                for batch in train_it:
                    outputs = model(batch)
                    outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    loss = loss_fn(outputs.view(-1, 2), batch['label'].view(-1))
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
                    val_truth.append(batch['label'].cpu().numpy())
                    val_preds.append(outputs.detach().argmax(-1).cpu().numpy())

                val_truth = np.concatenate(val_truth)
                val_preds = np.concatenate(val_preds)
                valid_metrics = compute_metrics((val_preds, val_truth))
                if valid_metrics['auc'] > global_acc:
                    torch.save(model.state_dict(), "model.pt")
                    global_acc = valid_metrics['auc']

            with torch.no_grad():
                # Testing
                model.load_state_dict(torch.load("model.pt"))
                model.to(device)
                preds = []
                truth = []
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
                it = tqdm(test_loader, desc="Testing")
                for batch in it:
                    outputs = model(batch)
                    outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    preds.append(outputs.detach().argmax(-1).cpu().numpy())
                    truth.append(batch["label"].cpu().numpy())

                preds = np.concatenate(preds)
                truth = np.concatenate(truth)
                test_metrics = compute_metrics((preds, truth))

                mv.log_metric(model_name, 'auc', test_metrics['auc'])
                # mv.log_metric(model_name, 'accuracy', test_metrics['accuracy'])
                # mv.log_metric(model_name, 'precision', test_metrics['precision'])
                # mv.log_metric(model_name, 'recall', test_metrics['recall'])
                # mv.log_metric(model_name, 'f1', test_metrics['f1'])

                mv.dump('result.mv')
                mv.raw_summary(round=4)
                mv.summary(round=4)
                mv.to_txt('result.txt')

