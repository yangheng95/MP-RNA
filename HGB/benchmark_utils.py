# -*- coding: utf-8 -*-
# file: transformer_models.py
# time: 14:55 20/01/2024
import json
import os
import random

import numpy as np
import pandas as pd
import tqdm

from transformers import AutoModel, AutoTokenizer


import torch


class AutoModelForTokenClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'hidden_size' not in config.__dict__:
            config.hidden_size = 256

        self.config = config
        self.num_labels = config.num_labels
        self.model = auto_model
        if not hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = 0
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, **kwargs):
        outputs = self.model(
            input_ids,
            **kwargs,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class AutoModelForSequenceClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'hidden_size' not in config.__dict__:
            config.hidden_size = 256

        self.config = config
        self.num_labels = config.num_labels
        self.pooler = BertPooler(config)
        self.model = auto_model
        if not hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = 0
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# -*- coding: utf-8 -*-
# file: transformer_models.py
# time: 14:55 20/01/2024
from transformers.models.bert.modeling_bert import BertPooler

import torch


class EncoderModelForTokenClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.model = auto_model
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.activation = torch.nn.Tanh()
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.activation(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class EncoderModelForSequenceClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.pooler = BertPooler(config)
        self.model = auto_model
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DecoderModelForTokenClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.model = auto_model
        self.score = torch.nn.Linear(config.d_model, self.num_labels, bias=False)
        self.activation = torch.nn.Tanh()

    def forward(self, inputs, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        input_ids = inputs['input_ids']
        transformer_outputs = self.model(
            input_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        logits = self.activation(logits)
        return logits


class DecoderModelForSequenceClassification(torch.nn.Module):
    def __init__(self, config, auto_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.model = auto_model
        self.score = torch.nn.Linear(config.d_model, self.num_labels, bias=False)
        self.activation = torch.nn.Tanh()

    def forward(self, inputs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        input_ids = inputs['input_ids']
        transformer_outputs = self.model(
            input_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        logits = self.activation(logits)

        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1

        pooled_logits = logits[torch.arange(input_ids.size(0), device=logits.device), sequence_lengths]
        return pooled_logits


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_seq_len=None, device=None, **kwargs):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len is not None else -1

        self.device = device

        self.data = []
        examples = []
        max_examples = kwargs.get("max_examples", None)
        if data_file[-3:] == 'csv':
            df = pd.read_csv(data_file)
            for i in range(len(df)):
                examples.append(df.iloc[i].to_dict())

        elif data_file[-4:] == 'json':
            with open(data_file, "r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = json.loads(lines[i])
            for line in lines:
                examples.append(line)

        elif data_file[-7:] == 'parquet':
            df = pd.read_parquet(data_file)
            for i in range(len(df)):
                examples.append(df.iloc[i].to_dict())

        else:
            raise Exception("Unknown file format.")

        random.shuffle(examples)
        if max_examples is not None:
            examples = examples[:max_examples]

        self._labels = set()
        for example in tqdm.tqdm(examples, desc=f'Tokenizing {data_file}'):
            try:
                self.data.append(self.encode(example))
            except Exception as e:
                print(e)

        self._pad_and_truncate()
        self._post_processing()

    def tokenize(self, seq, **kwargs):
        assert isinstance(seq, str), "input DNA/RNA seq must be string type"
        try:
            tokenized_inputs = self.tokenizer(
                ' '.join(list(seq.replace('U', 'T'))),
                # seq,
                padding='do_not_pad',
                truncation=True,
                return_tensors="pt",
                **kwargs
            )
            if len(tokenized_inputs['input_ids'][0]) <= 3:
                seq = seq.replace('T', 'U')
                seq = [seq[i:i + 3] for i in range(len(seq))]
                tokenized_inputs = self.tokenizer(
                    ' '.join(list(seq)),
                    padding='do_not_pad',
                    truncation=True,
                    return_tensors="pt",
                    **kwargs
                )
        except Exception as e:
            seq = seq.replace('T', 'U')
            seq = [seq[i:i+3] for i in range(len(seq))]
            tokenized_inputs = self.tokenizer.encode(
                ' '.join(seq),
                **kwargs
            )
            tokenized_inputs = {
                'input_ids': torch.tensor([tokenized_inputs.ids]),
                'attention_mask':  torch.tensor([tokenized_inputs.attention_mask])
            }
        self.max_seq_len = max(self.max_seq_len, len(tokenized_inputs['input_ids'][0]))
        if 'attention_mask' in tokenized_inputs:
            return {'input_ids': tokenized_inputs['input_ids'][0],
                    'attention_mask': tokenized_inputs['attention_mask'][0]}
        else:
            return {'input_ids': tokenized_inputs['input_ids'][0]}

    def sn_tokenize(self, seq):
        assert isinstance(seq, str), "input DNA/RNA seq must be string type"
        input_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.bos_token if self.tokenizer.bos_token else self.tokenizer.cls_token]
            + list(seq)
            + [self.tokenizer.eos_token if self.tokenizer.eos_token else self.tokenizer.pad_token])
        attention_mask = [1] * len(input_ids)

        self.max_seq_len = max(self.max_seq_len, len(input_ids[0]))

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def _pad_and_truncate(self):
        _input_ids_dtype = torch.int
        _attention_mask_dtype = torch.int
        if 'label' in self.data[0]:
            _label_dtype = self.data[0]['label'].dtype

            if len(self.data[0]['label'].shape) != 0:
                self.max_seq_len = max(self.max_seq_len, len(self.data[0]['label']))

        for i in range(len(self.data)):
            # pad
            if 'attention_mask' in self.data[i]:
                self.data[i]['input_ids'] = torch.cat(
                    [self.data[i]['input_ids'], torch.tensor(
                        [self.tokenizer.pad_token_id] * (self.max_seq_len - len(self.data[i]['input_ids'])))])
                self.data[i]['attention_mask'] = torch.cat(
                    [self.data[i]['attention_mask'],
                     torch.tensor([0] * (self.max_seq_len - len(self.data[i]['attention_mask'])))])

                # convert dtype
                self.data[i]['input_ids'].to(self.device)
                self.data[i]['attention_mask'] = self.data[i]['attention_mask'].to(
                    self.device)
            else:
                self.data[i]['input_ids'] = torch.cat(
                    [self.data[i]['input_ids'], torch.tensor(
                        [self.tokenizer.pad_token_id] * (self.max_seq_len - len(self.data[i]['input_ids'])))]).to(self.device)

            if 'label' in self.data[i]:
                if len(self.data[i]['label'].shape) != 0:
                    self.data[i]['label'] = torch.cat(
                        [self.data[i]['label'], torch.tensor([-100] * (self.max_seq_len - len(self.data[i]['label'])))])
                    self.data[i]['label'] = self.data[i]['label'].to(_label_dtype).to(self.device)

            # truncate
            self.data[i]['input_ids'] = self.data[i]['input_ids'][:self.max_seq_len].to(self.device)
            self.data[i]['input_ids'] = self.data[i]['input_ids'].to(_input_ids_dtype)

            if 'attention_mask' in self.data[i]:
                self.data[i]['attention_mask'] = self.data[i]['attention_mask'][:self.max_seq_len].to(self.device)
                self.data[i]['attention_mask'] = self.data[i]['attention_mask'].to(_attention_mask_dtype)

            if 'label' in self.data[i]:
                if len(self.data[i]['label'].shape) != 0:
                    self.data[i]['label'] = self.data[i]['label'][:self.max_seq_len].to(self.device)
                    self.data[i]['label'] = self.data[i]['label'].to(_label_dtype)

    def _post_processing(self):
        pass

    def encode(self, example):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def init_base_model(model_name, task_type, **kwargs):
    base_model = AutoModel.from_pretrained(model_name, **kwargs)
    if task_type in "token_classification":
        try:
            model = DecoderModelForTokenClassification(base_model.config, auto_model=base_model)
        except:
            model = EncoderModelForTokenClassification(base_model.config, auto_model=base_model)
    elif task_type in "sequence_classification":
        try:
            model = DecoderModelForSequenceClassification(base_model.config, auto_model=base_model)
        except:
            model = EncoderModelForSequenceClassification(base_model.config, auto_model=base_model)
    else:
        raise Exception("Unknown model type.")

    return model

def get_token_classification_model(model_name, num_labels):
    base_model = init_base_model(model_name, "token_classification", num_labels=num_labels, trust_remote_code=True)
    if 'codon' in model_name.lower():
        from benchmark_V2.codonbert_adapter import get_tokenizer
        tokenizer = get_tokenizer()
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.pad_token = '[PAD]'
        tokenizer.cls_token_id = 2
        tokenizer.sep_token_id = 3
        tokenizer.eos_token_id = 3
        tokenizer.pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    return base_model, tokenizer

def get_sequence_classification_model(model_name, num_labels):
    base_model = init_base_model(model_name, "sequence_classification", num_labels=num_labels, trust_remote_code=True)
    if 'codon' in model_name.lower():
        from benchmark_V2.codonbert_adapter import get_tokenizer
        tokenizer = get_tokenizer()
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.pad_token = '[PAD]'
        tokenizer.cls_token_id = 2
        tokenizer.sep_token_id = 3
        tokenizer.eos_token_id = 3
        tokenizer.pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    return base_model, tokenizer

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
