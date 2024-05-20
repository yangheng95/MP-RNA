







import json
import os
import shutil
import warnings

import findfile
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, BatchEncoding
from transformers.models.bert.modeling_bert import BertPooler

from ..misc.utils import RNA2StructureCache
from ..misc.utils import fprint, env_meta_info


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def last_hidden_state_forward(model, inputs, ss=None, tokenizer=None):
    
    assert ss in [
        None,
        "viennarna",
        "model",
    ], f'ss should be one of [None, "viennarna", "model"], got {ss}'

    if isinstance(inputs, tuple):
        input_ids = inputs[0]
        attention_mask = inputs[1] if len(inputs) > 1 else None
    elif isinstance(inputs, BatchEncoding) or isinstance(inputs, dict):
        input_ids = inputs["input_ids"]
        attention_mask = (
            inputs["attention_mask"] if "attention_mask" in inputs else None
        )
    else:
        raise ValueError(
            f"The inputs should be a tuple, BatchEncoding or a dictionary-like object, got {type(inputs)}."
        )

    try:
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    except Exception as e:
        if "attention_mask" not in str(e):
            raise e
        
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )

    if not hasattr(outputs, "last_hidden_state"):
        warnings.warn(
            f"last_hidden_state not found in the outputs from the {model.__class__.__name__} model."
        )

    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states"):
        last_hidden_state = outputs.hidden_states[-1]
    elif (
            isinstance(outputs, list)
            or isinstance(outputs, tuple)
            or isinstance(outputs, torch.Tensor)
    ):
        
        last_hidden_state = outputs[-1] if len(outputs[-1].shape) == 3 else outputs[0]
    else:
        raise ValueError(
            f"Cannot find the last hidden state in the outputs from the {model.__class__.__name__} \
            model, please check the model architecture."
        )

    if ss == "viennarna":
        if not hasattr(model, "rna2structure"):
            model.rna2structure = RNA2StructureCache()

        if hasattr(tokenizer, "base_tokenizer"):
            tokenizer = tokenizer.base_tokenizer
        sequences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        structures = model.rna2structure.fold(
            [seq.replace(" ", "") for seq in sequences]
        )
        tokenized_struct = tokenizer(
            structures,
            padding="max_length",
            max_length=input_ids.shape[1],
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokenized_struct.to(input_ids.device)
        ss_last_hidden_state = model(
            **tokenized_struct,
            output_hidden_states=True,
        )["last_hidden_state"]
    elif ss == "model":
        raise NotImplementedError(
            "The model-based secondary structure information is not implemented yet."
        )
        
        
        
        
        
        
    else:
        return last_hidden_state

    return last_hidden_state, ss_last_hidden_state


class OmniGenomePooling(torch.nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.pooler = BertPooler(self.config) if not self._is_causal_lm() else None


    def forward(self, inputs, last_hidden_state):
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
        elif isinstance(inputs, BatchEncoding) or isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = (
                inputs["attention_mask"] if "attention_mask" in inputs else None
            )
        else:
            raise ValueError(
                f"The inputs should be a tuple, BatchEncoding or a dictionary-like object, got {type(inputs)}."
            )

        if not self.pooler:
            pad_token_id = getattr(self.config, "pad_token_id", -100)
            sequence_lengths = input_ids.ne(pad_token_id).sum(dim=1) - 1
            last_hidden_state = last_hidden_state[
                torch.arange(input_ids.size(0), device=last_hidden_state.device),
                sequence_lengths,
            ]
        else:
            last_hidden_state = self.pooler(last_hidden_state)

        return last_hidden_state

    def _is_causal_lm(self):
        if (
                hasattr(self.config, "architectures")
                and "CausalLM" in str(self.config.architectures)
        ) or (
                hasattr(self.config, "auto_map") and "CausalLM" in str(self.config.auto_map)
        ):
            return True
        else:
            return False


class OmniGenomeModel(torch.nn.Module):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        self.loss_fn = None

        label2id = kwargs.pop("label2id", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        num_labels = kwargs.pop("num_labels", None)

        if label2id is not None and num_labels is None:
            num_labels = len(label2id)

        
        super().__init__(*args, **kwargs)

        if isinstance(config_or_model_model, str):
            config = AutoConfig.from_pretrained(
                config_or_model_model,
                num_labels=num_labels,
                label2id=label2id,
                trust_remote_code=trust_remote_code,
            )
            self.model = AutoModel.from_pretrained(
                config_or_model_model,
                config=config,
                trust_remote_code=trust_remote_code,
            )
            self.model.config = config
        elif isinstance(config_or_model_model, torch.nn.Module):
            self.model = config_or_model_model
        elif isinstance(config_or_model_model, AutoConfig):
            config = config_or_model_model
            self.model = AutoModel.from_config(config)
            self.model.config = config
        else:
            raise ValueError(
                "The config_or_model_model should be either a string, a torch.nn.Module or a AutoConfig object."
            )

        
        self.config = self.model.config
        if isinstance(label2id, dict):
            self.config.label2id = label2id
            self.config.id2label = {v: k for k, v in label2id.items()}

        
        self.metadata = env_meta_info()
        self.metadata["model_cls"] = self.__class__.__name__

        
        if hasattr(self.config, "n_embd"):
            self.config.hidden_size = self.config.n_embd
        elif hasattr(self.config, "d_model"):
            self.config.hidden_size = self.config.d_model
        elif hasattr(self.config, "hidden_size"):
            self.config.hidden_size = self.config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "base_tokenizer"):
            self.pad_token_id = self.tokenizer.base_tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.activation = torch.nn.Tanh()

    def loss_function(self, logits, labels):
        raise NotImplementedError(
            "The loss_function() function should be implemented for your model."
        )

    def set_loss_fn(self, loss_function):
        self.loss_fn = loss_function

    def predict(self, sequence_or_inputs, **kwargs):
        
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def inference(self, sequence_or_inputs, **kwargs):
        
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def forward(self, inputs):
        last_hidden_state = last_hidden_state_forward(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        outputs = {"last_hidden_state": last_hidden_state}
        return outputs

    def __call__(self, inputs, labels=None, *args, **kwargs):
        if isinstance(inputs, dict):
            labels = inputs.get("labels", None)
            label = inputs.get("label", None)
            labels = labels if labels is not None else label
            if labels is None:
                warnings.warn(
                    "No labels are provided in the inputs, the model will not calculate the loss."
                )
        elif isinstance(inputs, tuple):
            labels = inputs[1]
            inputs = inputs[0]
        elif labels is not None:
            labels = labels

        outputs = self.forward(inputs)

        if labels is not None:
            outputs["loss"] = self._calculate_loss(outputs, labels)
        else:
            outputs["loss"] = None
        return outputs

    def _calculate_loss(self, outputs, labels):
        loss = outputs.get("loss", None)
        if loss is not None:
            return outputs

        logits = outputs["logits"]
        if logits is not None or labels is not None:
            loss = self.loss_function(logits, labels)
            return loss
        else:
            raise RuntimeError(
                "The output of the forward() function should be a dictionary-like objective"
                " and have either 'loss', or 'logits' and 'labels' attribute."
            )

    def save(self, path, overwrite=False, dtype=torch.float16, **kwargs):
        self.eval()
        import dill

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"The path {path} already exists, please set overwrite=True to overwrite it."
            )

        if not os.path.exists(path):
            os.makedirs(path)

        for file in findfile.find_files(
                self.config.name_or_path,
                and_key=[],
                exclude_key=["pytorch_model", "model", "safetensors"],
        ):
            shutil.copyfile(file, f"{path}/{os.path.basename(file)}")

        _device = self.model.device
        _dtype = self.model.dtype
        self.model.to(dtype).to("cpu")
        with open(f"{path}/tokenizer.pkl", "wb") as f:
            dill.dump(self.tokenizer, f)
        with open(f"{path}/metadata.json", "w", encoding="utf8") as f:
            json.dump(self.metadata, f)
        self.model.save_pretrained(
            f"{path}", safe_serialization=False
        )  
        with open(f"{path}/pytorch_model.bin", "wb") as f:
            torch.save(self.state_dict(), f)

        self.model.to(_dtype).to(_device)
        fprint(f"The model is saved to {path}.")

    def load(self, path, **kwargs):
        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        if metadata["model_cls"] != self.__class__.__name__:  
            raise ValueError(
                f"The model class in the loaded model is {metadata['model_cls']}, "
                f"but the current model class is {self.__class__.__name__}."
            )
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        for key, value in config.__dict__.items():
            if key not in self.config.__dict__ or self.config.__dict__[key] != value:
                fprint(
                    f"Warning: The value of the key {key} in the loaded model is {value}, "
                    f"but the current value is {self.config.__dict__.get(key, None)}."
                )

        with open(f"{path}/pytorch_model.bin", "rb") as f:
            self.load_state_dict(
                torch.load(f, map_location=kwargs.get("device", "cpu")), strict=True
            )
        return self

    def _forward_from_raw_input(self, sequence_or_inputs, **kwargs):
        if not isinstance(sequence_or_inputs, BatchEncoding) and not isinstance(
                sequence_or_inputs, dict
        ):
            inputs = self.tokenizer(
                sequence_or_inputs,
                padding=kwargs.pop("padding", True),
                max_length=kwargs.pop("max_length", 1024),
                truncation=kwargs.pop('truncation', True),
                return_tensors=kwargs.pop('return_tensors', 'pt'),
                **kwargs
            )
        else:
            inputs = sequence_or_inputs
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            raw_outputs = self(inputs)
            raw_outputs["inputs"] = inputs
        return raw_outputs

    @staticmethod
    def from_pretrained(model_name_or_path, tokenizer, *args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        base_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
        return OmniGenomeModel(config, base_model, tokenizer, *args, **kwargs)

    def model_info(self):
        info = f"Model Name: {self.__class__.__name__}\n"
        info += f"Model Metadata: {self.metadata}\n"
        info += f"Base Model Name: {self.config.name_or_path}\n"
        info += f"Model Type: {self.config.model_type}\n"
        info += f"Model Architecture: {self.config.architectures}\n"
        info += f"Model Parameters: {count_parameters(self.model) / 1e6} M\n"
        info += f"Model Config: {self.config}\n"
        fprint(info)
        return info
