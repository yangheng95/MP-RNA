







import json
import os

import autocuda
import torch
from transformers import AutoConfig, AutoModel

from omnigenome.utility.hub_utils import query_models_info, download_model
from ...src.misc.utils import env_meta_info, fprint


class ModelHub:
    def __init__(self, *args, **kwargs):
        super(ModelHub, self).__init__(*args, **kwargs)

        self.metadata = env_meta_info()

    @staticmethod
    def load_model_and_tokenizer(model_name_or_path, local_only=False, **kwargs):
        model = ModelHub.load(model_name_or_path, local_only=local_only, **kwargs)
        fprint(f"The model and tokenizer has been loaded from {model_name_or_path}.")
        return model, model.tokenizer

    @staticmethod
    def load(model_name_or_path, local_only=False, device=None, **kwargs):
        if isinstance(model_name_or_path, str) and os.path.exists(model_name_or_path):
            path = model_name_or_path
        elif isinstance(model_name_or_path, str) and not os.path.exists(
            model_name_or_path
        ):
            path = download_model(model_name_or_path, local_only=local_only, **kwargs)
        else:
            raise ValueError("model_name_or_path must be a string.")
        import dill
        import importlib

        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        config.metadata = metadata
        base_model = AutoModel.from_config(config, trust_remote_code=True, **kwargs)
        model_lib = importlib.import_module(metadata["library_name"].lower()).model
        model_cls = getattr(model_lib, metadata["model_cls"])

        with open(f"{path}/tokenizer.pkl", "rb") as f:
            tokenizer = dill.load(f)

        model = model_cls(base_model, tokenizer, label2id=config.label2id, **kwargs)
        with open(f"{path}/pytorch_model.bin", "rb") as f:
            model.load_state_dict(
                torch.load(f, map_location=kwargs.get("device", "cpu")), strict=True
            )
        if device is None:
            model.to(autocuda.auto_cuda())
        else:
            model.to(device)
        return model

    def available_models(
        self, model_name_or_path=None, local_only=False, repo="", **kwargs
    ):
        models_info = query_models_info(
            model_name_or_path, local_only=local_only, repo=repo, **kwargs
        )
        return models_info

    def push(self, model, **kwargs):
        raise NotImplementedError("This method has not implemented yet.")
