








from transformers import Trainer
from transformers import TrainingArguments

from ... import __name__ as omnigenome_name
from ... import __version__ as omnigenome_version


class HFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }


class HFTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super(HFTrainingArguments, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }
