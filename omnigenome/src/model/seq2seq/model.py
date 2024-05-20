








from ...abc.abstract_model import OmniGenomeModel


class OmniGenomeEncoderModelForSeq2Seq(OmniGenomeModel):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
