







import warnings

from ..abc.abstract_tokenizer import OmniGenomeTokenizer


class OmniKmersTokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, k=3, overlap=0, max_length=512, **kwargs):
        super(OmniKmersTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.k = k
        self.overlap = overlap
        self.max_length = max_length
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        if self.u2t:
            sequence = "".join([seq.replace("U", "T").upper() for seq in sequence])
        if self.t2u:
            sequence = "".join([seq.replace("T", "U").upper() for seq in sequence])

        sequence_tokens = self.tokenize(sequence)[
            : kwargs.get("max_length", self.max_length) - 2
        ]
        tokenized_inputs = {
            "input_ids": [],
            "attention_mask": [],
        }
        bos_id = (
            self.base_tokenizer.bos_token_id
            if self.base_tokenizer.bos_token_id is not None
            else self.base_tokenizer.cls_token_id
        )
        eos_id = (
            self.base_tokenizer.eos_token_id
            if self.base_tokenizer.eos_token_id is not None
            else self.base_tokenizer.sep_token_id
        )

        for tokens in sequence_tokens:
            tokenized_inputs["input_ids"].append(
                [bos_id] + self.base_tokenizer.convert_tokens_to_ids(tokens) + [eos_id]
            )
            tokenized_inputs["attention_mask"].append(
                [1] * len(tokenized_inputs["input_ids"][-1])
            )

        for i, ids in enumerate(tokenized_inputs["input_ids"]):
            if ids.count(self.base_tokenizer.unk_token_id) / len(ids) > 0.1:
                warnings.warn(
                    f"Unknown tokens are more than 10% in the {i}th sequence, please check the tokenization process."
                )
        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        self = OmniKmersTokenizer(
            AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        if isinstance(sequence, str):
            sequences = [sequence]
        else:
            sequences = sequence

        sequence_tokens = []
        for i in range(len(sequences)):
            tokens = []
            for j in range(0, len(sequences[i]), self.k - self.overlap):
                tokens.append(sequences[i][j : j + self.k])

            sequence_tokens.append(tokens)

        return sequence_tokens

    def encode(self, input_ids, **kwargs):
        return self.base_tokenizer.encode(input_ids, **kwargs)

    def decode(self, input_ids, **kwargs):
        return self.base_tokenizer.decode(input_ids, **kwargs)

    def encode_plus(self, sequence, **kwargs):
        raise NotImplementedError("The encode_plus() function is not implemented yet.")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    
    
    
    
    
    
    
    
    

    RNA = "ACGUAGGUAUCGUAGA"
    
    base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    tokenizer = OmniKmersTokenizer(base_tokenizer, k=4, overlap=2, max_length=512)
    tokens = tokenizer.tokenize(RNA)
    print(tokens)
    tokenized_inputs = tokenizer(RNA)
    print(tokenized_inputs)
