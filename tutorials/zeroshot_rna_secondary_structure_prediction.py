







import torch

from transformers import OmniGenomeForTokenClassification, AutoTokenizer

if __name__ == "__main__":

    sequence = "GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC"

    
    
    ssp_model = OmniGenomeForTokenClassification.from_pretrained("anonymous8/OmniGenome-186M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssp_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("anonymous8/OmniGenome-186M")
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = ssp_model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]
    structure = [ssp_model.config.id2label[prediction.item()] for prediction in predictions[0]]
    print("".join(structure))
    

    
    
    structure = ssp_model.fold(sequence)
    print(structure)
    

    

    
    
    

