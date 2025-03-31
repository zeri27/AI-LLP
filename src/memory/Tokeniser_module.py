from transformers import AutoTokenizer

def get_tokenised_sections(texts, tokenizer):
    tokenised_sections = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return tokenised_sections

def get_tokeniser(model_name):
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    return tokeniser