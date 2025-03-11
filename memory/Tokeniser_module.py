from transformers import AutoTokenizer

# Function to get the tokenised version of a text using a tokeniser
def get_tokenised_sections(texts, tokenizer):
    tokenised_sections = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return tokenised_sections

# Function to get the tokeniser
def get_tokeniser(model_name):
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    return tokeniser