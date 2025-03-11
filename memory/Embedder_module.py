from transformers import AutoModel
import torch

# Function to get embeddings from a tokenised input using an embedding model
def get_embeddings(tokenised_sections, model):
    with torch.no_grad():
        model_output = model(**tokenised_sections)
        embeddings = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embeddings

# Function to get the embedding model
def get_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model