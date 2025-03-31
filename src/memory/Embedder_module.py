from transformers import AutoModel
import torch


def get_embeddings(tokenised_sections, model):
    with torch.no_grad():
        model_output = model(**tokenised_sections)
        # Use the embedding of the [CLS] token (first token) for each input
        embeddings = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embeddings

def get_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model