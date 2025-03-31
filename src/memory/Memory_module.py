import faiss
from src.memory.Embedder_module import get_embeddings
from src.memory.Tokeniser_module import get_tokenised_sections
import numpy as np

save_chat_to_memory_method = "whole_chat" #options are "whole_chat", "individual_messages", "pairs"

def create_index(d):
    index = faiss.IndexFlatL2(d)
    return index

def add_to_index(data, index, stored_memory, tokeniser, model):
    data = "<START OF SAVED DATA BY THE AGENT> " + data + " <END OF SAVED DATA BY THE AGENT>"
    tokenised_data = get_tokenised_sections([data], tokeniser)
    embeddings = get_embeddings(tokenised_data, model)
    index.add(embeddings)
    stored_memory.append(data)


def search(query, k, index, tokeniser, model):
    tokenised_query = get_tokenised_sections([query], tokeniser)
    query_embedding = get_embeddings(tokenised_query, model)
    D, I = index.search(query_embedding, k)
    return D, I

def save_index(index, path):
    path = path + ".index"
    faiss.write_index(index, path)


def load_index(path):
    path = path + ".index"
    return faiss.read_index(path)

def load_reference(path):
    #load numpy array
    path = path + ".npy"
    return np.load(path, allow_pickle=True).tolist()

def save_reference(reference, path):
    #save list as numpy array
    np.save(path, np.array(reference))



def add_to_index_wholechat(chat, index, stored_memory, tokeniser, model):
    combined_chat = ""
    for i, message in enumerate(chat):
        if i % 2 == 0: #then it is from the agent
            combined_chat += "<START OF AGENT MESSAGE>" + message + "<END OF AGENT MESSAGE>"
        else:
            combined_chat += "<START OF USER MESSAGE>" + message + "<END OF USER MESSAGE>"
        
    tokenised_chat = get_tokenised_sections([combined_chat], tokeniser)
    embeddings = get_embeddings(tokenised_chat, model)
    index.add(embeddings)
    stored_memory.append(combined_chat)

#putting each individual message as a memory entry
def add_to_index_individual_messages(chat, index, stored_memory, tokeniser, model):
    for i, message in enumerate(chat):
        if i % 2 == 0:
            tokenised_message = get_tokenised_sections(["<START OF AGENT MESSAGE>" + message + "<END OF AGENT MESSAGE>"], tokeniser)
        else:
            tokenised_message = get_tokenised_sections(["<START OF USER MESSAGE>" + message + "<END OF USER MESSAGE>"], tokeniser)
        embeddings = get_embeddings(tokenised_message, model)
        index.add(embeddings)
        stored_memory.append(tokenised_message)

#putting each pair of user message and agent message as a memory entry
def add_to_index_pairs(chat, index, stored_memory, tokeniser, model):
    for i in range(1, len(chat), 2):
        message_pair = ["<START OF USER MESSAGE>" + chat[i-1] + "<END OF USER MESSAGE>", "<START OF AGENT MESSAGE>" + chat[i] + "<END OF AGENT MESSAGE>"]
        tokenised_pair = get_tokenised_sections(message_pair, tokeniser)
        embeddings = get_embeddings(tokenised_pair, model)
        index.add(embeddings)
        stored_memory.append(message_pair)


#used after the session ends to save the chat to memory for future sessions. This is not done by the llm but automatically after each session
def save_chat_to_memory(chat, index, stored_memory, tokeniser, model):
    if save_chat_to_memory_method == "whole_chat":
        add_to_index_wholechat(chat, index, stored_memory, tokeniser, model)
    elif save_chat_to_memory_method == "individual_messages":
        add_to_index_individual_messages(chat, index, stored_memory, tokeniser, model)
    elif save_chat_to_memory_method == "pairs":
        add_to_index_pairs(chat, index, stored_memory, tokeniser, model)
    else:
        print("Invalid save_chat_to_memory_method: " + save_chat_to_memory_method)