import faiss
from memory.Embedder_module import get_embeddings
from memory.Tokeniser_module import get_tokenised_sections
import numpy as np


save_chat_to_memory_method = "individual_messages" #options are "whole_chat", "individual_messages", "pairs"

#function to create the index which is the vector database that will store the embeddings
def create_index(d):
    index = faiss.IndexFlatL2(d)
    return index

#function to add data to the index, this is invoked by the nlp module.
#  other functions like add_to_index_wholechat, add_to_index_individual_messages,
#  add_to_index_pairs are the ones that are invoked automatically after each session and not invoked by the model
def add_to_index(data, index, stored_memory, tokeniser, model):
    data = "<START OF SAVED DATA BY THE AGENT> " + data + " <END OF SAVED DATA BY THE AGENT>"
    tokenised_data = get_tokenised_sections([data], tokeniser)
    embeddings = get_embeddings(tokenised_data, model)
    index.add(embeddings)
    stored_memory.append(data)

#function to search the index for the k most similar embeddings to the query
def search(query, k, index, tokeniser, model):
    tokenised_query = get_tokenised_sections([query], tokeniser)
    query_embedding = get_embeddings(tokenised_query, model)
    D, I = index.search(query_embedding, k)
    return D, I

#saves the vector database to the drive so it can be loaded into memory again later
def save_index(index, path):
    path = path + ".index"
    faiss.write_index(index, path)

#loads the index from the drive
def load_index(path):
    path = path + ".index"
    return faiss.read_index(path)

# loads the long term memory data from the drive.
def load_reference(path):
    #load numpy array
    path = path + ".npy"
    return np.load(path, allow_pickle=True).tolist()

#saves the long term memory data to the drive so it can be loaded into memory again later
def save_reference(reference, path):
    #save list as numpy array
    np.save(path, np.array(reference))


# The following functions are used to save the chat to memory after each session. The chat can be saved in different ways
#  depending on the value of save_chat_to_memory_method. The options are "whole_chat", "individual_messages", "pairs"
# This function is adding the whole chat as a memory entry
def add_to_index_wholechat(chat, index, stored_memory, tokeniser, model):
    combined_chat = ""
    for i, message in enumerate(chat):
        # if i % 2 == 0: #then it is from the agent TODO: THIS IS ONLY CORRECT ONCE THE AGENT IS PROACTIVE. OTHERWISE IT IS THE USER
        #     combined_chat += "<START OF AGENT MESSAGE>" + message + "<END OF AGENT MESSAGE>"
        # else:
        #     combined_chat += "<START OF USER MESSAGE>" + message + "<END OF USER MESSAGE>"
        
        combined_chat += message
    tokenised_chat = get_tokenised_sections([combined_chat], tokeniser)
    embeddings = get_embeddings(tokenised_chat, model)
    index.add(embeddings)
    stored_memory.append(combined_chat)

#putting each individual message as a memory entry
def add_to_index_individual_messages(chat, index, stored_memory, tokeniser, model):
    for i, message in enumerate(chat):
        # if i % 2 == 0: #then it is from the agent TODO: THIS IS ONLY CORRECT ONCE THE AGENT IS PROACTIVE. OTHERWISE IT IS THE USER
        #     tokenised_message = get_tokenised_sections(["<START OF AGENT MESSAGE>" + message + "<END OF AGENT MESSAGE>"], tokeniser)
        # else:
        #     tokenised_message = get_tokenised_sections(["<START OF USER MESSAGE>" + message + "<END OF USER MESSAGE>"], tokeniser)
        tokenised_message = get_tokenised_sections(message, tokeniser)
        embeddings = get_embeddings(tokenised_message, model)
        index.add(embeddings)
        stored_memory.append(tokenised_message)

#putting each pair of user message and agent message as a memory entry
def add_to_index_pairs(chat, index, stored_memory, tokeniser, model):
    for i in range(1, len(chat), 2): #start at 1 because the first message is from the agent TODO: THIS IS ONLY CORRECT ONCE THE AGENT IS PROACTIVE. OTHERWISE IT IS THE USER
        message_pair = chat[i-1:i+1]
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