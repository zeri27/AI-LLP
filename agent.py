import os
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory.Embedder_module import get_model
from memory.Tokeniser_module import get_tokeniser
from NLP.LLM_module import create_agent
from TTS.TTS_module import tts_pipeline
from memory.Memory_module import search, add_to_index, create_index, save_chat_to_memory
from memory.Memory_module import save_index, load_index, save_reference, load_reference
from ASR.ASR_module import listen_for_speech, transcribe_audio, save_audio

class Agent:
    def __init__(self, llm_model_name = "llama3.1", embed_model_name = "sentence-transformers/all-MiniLM-L6-v2", index_path = None):
        if index_path is not None:
            newpath = "long_term_memory/" + index_path
            self.stored_memory = load_reference(newpath)
            self.index = load_index(newpath)
        else:
            self.stored_memory = []
            self.index = create_index(384)
        self.embedding_model = get_model(embed_model_name)
        self.tokeniser = get_tokeniser(embed_model_name)
        self.llm_model_name = llm_model_name
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = [
            Tool(name="fetch_From_Memory", func=self.fetch_From_Memory, description=self.fetch_From_Memory.__doc__),
            Tool(name="save_data_to_memory", func=self.save_data_to_memory, description=self.save_data_to_memory.__doc__)
        ]
        self.agent = create_agent(self.tools, llm_model_name=self.llm_model_name, memory=self.short_term_memory)
        #self.agent = create_agent([], self.short_term_memory, llm_model_name=self.llm_model_name)

    #create tool that can be called by the llm to fetch data from memory
    
    def fetch_From_Memory(self, query):
        """
        Fetch data from memory that can be used to generate a response to the user. don't call the tool with json data.
        query: any string you think will have the highest similarity to the data you want to fetch. THIS SHOULD BE AS INFORMATIVE AS POSSIBLE TO GET THE BEST RESULTS
        return: the data that has the highest similarity to the query
        """
        
        #test if k is a number
        #k: the number of entries you want to fetch. BEST IS TO KEEP BELOW 5.  
        k = 10
        int_k = 0
        try:
            int_k = int(k)
        except:
            return "Please enter a valid number for k"
        
        D, I = search(query, int_k, self.index, self.tokeniser, self.embedding_model)
        # if len(I) > 0 and I[0][0] != -1:
        #     stored_data = "Previously stored information: " + str(I[0])  # Convert memory to readable format
        #     detokenised_data = tokenizer.decode(stored_data)
        #     return detokenised_data
        information = "<START OF MEMORY DATA><ENTRY 1>"
        # if len(I) > 0 and I[0][0] != -1:
        #     # Retrieve stored text (assuming you stored them in a list)
        #     stored_text = stored_memory[I[0][0]]  # Map index back to original text
        #     return f"Previously stored information: {stored_text}"

        #add all the 10 most similar entries to the response
        for i in range(int_k):
            if I[0][i] != -1:
                information += self.stored_memory[I[0][i]] + "\n<ENTRY " + str(i+1) + ">"
        
        information += "<END OF MEMORY DATA>"
            
        if information == "<START OF MEMORY DATA><ENTRY 1><END OF MEMORY DATA>":
            return "No information found"
        
        information += "USE THE INFORMATION TO GENERATE A RESPONSE TO THE USER:"
        
        return information
    
    #create tool that can be called by the llm to store data in memory
    
    def save_data_to_memory(self, data : str):
        """
        Save data to memory that can be fetched later to generate a response to the user.  don't call the tool with json data.
        call this function when you want to save important data like user preferences, user instructions, user personal information.
        data: any string you want to save to memory. please format it in a way that it can be easily fetched later.
        """
        
        add_to_index(data, self.index, self.stored_memory, self.tokeniser, self.embedding_model)

    # def invoke(self, input):
    #     result = self.agent.invoke({"input": input})
    #     return result["output"]

    def chat(self, input):
        result = self.agent.invoke({"input": input})
        return result["output"]

    def save_index(self, path):
        path = "long_term_memory/" + path
        save_index(self.index, path)
        save_reference(self.stored_memory, path)

    def convert_short_to_long_memory(self):
        #create a list of strings from the buffer
        chat = []
        for message in self.short_term_memory.buffer:
            if type(message) == HumanMessage:
                chat.append("<START OF USER MESSAGE>" + message.content + "<END OF USER MESSAGE>")
            elif type(message) == AIMessage:
                chat.append("<START OF AGENT MESSAGE>" + message.content + "<END OF AGENT MESSAGE>")
        
        save_chat_to_memory(chat, self.index, self.stored_memory, self.tokeniser, self.embedding_model)

if os.path.exists("long_term_memory/jordy.index"):
    agent = Agent(index_path="jordy")
else:
    agent = Agent()
#agent = Agent()
tools = [agent.fetch_From_Memory, agent.save_data_to_memory]

import wave
from ASR.ASR_module import listen_for_speech, transcribe_audio

while True:
    print("\nWaiting for input (speak or type)...")

    # Listen for speech
    audio_data = listen_for_speech(silence_duration_ms=2000) # Stop after 2 seconds of silence
    
    if audio_data:
        # Save to a temporary file
        audio_file = "audio_recorded.wav"
        save_audio(audio_data, "audio_recorded.wav")
        
        # Transcribe the speech
        transcription = transcribe_audio(audio_file)
        if transcription:
            response = agent.chat(transcription)
            tts_pipeline(response)

    else:
        # Fallback: Allow text input
        message = input("Enter a message (or type 'exit' to quit): ")
        if message.lower() == "exit":
            break
        response = agent.chat(message)
        tts_pipeline(response)  

agent.convert_short_to_long_memory()

agent.save_index("jordy")
