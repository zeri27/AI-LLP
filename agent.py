from Memory_module import create_index
from Embedder_module import get_model
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor

from Memory_module import search
from Memory_module import add_to_index

from LLM_module import create_agent


class Agent:
    def __init__(self, llm_model_name = "llama3.1", embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.stored_memory = []
        self.index = create_index(384)
        self.embedding_model = get_model(embed_model_name)
        self.llm_model_name = llm_model_name
        self.agent = create_agent([self.fetch_From_Memory, self.save_data_to_memory], llm_model_name=self.llm_model_name)



    #create tool that can be called by the llm to fetch data from memory
    @tool
    def fetch_From_Memory(self, query, k=3):
        """
        Fetch data from memory that can be used to generate a response to the user
        query: any string you think will have the highest similarity to the data you want to fetch. THIS SHOULD BE AS INFORMATIVE AS POSSIBLE TO GET THE BEST RESULTS
        k: the number of entries you want to fetch. BEST IS TO KEEP BELOW 5. 
        return: the data that has the highest similarity to the query
        """
        
        #test if k is a number
        int_k = 0
        try:
            int_k = int(k)
        except:
            return "Please enter a valid number for k"
        
        D, I = search(query, int_k)
        # if len(I) > 0 and I[0][0] != -1:
        #     stored_data = "Previously stored information: " + str(I[0])  # Convert memory to readable format
        #     detokenised_data = tokenizer.decode(stored_data)
        #     return detokenised_data
        information = ""
        # if len(I) > 0 and I[0][0] != -1:
        #     # Retrieve stored text (assuming you stored them in a list)
        #     stored_text = stored_memory[I[0][0]]  # Map index back to original text
        #     return f"Previously stored information: {stored_text}"

        #add all the 10 most similar entries to the response
        for i in range(int_k):
            if I[0][i] != -1:
                information += self.stored_memory[I[0][i]] + "\n"
            
        if information == "":
            return "No information found"
        
        return information
    
    #create tool that can be called by the llm to store data in memory
    @tool
    def save_data_to_memory(data):
        """
        Save data to memory that can be fetched later to generate a response to the user. 
        call this function when you want to save important data like user preferences, user instructions, user personal information.
        data: any string you want to save to memory. please format it in a way that it can be easily fetched later.
        """
        
        add_to_index(data)


    def invoke(self, input):
        result = self.agent.invoke({"input": input})
        return result["output"]



agent = Agent()
tools = [agent.fetch_From_Memory, agent.save_data_to_memory]

print(agent.invoke("Hello"))


