from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import os

from memory.Memory_module import search



def create_agent(tools, memory, llm_model_name="llama3.1"):
    #use langchain agents for an integration of tool calling into the llm
    agents_llm = ChatOllama(model=llm_model_name, temperature=1.0)

    llm_with_tools = agents_llm.bind_tools(tools)

    if os.path.exists("NLP/system.txt"):
        with open("NLP/system.txt", "r", encoding="utf-8") as f:
            system_message = f.read()
    else:
        print("system.txt not found")	
        system_message = ""

    prompt = ChatPromptTemplate.from_messages([
        # ("system", """YOU ARE A CONVERSATIONAL AGENT WHOSE TASK IT IS TO HELP PEOPLE WITH STUDYING LANGUAGES. 
        
        #  YOUR NAME IS AILLP.
        
        #   YOU HAVE ACCESS TO A MEMORY THAT YOU CAN USE TO FETCH FROM PAST SESSIONS AND WRITE USER INFO TO REMEMBER.
        #   THE CONTENTS OF THE MEMORY IS ALL FROM THE USER AND THEIR PREFERENCES OR THEIR INSTRUCTIONS OR THEIR INFO OR THEIR PAST SESSIONS WITH YOU.
        #   EACH TIME YOU START TALKING TO A NEW USER, YOU ABSOLUTELY MUST FETCH FROM THE MEMORY TO SEE IF YOU HAVE ANY INFO ABOUT THE USER SUCH AS THEIR NAME, AGE, PREFERENCES, INSTRUCTIONS, OR PAST SESSIONS.
        #   FOR EXAMPLE YOU CAN FETCH \"NAME EVA USER PREFERENCES\" TO GET THE USER PREFERENCES OF EVA.
        #   GIVE A BRIEF RESPONSE TO THE USER.
        #   IF YOU DON'T USE YOUR MEMORY THEN JUST ANSWER THE USERS QUESTION WITHOUT SAYING ANYTHING ABOUT THE MEMORY."""),
        #("system", "if needed you can try to fetch from memory using the tools provided. It is not necessary to use the tools."),
        #("system", "you have to first read the users message!. Then if you know how to answer and in what way you have to answer, you can answer the user. If you don't know how to answer or in what way you have to use the tools to fetch from your memory by providing a query to the fetch_From_Memory tool. Then wait for the response of the tool and then you can answer the user. do not mention anything about using your tools to the user."),
        # ("system", """YOU ARE A CONVERSATIONAL AGENT WHO HELPS USERS WITH LANGUAGE LEARNING. 
        # YOUR NAME IS AILLP.
        # YOU HAVE ACCESS TO A MEMORY TOOL TO FETCH PAST USER SESSIONS AND PREFERENCES.
        
        # AT THE START OF EVERY CONVERSATION, YOU MUST CALL THE `fetch_From_Memory` TOOL USING THE USER'S NAME.
        # YOU MUST CALL THIS TOOL BEFORE RESPONDING.
        
        # NEVER GUESS PAST SESSIONS. ALWAYS FETCH MEMORY FIRST. IF YOU CANNOT FIND THE KNOWLEDGE FROM THE MEMORY TOOL, USE YOUR CONTEXT WINDOW TO RESPOND.
        # IF MEMORY IS FOUND, SUMMARIZE IT BEFORE ASKING THE USER HOW THEY WANT TO CONTINUE.
        # IF NO MEMORY IS FOUND, CONTINUE AS USUAL WITHOUT MENTIONING MEMORY.
        # WHEN THE USER GIVES YOU AN INSTRUCTION TO ALWAYS REMEMBER, USE THE `save_data_to_memory` TOOL TO SAVE THE DATA.

        # NEVER MENTION THAT YOU ARE USING A TOOL TO THE USER."""),


        ("system", system_message),

        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, memory=memory)

    return agent_executor
