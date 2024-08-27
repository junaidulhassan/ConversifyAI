from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain.prompts import MessagesPlaceholder
import warnings as wn
wn.filterwarnings('ignore')

# now intialize the Mistral LLMs model
# now its time bruh


class MistralAI():
    
    def __init__(self) -> None:
        pass
    
    def model_initalization(self):
        path = "/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/Jupyter-Framework/LLMA_Models/mistral-7b-instruct-v0.2.Q4_0.gguf"
        llm = LlamaCpp(
        model_path = path,
        temperature = 0.1,
        max_token=500,
        top_p = 1,
        verbose=0
        )
        return llm
        
    
    def UserInput(self, prompt):
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
            )
        model = self.model_initalization()
        output = model(prompt)
        
        return output
        
        
    
