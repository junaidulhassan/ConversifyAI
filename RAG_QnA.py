import warnings as wn
wn.filterwarnings('ignore')

from langchain_community.llms import HuggingFaceHub
from api_token import LargeLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from RAG import Retrieval_Augmented_Generation




class RAG_Model: 
    
    
    def __init__(self):
        self.token = LargeLanguageModel()
        self.api_key = self.token.get_Key()
        
        self.rag = Retrieval_Augmented_Generation()
        
        self.mem = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=False, 
            human_prefix='Human',
            ai_prefix='AI',
            input_key='question',
            verbose=False
        )
        
        self.window_mem = ConversationBufferWindowMemory(
            k=3,
            memory_key='chat_history', 
            return_messages=False, 
            human_prefix='Human',
            ai_prefix='AI',
            input_key = 'question',
            verbose=False
        )
        
        __huggingfaceHub_rep_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        self.filter = ['**Human**:','Human:','**Human:**']
        
        self.llm = HuggingFaceEndpoint(
            name="Web-Pilot",
            repo_id= __huggingfaceHub_rep_id,
            task="text-generation",
            huggingfacehub_api_token=self.api_key,
            verbose=False,
            streaming=True,
            temperature=0.9,
            return_full_text=True,
            max_new_tokens=500,
            stop_sequences=self.filter,
            repetition_penalty=1.1
        )
    
    def PromptEngineering(self):
      
        template = """You are a chatbot designed to answer user questions based on the provided context. 
        If you don't know the answer, say "I don't know" and don't make up an answer. 
        Use a maximum of ten sentences and keep your answer informative. 
        Always end with "Thanks for asking!"
        
        Chat History: {chat_history}
        
        Context: {context}
        Question: {question}
        Answer:
        """
        
        qa_chain_prompt = PromptTemplate.from_template(
            template=template
        )
        
        self.rag.delete_all_in_directory()
        database = self.rag.VectorDatabase()
        
        chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            
            retriever = database.as_retriever(
                search_type = "mmr",
                search_kwargs = {
                    'k':3,
                    'fetch_k':50
                }
            ),
            
            return_source_documents = False,
            chain_type_kwargs={
                'prompt':qa_chain_prompt,
                'verbose':False,
                'memory':self.window_mem
            }
        )
                
        return chain
        
    def clean_string(self, input_text):
        terms = self.filter
        earliest_position = len(input_text)
        for term in terms:
            position = input_text.find(term)
            if position != -1 and position < earliest_position:
                earliest_position = position
        
        return input_text[:earliest_position].strip()
    

    def generateResponse(self, prompt):
        chain = self.PromptEngineering()
        response = chain.invoke({
            'query':prompt
        })
        response = response['result']
        response = self.clean_string(response)
        return response
    
    
    
