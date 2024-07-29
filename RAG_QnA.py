import warnings as wn
# Ignore warning messages
wn.filterwarnings('ignore')

from langchain_community.llms import HuggingFaceHub
from api_token import LargeLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from RAG import Retrieval_Augmented_Generation

# Define RAG_Model class
class RAG_Model: 
    
    def __init__(self):
        # Initialize API token for the large language model
        self.token = LargeLanguageModel()
        self.api_key = self.token.get_Key()
        
        # Initialize Retrieval Augmented Generation (RAG)
        self.rag = Retrieval_Augmented_Generation()
        
        # Set up conversation memory
        self.mem = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=False, 
            human_prefix='Human',
            ai_prefix='AI',
            input_key='question',
            verbose=False
        )
        
        # Set up window memory for conversation
        self.window_mem = ConversationBufferWindowMemory(
            k=3,  # Number of messages to remember
            memory_key='chat_history', 
            return_messages=False, 
            human_prefix='Human',
            ai_prefix='AI',
            input_key='question',
            verbose=False
        )
        
        
        # Set HuggingFace model repository ID
        __huggingfaceHub_rep_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        # Define filter terms to stop the generation
        self.filter = ['**Question**:', 'Question:', '**Question:**']
        
        # Set up the language model endpoint
        self.llm = HuggingFaceEndpoint(
            name="Web-Pilot",
            repo_id=__huggingfaceHub_rep_id,
            task="text-generation",
            huggingfacehub_api_token=self.api_key,
            verbose=False,
            # show output in text streaming
            streaming=True,
            temperature=0.9,
            return_full_text=True,
            max_new_tokens=500,
            # Stop sequences is filter for stop criteria
            stop_sequences=self.filter,
            repetition_penalty=1.1
        )
    
    def load_Database(self):
        # create vector database for fetch knowledge from database
        self.database = self.rag.VectorDatabase()
        self.database.delete_collection()
        self.database = self.rag.VectorDatabase()

    
    def __PromptEngineering(self):
        # Define the prompt template
        template = """
        Your name is **WEB-PILOT**, a chatbot that answers user questions based on provided scraped website context. 
        If you don't know the answer, say "I don't know." Keep answers under 60 words, in simple and clear English.
        
        Chat History: {chat_history}
        Context: {context}
        Question: {question}
        Answer:
        """
        
        # Create a prompt with the template
        qa_chain_prompt = PromptTemplate.from_template(
            template=template
        )
        
        # Delete all data in the directory and create a vector database
        # self.rag.delete_all_in_directory()
        
        # Create the chain with the prompt and memory
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.database.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': 3,  # Number of results to return
                    'fetch_k': 50  # Number of results to fetch
                }
            ),
            return_source_documents=False,
            chain_type_kwargs={
                'prompt': qa_chain_prompt,
                'verbose': False,
                'memory': self.window_mem
            }
        )
                
        return chain
        
    def __clean_string(self, input_text):
        # Clean the string from unwanted filter terms
        terms = self.filter
        earliest_position = len(input_text)
        for term in terms:
            position = input_text.find(term)
            if position != -1 and position < earliest_position:
                earliest_position = position
        
        return input_text[:earliest_position].strip()
    
    def generateResponse(self, prompt):
        # Generate a response using the prompt chain
        chain = self.__PromptEngineering()
        response = chain.invoke({
            'query': prompt
        })
        response = response['result']
        response = self.__clean_string(response)
        return response
