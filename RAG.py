import warnings as wn
import os
import shutil
from langchain_community.llms import HuggingFaceHub
from api_token import LargeLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma, pinecone

# Ignore warning messages
wn.filterwarnings('ignore')

# Define the RetrievalAugmentedGeneration class
class RetrievalAugmentedGeneration:
    
    # Define the path for the database
    __DB_path = "/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/LLMs Projects/Web_pilot/Web-Content/Docs/Chroma"
    
    def __init__(self):
        # Initialize API token for the large language model
        self.token = LargeLanguageModel()
        self.api_key = self.token.get_Key()
        
        # Initialize the embedding model
        self.embedding_model = self.__embed()
        
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
    
    def __load_docs(self):
        try:
            # Load documents from file path
            loader = TextLoader(
                file_path="/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/LLMs Projects/Web_pilot/text_file.txt/text_file.txt"
            )
            
            # Load documents using the loader
            docs = loader.load()
            return docs
        
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None
    
    def __text_spliter(self, chunks_size=500, chunks_overlap=50):
        # Define the chunks and overlap
        chunks_size = 1000
        chunks_overlap = 40

        # Use RecursiveCharacterTextSplitter to split documents into chunks
        rec_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            chunk_size=chunks_size,
            chunk_overlap=chunks_overlap,
            is_separator_regex=False
        )
        
        # Split the loaded documents into chunks
        split = rec_splitter.split_documents(
            self.__load_docs()
        )
        
        return split
    
    def __embed(self):
        # Create an embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        return embeddings
    
    def VectorDatabase(self):
        # Define chunk size and overlap for splitting
        chunk_size = 1000
        chunk_overlap = 50
        
        # Split the documents into chunks
        split = self.__text_spliter(
            chunks_size=chunk_size,
            chunks_overlap=chunk_overlap
        )
        
        # Create a vector database using the split documents and embeddings
        db = Chroma.from_documents(
            documents=split,
            embedding=self.embedding_model,
            collection_name='Web_vectors',
            persist_directory=self.__DB_path,
        )
        
        return db
    
    def delete_all_in_directory(self):
        # Define the directory path
        directory_path = self.__DB_path
    
        if not os.path.exists(
            directory_path
        ):
            print(f"The directory {directory_path} does not exist.")
            return
        else:
            # Delete the collection in the vector database
            db = self.VectorDatabase()
            db.delete_collection()
            return "Collection Deleted"

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
        
        # Create the chain with the prompt and memory
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.VectorDatabase().as_retriever(
                search_type="mmr", # search type is the techique to search documents
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
    
