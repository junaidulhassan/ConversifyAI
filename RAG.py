import warnings as wn
wn.filterwarnings('ignore')
import os
import shutil

from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.vectorstores import pinecone

class Retrieval_Augmented_Generation:
    
    __DB_path = "/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/LLMs Projects/Web_pilot/Web-Content/Docs/Chroma"
    
    def __init__(self):
        self.embedding_model = self.__embed()
    
    def __load_docs(self):
        try:
            # Load documents from file path
            loader = TextLoader(
                file_path="/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/LLMs Projects/Web_pilot/text_file.txt/text_file.txt"
            )
            
            docs = loader.load()
            return docs
        
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None
    
    def __text_spliter(self, chunks_size=500, chunks_overlap=50):
        # Define the chunks and overlap
        # now define the chunks and overlap

        chunks_size = 1000
        chunks_overlap = 40

        rec_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            chunk_size=chunks_size,
            chunk_overlap=chunks_overlap,
            is_separator_regex=False
        )
        
        split = rec_splitter.split_documents(
            self.__load_docs()
        )
        
        return split
    
    def __embed(self):
        # Embed the text
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        return embeddings
    
    def VectorDatabase(self):
        
        chunk_size = 1000
        chunk_overlap = 50
        
        split = self.__text_spliter(
            chunks_size=chunk_size,
            chunks_overlap=chunk_overlap
        )
        
        db = Chroma.from_documents(
            documents=split,
            embedding=self.embedding_model,
            collection_name='Web_vectors',
            persist_directory=self.__DB_path,
        )
        
        return db
    
        
    def delete_all_in_directory(self):
        
        directory_path = self.__DB_path
    
        if not os.path.exists(directory_path):
            print(f"The directory {directory_path} does not exist.")
            return
        else:
            db = self.VectorDatabase()
            db.delete_collection()
            return "Collection Deleted"
    
    