from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader


class Retrievel_Augmented_Generation:
    
    def __init__(self, file_path):
        pass
    
    def __Loader_text():
        loader = TextLoader