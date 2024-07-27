import warnings as wn
wn.filterwarnings('ignore')

from langchain_community.llms import HuggingFaceHub
from api_token import LargeLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import requests
import json
import os
import re



class ChatModel: 
    
    
    def __init__(self):
        self.token = LargeLanguageModel()
        self.api_key = self.token.get_Key()
        
        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key='chat_history', 
            return_messages=False, 
            human_prefix='Human',
            ai_prefix='Faraz',
            verbose=False
        )
        
        
        huggingfaceHub_rep_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        self.filter = ['**Human**:','Human:','**Human:**']
        
        self.llm = HuggingFaceEndpoint(
            name="Web-Pilot",
            repo_id= huggingfaceHub_rep_id,
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
        
        prompt_template = """
        * Your name is **FITNESS RAT**, a fitness chatbot designed to motivate and guide people on their fitness journey. 
        * Provide clear and encouraging fitness advice and tips. 
        * Don't create any questions or answers on your own. 
        * Keep your responses to one sentence maximum.

        {chat_history}
        
        **Human:** {context} 
        **You:**
        """
        
        prompt = PromptTemplate(
            input_variables=["chat_history","context"],
            template=prompt_template
        )

        chain = LLMChain(
            llm = self.llm,
            prompt = prompt,
            memory = self.memory,
            verbose=False
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
        response = self.PromptEngineering().invoke(prompt)
        response = response['text']
        response = self.clean_string(response)
        return response
    
    
