import streamlit as st
import time
import re

from RAG_QnA import RAG_Model
from scrap import Scraper
import requests


# Initialize the model only once and store it in the session state
if "model" not in st.session_state:
    st.session_state.model = RAG_Model()
    print("Model initialized")

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


if "scrap" not in st.session_state:
    st.session_state.scrap = Scraper()
    print("Scrapping instance created")

# Add a flag to track if scraping and database loading are done
if "scraping_done" not in st.session_state:
    st.session_state.scraping_done = False
    
def is_pdf_url(url):
    if url.lower().endswith('.pdf'):
        return True
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        return content_type.lower() == 'application/pdf'
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False

# Input field for the URL
url = st.sidebar.text_input("Enter website URL you want to chat", 
                            "", placeholder="https://example.com")

# Scrape the website if a valid URL is entered and scraping is not already done
if url and not st.session_state.scraping_done:
    if is_valid_url(
        url=url
    ):
        if is_pdf_url(url):
            st.session_state.model.load_Database(
                pdf_url=url,
                is_pdf=True
            )
        else:
            try:
                response = st.session_state.scrap.scrape_website(
                    url=url
                )
                if response == 200:
                    st.session_state.model.load_Database()
                    print("LOAD DATABASE in streamlit code...")
                    st.sidebar.success("Scraped website successfully.")
                    st.session_state.scraping_done = True
                else:
                    st.sidebar.error("This website does not allow scraping its content.")
            except Exception as e:
                st.sidebar.error(f"Error scraping the website: {e}")
    else:
        st.sidebar.error("Invalid URL format. Please enter a correct URL.")
        
if url == "":
    st.session_state.scraping_done = False

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.model.window_mem.clear()
    st.rerun()

# Streamed response emulator
def response_generator(prompt):
    response = st.session_state.model.generateResponse(
        prompt=prompt
    )
    
    return response

st.markdown('<div style="text-align: center;"><h1>WEB PILOT</h1></div>', 
            unsafe_allow_html=True)

st.markdown('<div style="text-align: center;">Chat with Any Website 😊</div>', 
            unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your prompt here..."):
    if st.session_state.scraping_done:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = ""
                response_container = st.empty()  # Create an empty container for the response
                for word in response_generator(
                    prompt=prompt
                ):
                    response += word
                    response_container.markdown(response.replace("\n", "  \n") + "▌")  # Add double space for markdown newline
                    time.sleep(0.04)
                    response_container.markdown(response.replace("\n", "  \n"))  # Add double space for markdown newline
                    
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })
        
    else:
        st.error("Please add the website URL first.")
