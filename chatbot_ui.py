import streamlit as st
import time
import re
from LLM import ChatModel
from RAG_QnA import RAG_Model
from scrap import Scraper

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


def reset_scrap_state():
    st.session_state.scrap = None

# Input field for the URL
url = st.sidebar.text_input("Enter website URL you want to chat", 
                            "", placeholder="https://example.com",
                            on_change=reset_scrap_state
                           )

# Scrape the website if a valid URL is entered
if url:
    if is_valid_url(url):
        if st.session_state.scrap is None:
            try:
                st.session_state.scrap = Scraper(url=url)
                texts = st.session_state.scrap.scrape_website()
                st.write(texts)
            except Exception as e:
                st.sidebar.error(f"Error scraping the website: {e}")
    else:
        st.sidebar.error("Invalid URL format. Please enter a correct URL.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.model.window_mem.clear()
    st.experimental_rerun()

# Streamed response emulator
def response_generator(prompt):
    response = st.session_state.model.generateResponse(prompt=prompt)
    return response

st.markdown('<div style="text-align: center;"><h1>WEB PILOT</h1></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center;">Chat with Any Website 😊</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your prompt here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = ""
            response_container = st.empty()  # Create an empty container for the response
            for word in response_generator(prompt):
                response += word
                response_container.markdown(response.replace("\n", "  \n") + "▌")  # Add double space for markdown newline
                time.sleep(0.03)
                response_container.markdown(response.replace("\n", "  \n"))  # Add double space for markdown newline

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
