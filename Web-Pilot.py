import streamlit as st
import time
import re
from RAG_QnA import RAG_Model
from scrap import Scraper
import requests
import PyPDF2

st.set_page_config(
    page_title='Conversify AI', 
    layout = 'centered',
    page_icon = '/Images/brain_logo.jpeg', 
    initial_sidebar_state = 'auto',
)
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

def is_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+',
        re.IGNORECASE
    )
    return re.match(youtube_regex, url) is not None

st.sidebar.header("Data Source Selection",divider='rainbow')

options = {
    "Llama-3-8b-instruct": 0,
    "Mistral-7b-instruct-V0.3": 1,
    "Falcon-7b-instruct": 2
}

# Initialize previous model selection if not present
if "previous_model" not in st.session_state:
    st.session_state.previous_model = list(options.keys())[0]

model = st.sidebar.radio(
    label="Select LLM Model",
    options=list(options.keys()),
    index=list(options.keys()).index(st.session_state.previous_model)
)

# Load new model only if selection has changed
if model != st.session_state.previous_model:
    st.session_state.model.Load_llm(
        llm_model=options[model]
    )
    st.session_state.previous_model = model

if "scrap" not in st.session_state:
    st.session_state.scrap = Scraper()
    print("Scraping instance created")

# Add a flag to track if scraping and database loading are done
if "scraping_done" not in st.session_state:
    st.session_state.scraping_done = False

if "database_loaded" not in st.session_state:
    st.session_state.database_loaded = False

def is_pdf_url(url):
    if url.lower().endswith('.pdf'):
        return True
    try:
        response = requests.head(
            url=url, 
            allow_redirects=True
        )
        content_type = response.headers.get('Content-Type', '')
        return content_type.lower() == 'application/pdf'

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False

    
file = st.sidebar.file_uploader(
    label="Upload your Pdf file.",
    accept_multiple_files=False,
    type='pdf'
)

# Input field for the URL
url = st.sidebar.text_input(
    "Enter website or Youtube Video URL you want to chat", 
    "",
    placeholder="https://example.com"
)

# Check if URL is different from the previous one
if "previous_url" not in st.session_state:
    st.session_state.previous_url = ""
    

# Process file only if it's new and not processed before
if file is not None and file.name != st.session_state.get("last_uploaded_file"):
    name = file.name
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    st.text_area(label="PDF",value = text, height=30)
    st.session_state.model.load_Database(
        pdf_text=text,
        is_pdf_file=True
    )
    
    st.session_state.database_loaded = True
    st.session_state.scraping_done = True
    st.session_state.last_uploaded_file = file.name  # Mark this file as processed
    st.sidebar.success("Loaded PDF successfully.")

if url and url != st.session_state.previous_url:
    st.session_state.previous_url = url
    if is_valid_url(url):
        if is_pdf_url(url):
            st.session_state.model.load_Database(
                pdf_url=url, 
                is_pdf=True
            )
            st.session_state.database_loaded = True
            st.session_state.scraping_done = True
            st.sidebar.success("Loaded PDF successfully.")
            
        elif is_youtube_url(url=url):
            st.session_state.model.load_Database(
                youtube_url = url,
                is_youtube_url = True
            )
            print("This is Youtube Video URL...")
            st.session_state.database_loaded = True
            st.session_state.scraping_done = True
            st.sidebar.success("Loaded Youtube Video successfully.")
            
        else:
            try:
                response = st.session_state.scrap.scrape_website(
                    url=url
                )
                if response == 200:
                    st.session_state.model.load_Database()
                    st.session_state.database_loaded = True
                    st.session_state.scraping_done = True
                    st.sidebar.success("Scraped website successfully.")
                else:
                    st.sidebar.error("This website does not allow scraping its content.")
            except Exception as e:
                st.sidebar.error(f"Error scraping the website: {e}")
    else:
        st.sidebar.error("Invalid URL format. Please enter a correct URL.")
elif url == "":
    st.session_state.scraping_done = False

# if st.sidebar.button("Clear Chat"):
#     st.session_state.messages = []
#     st.rerun()

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

st.markdown(
    body='<div style="text-align: center;"><h1>Conversify AI</h1></div>', 
    unsafe_allow_html=True
)

st.markdown(
    body='<div style="text-align: center;">Explore large Contents in seconds 😊</div>', 
    unsafe_allow_html=True
)

st.markdown("""<style>
            .chat-messages {
            margin-bottom: 70px;
            }
        </style>""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_avatar(role):
    if role == "user":
        return "/Images/brain_logo.jpeg"
    else:
        return '/Images/sparkledny.jpeg'

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your prompt here..."):
    if st.session_state.scraping_done or st.session_state.database_loaded:
        if st.session_state.database_loaded:
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            # Display user message in chat message container
            with st.chat_message("user", 
                                 avatar=get_avatar("user")):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant", 
                                 avatar=get_avatar("assistant")):
                with st.spinner("Generating response..."):
                    response = ""
                    response_container = st.empty()  # Create an empty container for the response
                    for word in response_generator(
                        prompt=prompt
                        ):
                        response += word
                        response_container.markdown(
                            body=response.replace("\n", "  \n") + "▌"
                        )  # Add double space for markdown newline
                        time.sleep(0.03)
                        response_container.markdown(
                            body=response.replace("\n", "  \n")
                        )  # Add double space for markdown newline

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            
        else:
            st.error("Database is not loaded. Please add the website URL or upload a PDF first.")
    else:
        st.error("Please add the website URL or upload a PDF first.")
