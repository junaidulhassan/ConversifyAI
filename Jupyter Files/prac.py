import streamlit as st
from langchain.document_loaders import PyPDFLoader
import streamlit as st
import PyPDF2

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    print(pdf_reader)
    
    # Initialize text container
    text = ""
    
    # Iterate through the pages and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Display the extracted text
    st.write("Here's the extracted text from the PDF file:")
    st.text_area("PDF Content", text, height=300)
else:
    st.write("No PDF file uploaded yet.")
