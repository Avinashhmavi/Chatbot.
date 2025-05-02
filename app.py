import streamlit as st
import os
import tempfile
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import math

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_text" not in st.session_state:
    st.session_state.processed_text = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

# Streamlit app configuration
st.set_page_config(page_title="Chat Assistant", page_icon="🤖")

# Sidebar configuration
with st.sidebar:
    st.title("File Upload")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Word, CSV, Excel)",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True
    )

# Function to chunk large text
def chunk_text(text, max_chunk_size=10000):
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    return chunks

# Process uploaded files
def process_files(uploaded_files):
    all_text_chunks = []
    dataframes = {}
    
    for file in uploaded_files:
        file_type = file.name.split(".")[-1]
        file_name = file.name
        
        try:
            if file_type == "pdf":
                reader = PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages])
                chunks = chunk_text(text)
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            elif file_type == "docx":
                doc = Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
                chunks = chunk_text(text)
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            elif file_type in ["csv", "xlsx"]:
                if file_type == "csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                # Store dataframe for row fetching
                dataframes[file_name] = df
                
                # Convert to string for text context
                text = df.to_string()
                chunks = chunk_text(text)
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            else:
                all_text_chunks.append(f"Unsupported file type: {file_type} for file: {file_name}")
                
        except Exception as e:
            all_text_chunks.append(f"Error processing {file_name}: {str(e)}")
    
    return all_text_chunks, dataframes

# Process files when uploaded
if uploaded_files:
    st.session_state.processed_text, st.session_state.dataframes = process_files(uploaded_files)

# Main chat interface
st.title("💬 Chat Assistant")
st.caption("🚀 A chatbot powered by OpenAI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to fetch row from dataframe
def fetch_row(file_name, row_index):
    if file_name in st.session_state.dataframes:
        df = st.session_state.dataframes[file_name]
        try:
            row_index = int(row_index)
            if 0 <= row_index < len(df):
                return df.iloc[row_index].to_string()
            else:
                return f"Row index {row_index} is out of range for {file_name}"
        except ValueError:
            return "Invalid row index. Please provide a number."
    return f"No dataframe found for {file_name}"

# Chat input and processing
if prompt := st.chat_input("Ask me anything..."):
    # Initialize OpenAI client with API key from secrets
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error("Error initializing OpenAI client. Please check your API key in secrets.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if user is asking to fetch a row
    row_fetch_response = ""
    if "fetch row" in prompt.lower() or "get row" in prompt.lower():
        words = prompt.split()
        try:
            # Assuming format like "fetch row 5 from filename.csv"
            row_index = None
            file_name = None
            for i, word in enumerate(words):
                if word.isdigit():
                    row_index = word
                elif any(word.endswith(ext) for ext in ['.csv', '.xlsx']):
                    file_name = word
            
            if row_index and file_name:
                row_fetch_response = fetch_row(file_name, row_index)
        except:
            row_fetch_response = "Unable to process row fetch request. Please use format: 'fetch row [number] from [filename]'"

    # Prepare context with all chunks
    context = f"""
    User Question: {prompt}
    
    Uploaded Files Context (chunked):
    {'\n\n'.join(st.session_state.processed_text)}
    
    Row Fetch Result (if applicable): {row_fetch_response}
    
    Please answer the question based on the provided context, row fetch result (if any), and your general knowledge.
    """
    
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",  # Using a default OpenAI model
            messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                     [{"role": m["role"], "content": context + m["content"]} 
                      for m in st.session_state.messages],
            temperature=0.5,
            max_tokens=10000
        )
        
        # Get AI response
        ai_response = response.choices[0].message.content
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(ai_response)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
