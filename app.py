import streamlit as st
import os
import tempfile
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_dataframes" not in st.session_state:
    st.session_state.file_dataframes = {}

# Streamlit app configuration
st.set_page_config(page_title="OpenAI RAG Chat Assistant", page_icon="🤖")

# Sidebar configuration
with st.sidebar:
    st.title("OpenAI Settings")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", ""))
    selected_model = st.selectbox("Choose a Model", ["gpt-4o", "gpt-3.5-turbo"])
    
    st.markdown("---")
    st.markdown("### File Upload Settings")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Word, CSV, Excel)",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True
    )

# Function to process uploaded files and create chunks
def process_files(uploaded_files, chunk_size=1000, chunk_overlap=200):
    all_text = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    for file in uploaded_files:
        file_type = file.name.split(".")[-1]
        file_name = file.name
        
        try:
            if file_type == "pdf":
                reader = PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages])
            elif file_type == "docx":
                doc = Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_type in ["csv", "xlsx"]:
                df = pd.read_csv(file) if file_type == "csv" else pd.read_excel(file)
                st.session_state.file_dataframes[file_name] = df
                text = df.to_string()
            else:
                text = f"Unsupported file type: {file_type}"
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            all_text.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
        
        except Exception as e:
            all_text.append(f"Error processing {file_name}: {str(e)}")
    
    return all_text

# Function to create vector store
def create_vector_store(text_chunks, api_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        documents = [LangchainDocument(page_content=chunk) for chunk in text_chunks]
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Process files when uploaded
if uploaded_files:
    st.session_state.processed_text = process_files(uploaded_files)
    if openai_api_key and st.session_state.processed_text:
        st.session_state.vector_store = create_vector_store(st.session_state.processed_text, openai_api_key)

# Function to retrieve row from CSV/Excel
def retrieve_row(file_name, row_index):
    if file_name in st.session_state.file_dataframes:
        df = st.session_state.file_dataframes[file_name]
        try:
            row_index = int(row_index)
            if 0 <= row_index < len(df):
                return df.iloc[row_index].to_string()
            else:
                return f"Row index {row_index} is out of range for {file_name}"
        except ValueError:
            return "Please provide a valid row index number"
    return f"No data found for file: {file_name}"

# Main chat interface
st.title("💬 RAG Chat Assistant")
st.caption("🚀 A chatbot powered by OpenAI with RAG capabilities")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Ask me anything..."):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar or configure it in secrets")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Check if the prompt is requesting a specific row from CSV/Excel
        if "row" in prompt.lower() and any(ext in prompt.lower() for ext in ["csv", "xlsx"]):
            # Parse the prompt for file name and row index
            words = prompt.split()
            file_name = None
            row_index = None
            for i, word in enumerate(words):
                if any(word.endswith(ext) for ext in [".csv", ".xlsx"]):
                    file_name = word
                elif word.isdigit():
                    row_index = word
            
            if file_name and row_index:
                row_data = retrieve_row(file_name, row_index)
                st.session_state.messages.append({"role": "assistant", "content": row_data})
                with st.chat_message("assistant"):
                    st.markdown(row_data)
                st.stop()
        
        # RAG implementation
        context = ""
        if st.session_state.vector_store:
            # Retrieve relevant chunks
            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prepare messages for OpenAI
        system_message = {
            "role": "system",
            "content": f"""
            You are a helpful RAG-based assistant. Answer the user's question based on the following context from uploaded files:
            {context}
            
            If the context doesn't contain relevant information, use your general knowledge to provide a helpful response.
            """
        }
        
        messages = [system_message] + [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages
        ]
        
        # Create chat completion
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0.5,
            max_tokens=7000
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
