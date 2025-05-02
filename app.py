import streamlit as st
import os
import tempfile
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import math
import tiktoken

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
        "Upload up to 4 files (PDF, Word, CSV, Excel)",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True,
        help="You can upload a maximum of 4 files at once."
    )
    
    # Enforce maximum of 4 files
    if uploaded_files and len(uploaded_files) > 4:
        st.error("You can only upload up to 4 files. Please remove some files.")
        uploaded_files = uploaded_files[:4]
        st.session_state.uploaded_files = uploaded_files

# Function to chunk large text
def chunk_text(text, max_chunk_size=10000):
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    return chunks

# Function to estimate tokens
def estimate_tokens(text, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: conservative estimate of 3 characters per token
        return len(text) // 3

# Function to truncate context to fit token limit
def truncate_context(context, messages, relevant_file=None, max_tokens=15000, model="gpt-4o"):
    system_prompt = "You are a helpful assistant."
    total_tokens = estimate_tokens(system_prompt + context, model)
    
    # Estimate tokens for messages
    message_tokens = sum(estimate_tokens(m["content"], model) for m in messages)
    total_tokens += message_tokens
    
    # If within limit, return as is
    if total_tokens <= max_tokens:
        return context, messages
    
    # Prioritize chunks from relevant file (if specified)
    processed_text = st.session_state.processed_text
    if relevant_file:
        prioritized_chunks = [chunk for chunk in processed_text if relevant_file in chunk]
        other_chunks = [chunk for chunk in processed_text if relevant_file not in chunk]
        processed_text = prioritized_chunks + other_chunks
    
    # Truncate processed_text chunks
    while processed_text and total_tokens > max_tokens:
        processed_text.pop()
        context = f"""
        User Question: {messages[-1]["content"]}
        
        Uploaded Files Context (chunked):
        {'\n\n'.join(processed_text)}
        
        Row Fetch Result (if applicable): {row_fetch_response}
        """
        total_tokens = estimate_tokens(system_prompt + context, model) + message_tokens
    
    # If still over limit, truncate older messages
    truncated_messages = messages.copy()
    while truncated_messages and total_tokens > max_tokens and len(truncated_messages) > 1:
        truncated_messages.pop(0)  # Remove oldest message
        message_tokens = sum(estimate_tokens(m["content"], model) for m in truncated_messages)
        total_tokens = estimate_tokens(system_prompt + context, model) + message_tokens
    
    return context, truncated_messages

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

# Function to fetch row from dataframe based on column value
def fetch_row(file_name, column_name, column_value):
    if file_name in st.session_state.dataframes:
        df = st.session_state.dataframes[file_name]
        try:
            # Search for rows where the column matches the value
            if column_name in df.columns:
                result = df[df[column_name] == column_value]
                if not result.empty:
                    return result.to_string(index=False)
                else:
                    return f"No rows found for {column_name} = {column_value} in {file_name}"
            else:
                return f"Column {column_name} not found in {file_name}"
        except Exception as e:
            return f"Error processing request: {str(e)}"
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
    relevant_file = None
    if "fetch row" in prompt.lower() or "get row" in prompt.lower():
        try:
            # Handle queries like "fetch row for customer id CUST005 in filename.csv"
            words = prompt.lower().split()
            column_value = None
            column_name = None
            file_name = None
            
            # Look for column value (e.g., CUST005)
            for i, word in enumerate(words):
                if word.startswith("cust") and word.isalnum():  # Assuming customer IDs are alphanumeric
                    column_value = word.upper()
                elif any(word.endswith(ext) for ext in ['.csv', '.xlsx']):
                    file_name = word
                elif "id" in word:
                    column_name = "customer id"  # Adjust based on common column names
            
            if column_value and file_name:
                relevant_file = file_name
                row_fetch_response = fetch_row(file_name, column_name or "customer id", column_value)
            elif column_value:
                # Try all CSV/Excel files if no file specified
                for fname in st.session_state.dataframes:
                    if fname.endswith(('.csv', '.xlsx')):
                        result = fetch_row(fname, column_name or "customer id", column_value)
                        if not result.startswith(("No rows", "Column", "No dataframe", "Error")):
                            row_fetch_response = result
                            relevant_file = fname
                            break
                        row_fetch_response = result  # Store last result
            else:
                row_fetch_response = "Unable to process row fetch request. Please use format: 'fetch row for [column] [value] in [filename]'"
        except:
            row_fetch_response = "Unable to process row fetch request. Please use format: 'fetch row for [column] [value] in [filename]'"

    # Prepare context
    context = f"""
    User Question: {prompt}
    
    Uploaded Files Context (chunked):
    {'\n\n'.join(st.session_state.processed_text)}
    
    Row Fetch Result (if applicable): {row_fetch_response}
    
    Please answer the question based on the provided context, row fetch result (if any), and your general knowledge.
    """
    
    # Truncate context if necessary, prioritizing relevant file
    context, messages_to_send = truncate_context(context, st.session_state.messages, relevant_file)
    
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                     [{"role": m["role"], "content": context + m["content"]} 
                      for m in messages_to_send],
            temperature=0.5,
            max_tokens=5000
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
