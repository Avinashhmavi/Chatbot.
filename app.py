import streamlit as st
import os
import tempfile
from groq import Groq, RateLimitError
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import time
import re
import uuid
import base64
from PIL import Image
import io

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_text" not in st.session_state:
    st.session_state.processed_text = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "file_keywords" not in st.session_state:
    st.session_state.file_keywords = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

# Streamlit app configuration
st.set_page_config(page_title="Chat Assistant", page_icon="🤖")

# Sidebar configuration
with st.sidebar:
    st.title("File Upload")
    uploaded_files = st.file_uploader(
        "Upload up to 4 files (PDF, Word, CSV, Excel, Images)",
        type=["pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="You can upload a maximum of 4 files at once."
    )
    
    # Enforce maximum of 4 files
    if uploaded_files and len(uploaded_files) > 4:
        st.error("You can only upload up to 4 files. Please remove some files.")
        uploaded_files = uploaded_files[:4]
        st.session_state.uploaded_files = uploaded_files

# Function to chunk large text
def chunk_text(text, max_chunk_size=3000):
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    return chunks

# Function to estimate tokens (compatible with Groq and OpenAI)
def estimate_tokens(text, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    # Conservative estimate: 2 characters per token
    return len(text) // 2 + 100  # Add padding for safety

# Function to reset session state
def reset_session():
    st.session_state.messages = []
    st.session_state.processed_text = []
    st.session_state.dataframes = {}
    st.session_state.file_keywords = []
    st.session_state.uploaded_images = {}
    st.session_state.session_id = str(uuid.uuid4())
    st.warning("Chat session has been reset due to approaching token limit. Starting fresh.")

# Function to truncate context to fit token limit
def truncate_context(context, messages, relevant_file=None, row_fetch_response="", max_tokens=7000, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    system_prompt = "You are a helpful assistant. Answer based on provided file content, image content, or general knowledge if no relevant files are provided."
    total_tokens = estimate_tokens(system_prompt + context, model)
    
    # Estimate tokens for messages
    message_tokens = sum(estimate_tokens(m["content"], model) for m in messages)
    total_tokens += message_tokens
    
    # If approaching limit, reset session
    if total_tokens > 10000:
        reset_session()
        return "", []
    
    # If within limit, return as is
    if total_tokens <= max_tokens:
        return context, messages
    
    # Prioritize chunks from relevant file
    processed_text = st.session_state.processed_text.copy()
    if relevant_file:
        prioritized_chunks = [chunk for chunk in processed_text if relevant_file in chunk]
        other_chunks = [chunk for chunk in processed_text if relevant_file not in chunk]
        processed_text = prioritized_chunks + other_chunks
    
    # Truncate processed_text chunks aggressively
    while processed_text and total_tokens > max_tokens:
        processed_text.pop()
        context = f"""
        User Question: {messages[-1]["content"] if messages else ""}
        
        Uploaded Files Context (chunked, limited to 1 chunk):
        {'\n\n'.join(processed_text[:1])}
        
        Row Fetch Result (if applicable): {row_fetch_response}
        """
        total_tokens = estimate_tokens(system_prompt + context, model) + message_tokens
    
    # Keep only the last message
    truncated_messages = [messages[-1]] if messages else []
    message_tokens = sum(estimate_tokens(m["content"], model) for m in truncated_messages)
    total_tokens = estimate_tokens(system_prompt + context, model) + message_tokens
    
    return context, truncated_messages

# Function to extract keywords from file content
def extract_file_keywords(processed_text, file_names):
    keywords = set(file_names)  # Include file names as keywords
    for chunk in processed_text:
        # Extract words with 5+ characters, excluding common words
        words = re.findall(r'\b\w{5,}\b', chunk.lower())
        keywords.update([word for word in words if word not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have']])
    return list(keywords)[:10]  # Limit to top 10 unique keywords

# Function to process image content
def process_image(image_file):
    try:
        image = Image.open(image_file)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_image, image_file.name
    except Exception as e:
        return None, f"Error processing image {image_file.name}: {str(e)}"

# Process uploaded files
def process_files(uploaded_files):
    all_text_chunks = []
    dataframes = {}
    uploaded_images = {}
    max_chunks_per_file = 1
    file_names = [file.name for file in uploaded_files]
    
    for file in uploaded_files:
        file_type = file.name.split(".")[-1].lower()
        file_name = file.name
        
        try:
            if file_type == "pdf":
                reader = PdfReader(file)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                chunks = chunk_text(text)[:max_chunks_per_file]
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            elif file_type == "docx":
                doc = Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
                chunks = chunk_text(text)[:max_chunks_per_file]
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            elif file_type in ["csv", "xlsx"]:
                if file_type == "csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                dataframes[file_name] = df
                
                text = df.head(10).to_string()
                chunks = chunk_text(text)[:max_chunks_per_file]
                all_text_chunks.extend([f"File: {file_name}\n{chunk}" for chunk in chunks])
                
            elif file_type in ["png", "jpg", "jpeg"]:
                base64_image, image_info = process_image(file)
                if base64_image:
                    uploaded_images[file_name] = base64_image
                    all_text_chunks.append(f"Image: {file_name}\n[Image uploaded]")
                else:
                    all_text_chunks.append(image_info)
                
            else:
                all_text_chunks.append(f"Unsupported file type: {file_type} for file: {file_name}")
                
        except Exception as e:
            all_text_chunks.append(f"Error processing {file_name}: {str(e)}")
    
    # Extract keywords after processing files
    file_keywords = extract_file_keywords(all_text_chunks, file_names)
    return all_text_chunks, dataframes, file_keywords, uploaded_images

# Process files when uploaded
if uploaded_files:
    st.session_state.processed_text, st.session_state.dataframes, st.session_state.file_keywords, st.session_state.uploaded_images = process_files(uploaded_files)
    total_tokens = sum(estimate_tokens(chunk) for chunk in st.session_state.processed_text)
    if total_tokens > 5000:
        st.warning("Uploaded files may exceed Groq's token limits (12,000 TPM). Consider uploading smaller files or upgrading to a paid plan at https://console.groq.com/settings/billing.")

# Main chat interface
st.title("💬 Chat Assistant")
st.caption("🚀 A chatbot powered by Groq with OpenAI vision and fallback, capable of answering based on files, images, and general knowledge")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to fetch row from dataframe based on column value
def fetch_row(file_name, column_name, column_value):
    if file_name in st.session_state.dataframes:
        df = st.session_state.dataframes[file_name]
        try:
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

# Function to check if the prompt is related to uploaded files or images
def is_prompt_related_to_files_or_images(prompt, file_keywords, image_names):
    if not prompt:
        return False, False
    prompt_lower = prompt.lower()
    # Check for row fetch commands
    if "fetch row" in prompt_lower or "get row" in prompt_lower:
        return True, False
    
    # Check if prompt contains any file-related keywords
    is_file_related = any(keyword.lower() in prompt_lower for keyword in file_keywords)
    
    # Check if prompt references an image
    is_image_related = any(image_name.lower() in prompt_lower for image_name in image_names) or "image" in prompt_lower
    
    return is_file_related, is_image_related

# Chat input and processing
if prompt := st.chat_input("Ask me anything..."):
    if not prompt.strip():
        st.error("Please enter a valid question or command.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Initialize clients
    try:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing API clients: {str(e)}. Please check your API keys in secrets.")
        st.stop()

    # Check for row fetch
    row_fetch_response = ""
    relevant_file = None
    if "fetch row" in prompt.lower() or "get row" in prompt.lower():
        try:
            file_match = re.search(r'\b[\w\-]+\.(csv|xlsx)\b', prompt.lower())
            value_match = re.search(r'\bcust[\w\-]+\b', prompt.lower())
            column_name = "customer id"
            
            if file_match:
                relevant_file = file_match.group(0)
            if value_match:
                column_value = value_match.group(0).upper()
            
            if column_value and relevant_file:
                row_fetch_response = fetch_row(relevant_file, column_name, column_value)
            elif column_value:
                for fname in st.session_state.dataframes:
                    if fname.endswith(('.csv', '.xlsx')):
                        result = fetch_row(fname, column_name, column_value)
                        if not result.startswith(("No rows", "Column", "No dataframe", "Error")):
                            row_fetch_response = result
                            relevant_file = fname
                            break
                        row_fetch_response = result
            else:
                row_fetch_response = "Unable to process row fetch request. Please use format: 'fetch row for [column] [value] in [filename]'"
        except Exception as e:
            row_fetch_response = f"Error processing row fetch request: {str(e)}"

    # Prepare context
    context = f"""
    User Question: {prompt}
    
    Uploaded Files Context (chunked, limited to 1 chunk):
    {'\n\n'.join(st.session_state.processed_text[:1])}
    
    Row Fetch Result (if applicable): {row_fetch_response}
    """
    
    # Truncate context
    context, messages_to_send = truncate_context(context, st.session_state.messages, relevant_file, row_fetch_response)
    
    # Check if prompt is related to files or images
    is_file_related, is_image_related = is_prompt_related_to_files_or_images(
        prompt, st.session_state.file_keywords, st.session_state.uploaded_images.keys()
    )
    
    # Debug token usage
    total_tokens = estimate_tokens(context + ''.join(m['content'] for m in messages_to_send))
    st.write(f"Estimated tokens: {total_tokens}")
    
    try:
        if is_image_related:
            # Find referenced image
            referenced_image = None
            for image_name in st.session_state.uploaded_images:
                if image_name.lower() in prompt.lower():
                    referenced_image = image_name
                    break
            
            if referenced_image and referenced_image in st.session_state.uploaded_images:
                # Use OpenAI's vision model for image analysis
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{st.session_state.uploaded_images[referenced_image]}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.5,
                        max_tokens=3000
                    )
                    ai_response = response.choices[0].message.content
                except Exception as e:
                    ai_response = f"Error processing image with vision model: {str(e)}"
            else:
                ai_response = "No relevant image found. Please specify the image name or upload an image."
        else:
            # Use Groq for file-related or general knowledge questions
            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Answer based on provided file content, or general knowledge if no relevant files are provided."
                        }
                    ] + [
                        {"role": m["role"], "content": context + m["content"]}
                        for m in messages_to_send
                    ],
                    temperature=0.5,
                    max_tokens=3000
                )
                ai_response = response.choices[0].message.content
            except RateLimitError as e:
                if "429" in str(e):
                    st.warning("Groq rate limit reached. Switching to OpenAI model (gpt-3.5-turbo).")
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant. Answer based on provided file content, or general knowledge if no relevant files are provided."
                                }
                            ] + [
                                {"role": m["role"], "content": context + m["content"]}
                                for m in messages_to_send
                            ],
                            temperature=0.5,
                            max_tokens=3000
                        )
                        ai_response = response.choices[0].message.content
                    except Exception as e:
                        ai_response = f"Error with OpenAI fallback: {str(e)}"
                else:
                    raise e
            except Exception as e:
                ai_response = f"Error with Groq API: {str(e)}"
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(ai_response)
    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
