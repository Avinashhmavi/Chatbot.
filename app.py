import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "file_dataframes" not in st.session_state:
    st.session_state.file_dataframes = {}

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

# Process uploaded files
def process_files(uploaded_files):
    all_text = ""
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
            
            all_text += f"\n\nFile: {file_name}\n{text}"
        
        except Exception as e:
            all_text += f"\n\nError processing {file_name}: {str(e)}"
    
    return all_text

# Process files when uploaded
if uploaded_files:
    st.session_state.processed_text = process_files(uploaded_files)

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
st.title("💬 Chat Assistant")
st.caption("🚀 A chatbot powered by OpenAI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Ask me anything..."):
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OPENAI_API_KEY not found in Streamlit secrets")
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
        
        # Prepare context with file content and chat history
        context = f"""
        Uploaded Files Content:
        {st.session_state.processed_text}
        
        Chat History:
        {chr(10).join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])}
        
        Current Question: {prompt}
        
        Answer the question based on the provided file content and chat history. If the information is not available, use your general knowledge.
        """
        
        # Prepare messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately."
            },
            {"role": "user", "content": context}
        ]
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
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
