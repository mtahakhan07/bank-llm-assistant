import streamlit as st
import os
from utils.data_processor import preprocess_excel, add_document_to_index
from utils.vector_store import VectorStore
from models.llm import LLMModel
from utils.guardrails import apply_guardrails
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="NUST Bank Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = LLMModel()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_queries': 0,
        'successful_responses': 0,
        'failed_responses': 0,
        'start_time': datetime.now()
    }

# App title and description
st.title("NUST Bank Assistant 🏦")
st.markdown("""
    Welcome to the NUST Bank Assistant! I'm here to help you with information about our products and services.
    Feel free to ask questions about our banking solutions, account types, or any other banking-related queries.
""")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("📚 Knowledge Management")
    
    # Document upload section
    with st.expander("Upload New Document", expanded=True):
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv', 'json', 'txt'])
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                # Save the uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                os.makedirs("uploads", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file and add to the index
                success = add_document_to_index(file_path, st.session_state.vector_store)
                
                if success:
                    st.success(f"✅ Document {uploaded_file.name} added successfully!")
                else:
                    st.error("❌ Failed to process document. Please check the format.")
    
    st.divider()
    
    # Settings section
    st.header("⚙️ Settings")
    with st.expander("Response Settings", expanded=True):
        response_length = st.slider(
            "Response Length",
            min_value=100,
            max_value=500,
            value=250,
            step=50,
            help="Control the length of the assistant's responses"
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make responses more creative, lower values make them more focused"
        )
    
    st.divider()
    
    # Statistics section
    st.header("📊 Statistics")
    with st.expander("Usage Statistics", expanded=True):
        st.metric("Total Queries", st.session_state.stats['total_queries'])
        st.metric("Success Rate", f"{(st.session_state.stats['successful_responses'] / max(1, st.session_state.stats['total_queries'])) * 100:.1f}%")
        st.metric("Session Duration", str(datetime.now() - st.session_state.stats['start_time']).split('.')[0])

# Main chat interface
st.markdown("### 💬 Chat Interface")

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with improved styling
if prompt := st.chat_input("Ask me about NUST Bank products and services..."):
    # Update statistics
    st.session_state.stats['total_queries'] += 1
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Apply guardrails to user input
    if not apply_guardrails(prompt):
        response = "I apologize, but I cannot respond to that query as it appears to be requesting sensitive information or contains inappropriate content. Please ask about NUST Bank's products and services instead."
        st.session_state.stats['failed_responses'] += 1
    else:
        # Get response from LLM
        with st.spinner("🤔 Thinking..."):
            try:
                # Get relevant context from vector store
                context = st.session_state.vector_store.search(prompt, top_k=3)
                
                # Generate response using the LLM
                response = st.session_state.llm_model.generate_response(
                    prompt, 
                    context, 
                    max_length=response_length, 
                    temperature=temperature
                )
                st.session_state.stats['successful_responses'] += 1
            except Exception as e:
                response = "I apologize, but I encountered an error while processing your request. Please try again."
                st.session_state.stats['failed_responses'] += 1
                print(f"Error: {str(e)}")
    
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Initial data loading (if not already done)
if not st.session_state.vector_store.is_initialized():
    with st.spinner("📚 Initializing knowledge base..."):
        excel_path = "NUST Bank-Product-Knowledge.xlsx"
        if os.path.exists(excel_path):
            chunks = preprocess_excel(excel_path)
            if chunks:
                st.session_state.vector_store.add_texts(chunks)
                st.success("✅ Knowledge base initialized successfully!")
            else:
                st.error("❌ Failed to preprocess the Excel file. Please check the format.")
        else:
            st.error("❌ Knowledge base file not found. Please upload a file to start.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>NUST Bank Assistant v1.0 | Powered by Llama 3.2 3B Instruct</p>
    </div>
""", unsafe_allow_html=True) 