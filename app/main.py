# Import initialization first
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from app import streamlit_init

# Import Streamlit and set page config
import streamlit as st
st.set_page_config(
    page_title="NUST Bank Assistant",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other dependencies
import pandas as pd
from models.llm import LLMModel
from utils.vector_store import VectorStore
from utils.data_processor import DataProcessor
from utils.guardrails import filter_response, detect_out_of_domain, validate_query, log_interaction
from typing import List, Dict, Any
import time
from datetime import datetime
import os
from pathlib import Path

# Import logging
import logging
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_queries': 0,
        'successful_responses': 0,
        'failed_responses': 0,
        'start_time': datetime.now()
    }

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header Styles */
    h1 {
        color: #1e293b;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        color: #334155;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #475569;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Chat Message Styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 85%;
    }
    
    .chat-message.user {
        background-color: #2563eb;
        color: white;
        margin-left: auto;
    }
    
    .chat-message.assistant {
        background-color: white;
        border: 1px solid #e2e8f0;
        margin-right: auto;
    }
    
    .chat-message .role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .chat-message .content {
        line-height: 1.6;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        border-radius: 0.75rem;
        height: 3.5em;
        background-color: #2563eb;
        color: white;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8fafc;
        padding: 2rem 1rem;
    }
    
    /* Upload Section Styles */
    .upload-section {
        padding: 2rem;
        border: 2px dashed #cbd5e1;
        border-radius: 1rem;
        margin: 1.5rem 0;
        background-color: white;
        transition: all 0.2s ease;
    }
    
    .upload-section:hover {
        border-color: #2563eb;
        background-color: #f8fafc;
    }
    
    /* Stats Card Styles */
    .stats-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .stats-card p {
        margin: 0.5rem 0;
        color: #475569;
    }
    
    .stats-card strong {
        color: #1e293b;
    }
    
    /* Input Area Styles */
    .stTextInput>div>div>input {
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.1);
    }
    
    /* Success/Error Message Styles */
    .stSuccess {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #86efac;
    }
    
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_components():
    """Initialize LLM model and vector store."""
    if st.session_state.llm_model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.llm_model = LLMModel()
    
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            st.session_state.vector_store = VectorStore()

def clear_cache():
    """Clear all cached data and reset components."""
    try:
        # Clear vector store
        if 'vector_store' in st.session_state:
            st.session_state.vector_store.reset_index()
            del st.session_state.vector_store
        
        # Reset session state variables
        for key in ['processed_docs', 'chat_history', 'current_doc', 'current_doc_name', 'last_uploaded_file']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Delete cache directory and its contents
        cache_dir = Path("cache")
        if cache_dir.exists():
            import shutil
            try:
                # First try to remove all files in the directory
                for file in cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                # Then remove the directory itself
                cache_dir.rmdir()
                print(f"Deleted cache directory: {cache_dir}")
            except Exception as e:
                print(f"Error deleting cache directory: {str(e)}")
                # If normal deletion fails, try force deletion
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print("Force deleted cache directory")
                except Exception as e2:
                    print(f"Force deletion also failed: {str(e2)}")
        
        # Clean up any temporary files
        for temp_file in Path(".").glob("temp_*"):
            try:
                temp_file.unlink()
                print(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                print(f"Error deleting temporary file {temp_file}: {str(e)}")
        
        # Reinitialize components
        initialize_components()
        
        st.success("Knowledge base cleared successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing cache: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")

def process_uploaded_file(file_path):
    """Process uploaded document and update vector store."""
    if file_path is not None:
        with st.spinner("Processing document..."):
            try:
                # Handle both uploaded files and existing files
                if hasattr(file_path, 'getvalue'):  # It's an uploaded file
                    # Save uploaded file temporarily
                    temp_path = f"temp_{file_path.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file_path.getvalue())
                    file_to_process = temp_path
                else:  # It's an existing file path
                    file_to_process = str(file_path)
                
                # Process document
                processor = DataProcessor()
                processed_data = processor.process_document(file_to_process)
                
                if not processed_data:
                    st.error("No valid content found in the document.")
                    return False
                
                # Convert processed data to Document objects
                from langchain_core.documents import Document
                documents = [Document(page_content=chunk) for chunk in processed_data]
                
                # Create metadata
                metadata = [{
                    'source': str(file_path.name if hasattr(file_path, 'name') else file_path),
                    'upload_time': datetime.now().isoformat(),
                    'content_type': file_path.type if hasattr(file_path, 'type') else 'file',
                    'size': file_path.size if hasattr(file_path, 'size') else os.path.getsize(file_to_process)
                } for _ in documents]
                
                # Update vector store
                st.session_state.vector_store.add_documents(documents, metadata)
                
                # Clean up temporary file if it was created
                if hasattr(file_path, 'getvalue'):
                    os.remove(file_to_process)
                
                st.success(f"Document processed successfully! Added {len(documents)} chunks to the knowledge base.")
                return True
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
                return False

def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
            <div class="chat-message user">
                <div class="role">You</div>
                <div class="content">{content}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message assistant">
                <div class="role">Assistant</div>
                <div class="content">{content}</div>
            </div>
        """, unsafe_allow_html=True)

def display_stats():
    """Display system statistics."""
    with st.sidebar:
        st.markdown("### System Statistics")
        st.markdown("""
            <div class="stats-card">
                <p><strong>Total Queries:</strong> {}</p>
                <p><strong>Success Rate:</strong> {:.1f}%</p>
                <p><strong>Session Duration:</strong> {}</p>
            </div>
        """.format(
            st.session_state.stats['total_queries'],
            (st.session_state.stats['successful_responses'] / max(1, st.session_state.stats['total_queries'])) * 100,
            str(datetime.now() - st.session_state.stats['start_time']).split('.')[0]
        ), unsafe_allow_html=True)

def main():
    # Initialize components
    initialize_components()
    
    # Title and description
    st.title("NUST Bank Assistant")
    st.markdown("""
    <div style="background-color: white; padding: 2rem; border-radius: 1rem; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
        <h3 style="color: #1e293b; margin-bottom: 1rem;">Welcome to the NUST Bank Assistant! 🏦</h3>
        <p style="color: #475569; line-height: 1.6;">I can help you with:</p>
        <ul style="color: #475569; line-height: 1.6;">
            <li>Banking products and services</li>
            <li>Account information</li>
            <li>Transaction details</li>
            <li>General banking queries</li>
        </ul>
        <p style="color: #475569; line-height: 1.6;">Feel free to ask any banking-related questions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Document Upload")
        st.markdown("""
            <div class="upload-section">
                <p style="color: #475569; margin-bottom: 1rem;">Upload documents to enhance my knowledge about banking products and services.</p>
                <p style="color: #64748b; font-size: 0.9rem;">Supported formats: Excel, CSV, JSON, and Text files.</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['xlsx', 'csv', 'json', 'txt'],
            key="file_uploader"
        )
        
        # Only process file if it's newly uploaded
        if uploaded_file and ('last_uploaded_file' not in st.session_state or 
                            st.session_state.last_uploaded_file != uploaded_file.name):
            if process_uploaded_file(uploaded_file):
                st.session_state.last_uploaded_file = uploaded_file.name
        
        # Add Clear Knowledge Base button
        if st.button("Clear Knowledge Base", type="secondary"):
            clear_cache()
            st.session_state.last_uploaded_file = None
        
        # Display statistics
        display_stats()
    
    # Chat interface
    st.subheader("Chat")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"])
    
    # Chat input with form
    with st.form(key=f"chat_form_{len(st.session_state.chat_history)}", clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key=f"user_input_{len(st.session_state.chat_history)}")
        submit = st.form_submit_button("Send")
    
    if submit and user_input:
        logger.info(f"Processing new query: {user_input}")
        # Update statistics
        st.session_state.stats['total_queries'] += 1
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate response with enhanced loading animation
        with st.spinner("🤔 Thinking..."):
            try:
                # Validate query
                logger.info("Validating query...")
                validation = validate_query(user_input)
                logger.info(f"Query validation result: {validation}")
                
                if not validation['is_valid']:
                    logger.warning("Query validation failed")
                    response = "I apologize, but I cannot respond to that query as it appears to be requesting sensitive information or contains inappropriate content. Please ask about NUST Bank's products and services instead."
                    st.session_state.stats['failed_responses'] += 1
                else:
                    # Get relevant context if documents are available
                    context = None
                    if st.session_state.vector_store and st.session_state.vector_store.is_initialized():
                        logger.info("Searching for relevant context...")
                        with st.spinner("🔍 Searching for relevant information..."):
                            context = st.session_state.vector_store.search(user_input)
                            logger.info(f"Found {len(context) if context else 0} relevant documents")
                            if context:
                                for i, doc in enumerate(context):
                                    logger.info(f"Context {i}: {doc['text'][:200]}... (score: {doc['score']:.4f})")
                    
                    # Generate response using LLM's knowledge
                    logger.info("Generating response from LLM...")
                    with st.spinner("💭 Generating response..."):
                        try:
                            answer = st.session_state.llm_model.generate_response(
                                query=user_input,
                                context=context,
                                max_length=512,
                                temperature=0.7
                            )
                            logger.info(f"Raw LLM response: {answer}")
                            
                            # Parse only the assistant's reply
                            if "Assistant:" in answer:
                                response = answer.split("Assistant:")[-1].strip()
                            else:
                                response = answer.strip()
                            
                            logger.info(f"Final processed response: {response}")
                            st.session_state.stats['successful_responses'] += 1
                        except Exception as e:
                            logger.error(f"Error in LLM response generation: {str(e)}", exc_info=True)
                            response = "I apologize, but I encountered an error while generating the response. Please try again."
                            st.session_state.stats['failed_responses'] += 1
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Log interaction
                log_interaction(user_input, response, validation)
                
            except Exception as e:
                logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
                response = "I apologize, but I encountered an error while processing your request. Please try again."
                st.session_state.stats['failed_responses'] += 1
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the chat display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>NUST Bank Assistant v1.0 | Powered by Llama 3.2 3B Instruct</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 