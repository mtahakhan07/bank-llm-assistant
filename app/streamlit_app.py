import os
import sys
import json
import streamlit as st
import logging
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import BankDataPreprocessor
from utils.embeddings import EmbeddingManager
from models.llm import LlamaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Bank LLM Assistant",
    page_icon="💰",
    layout="wide",
)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'xlsx', 'xls'}

# Initialize components
@st.cache_resource
def load_embedding_manager():
    return EmbeddingManager(
        index_path=os.path.join(DATA_DIR, 'faiss_index'),
        docs_path=os.path.join(DATA_DIR, 'processed_data.json')
    )

@st.cache_resource
def load_llm():
    # Check if we should use mock mode from environment variable
    use_mock = os.getenv('USE_MOCK_MODEL', 'True').lower() == 'true'
    load_8bit = os.getenv('LOAD_8BIT', 'False').lower() == 'true'
    
    # Get model name from environment
    model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-3.2-3B-Instruct')
    
    # Create and return LLM instance
    return LlamaModel(
        model_name=model_name,
        load_in_8bit=load_8bit,
        mock_mode=use_mock
    )

def check_and_load_index(embedding_manager):
    try:
        embedding_manager.load_index()
        st.session_state.index_loaded = True
        return True
    except Exception as e:
        st.session_state.index_loaded = False
        logger.error(f"Error loading index: {str(e)}")
        return False

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'index_loaded' not in st.session_state:
        st.session_state.index_loaded = False
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'documents_count' not in st.session_state:
        st.session_state.documents_count = 0

def create_sidebar():
    st.sidebar.title("Bank LLM Assistant")
    st.sidebar.markdown("An AI-powered banking assistant using Llama-3.2-3B")
    
    # Create section for uploading documents
    st.sidebar.header("Upload New Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload bank documents (JSON, CSV, TXT)",
        type=list(ALLOWED_EXTENSIONS)
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Process Document"):
            process_uploaded_file(uploaded_file)
    
    # System status
    st.sidebar.header("System Status")
    
    # Documents status
    if st.session_state.documents_loaded:
        st.sidebar.success(f"📄 {st.session_state.documents_count} documents loaded")
    else:
        st.sidebar.warning("📄 No documents loaded")
    
    # Index status
    if st.session_state.index_loaded:
        st.sidebar.success("🔍 Vector index ready")
    else:
        st.sidebar.warning("🔍 Vector index not loaded")
        
    # Model status
    st.sidebar.success("🤖 Llama-3.2-3B model available")
    
    # Add information about guard rails
    st.sidebar.header("Security Notes")
    st.sidebar.info(
        "This assistant implements guard rails to prevent:\n"
        "- Sharing sensitive personal information\n"
        "- Responding to harmful requests\n"
        "- Providing instructions for unauthorized activities"
    )
    
    # Add credits
    st.sidebar.markdown("---")
    st.sidebar.caption("CS416: Large Language Models (BESE-12)")

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and update embeddings"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process the file
        with st.spinner("Processing document..."):
            preprocessor = BankDataPreprocessor(data_dir=DATA_DIR)
            preprocessor.load_data(tmp_file_path)
            processed_chunks = preprocessor.chunk_documents()
            
            # Update the processed data
            processed_data_path = os.path.join(DATA_DIR, 'processed_data.json')
            try:
                # Try to load existing processed data
                if os.path.exists(processed_data_path):
                    with open(processed_data_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Append new data
                    existing_data.extend(processed_chunks)
                    
                    # Save updated data
                    with open(processed_data_path, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                else:
                    # Create new processed data file
                    with open(processed_data_path, 'w') as f:
                        json.dump(processed_chunks, f, indent=2)
            
            except Exception as e:
                # Create new processed data file
                with open(processed_data_path, 'w') as f:
                    json.dump(processed_chunks, f, indent=2)
            
            # Update embeddings
            embedding_manager = load_embedding_manager()
            embedding_manager.load_documents()
            st.session_state.documents_count = len(embedding_manager.documents)
            st.session_state.documents_loaded = True
            
            embedding_manager.generate_embeddings()
            embedding_manager.build_faiss_index()
            embedding_manager.save_index()
            st.session_state.index_loaded = True
            
            st.sidebar.success(f"✅ Added {len(processed_chunks)} document chunks")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        logger.error(f"Error processing file: {str(e)}")

def create_main_interface():
    st.title("Bank LLM Assistant")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if documents are loaded
    if not st.session_state.documents_loaded and not st.session_state.index_loaded:
        st.warning(
            "⚠️ No bank documents loaded. Please upload documents through the sidebar "
            "to start using the assistant."
        )
    
    # Get user input
    user_question = st.chat_input("Ask a question about banking services...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if we have documents
                    embedding_manager = load_embedding_manager()
                    
                    if not check_and_load_index(embedding_manager):
                        response = "I don't have any bank information loaded yet. Please upload some documents first."
                    else:
                        # Get relevant context from vector search
                        context_results = embedding_manager.search(user_question, k=3)
                        
                        # Get answer from Llama
                        llm = load_llm()
                        response = llm.generate_answer(user_question, context_results)
                        
                        # Show sources if we have context
                        if context_results and response:
                            response += "\n\n---\n\n**Sources:**\n"
                            for i, doc in enumerate(context_results[:3], 1):
                                snippet = doc.get('text', '')[:100] + "..."
                                response += f"{i}. {snippet}\n"
                except Exception as e:
                    response = f"I encountered an error: {str(e)}"
                    logger.error(f"Error generating response: {str(e)}")
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})


def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Create main interface
    create_main_interface()
    
    # Try to load embeddings if they exist
    if not st.session_state.index_loaded or not st.session_state.documents_loaded:
        embedding_manager = load_embedding_manager()
        try:
            embedding_manager.load_index()
            st.session_state.index_loaded = True
            st.session_state.documents_loaded = True
            st.session_state.documents_count = len(embedding_manager.documents)
        except:
            try:
                embedding_manager.load_documents()
                st.session_state.documents_loaded = True
                st.session_state.documents_count = len(embedding_manager.documents)
            except:
                pass

if __name__ == "__main__":
    main() 