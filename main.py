import streamlit as st
import os
from utils.data_processor import preprocess_excel, add_document_to_index
from utils.vector_store import VectorStore
from models.llm import LLMModel
from utils.guardrails import apply_guardrails

# Page configuration
st.set_page_config(
    page_title="NUST Bank Assistant",
    page_icon="🏦",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = LLMModel()
    
# App title
st.title("NUST Bank Assistant 🏦")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Add Knowledge")
    uploaded_file = st.file_uploader("Upload new document", type=['xlsx', 'csv', 'json', 'txt'])
    
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
                st.success(f"Document {uploaded_file.name} added successfully!")
            else:
                st.error("Failed to process document. Please check the format.")
    
    st.divider()
    st.header("Settings")
    st.caption("Adjust how the assistant responds to your questions")
    
    # Settings options
    response_length = st.slider("Response Length", min_value=100, max_value=500, value=250, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about NUST Bank products and services..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Apply guardrails to user input
    if not apply_guardrails(prompt):
        response = "I apologize, but I cannot respond to that query as it appears to be requesting sensitive information or contains inappropriate content. Please ask about NUST Bank's products and services instead."
    else:
        # Get response from LLM
        with st.spinner("Thinking..."):
            # Get relevant context from vector store
            context = st.session_state.vector_store.search(prompt, top_k=3)
            
            # Generate response using the LLM
            response = st.session_state.llm_model.generate_response(
                prompt, 
                context, 
                max_length=response_length, 
                temperature=temperature
            )
    
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Initial data loading (if not already done)
if not st.session_state.vector_store.is_initialized():
    with st.spinner("Initializing knowledge base..."):
        excel_path = "NUST Bank-Product-Knowledge.xlsx"
        if os.path.exists(excel_path):
            chunks = preprocess_excel(excel_path)
            if chunks:
                st.session_state.vector_store.add_texts(chunks)
                st.success("Knowledge base initialized successfully!")
            else:
                st.error("Failed to preprocess the Excel file. Please check the format.")
        else:
            st.error("Knowledge base file not found. Please upload a file to start.") 