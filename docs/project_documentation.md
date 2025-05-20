# NUST Bank LLM Assistant Documentation

## 1. Introduction
### 1.1 Project Overview
The NUST Bank LLM Assistant is a sophisticated banking chatbot that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to customer inquiries. The system combines the power of a fine-tuned language model with efficient document retrieval to deliver banking-specific information while maintaining security and privacy.

## 2. System Architecture
### 2.1 Data Ingestion & Preprocessing
#### 2.1.1 Document Processing Pipeline
- Supports multiple file formats: Excel, CSV, JSON, and Text files
- Implements text cleaning and normalization
- Handles document chunking for optimal context retrieval
- Performs PII detection and anonymization
- Location: `utils/data_processor.py`

### 2.2 Large Language Model Selection
#### 2.2.1 Model Specifications
- **Model**: Llama 3.2 3B Instruct (Bank Query Fine-tuned)
- **Model Name**: `taha99/llama3-3b-instruct-bank-query-10k`

#### 2.2.2 Model Justification
- Optimized for banking domain queries
- 3B parameters for efficient CPU deployment
- Fine-tuned on banking-specific data
- Supports instruction following
- Location: `models/llm.py`

### 2.3 Embedding & Indexing
#### 2.3.1 Vector Store Implementation
- Uses FAISS for efficient similarity search
- Sentence Transformer: `all-MiniLM-L6-v2`
- Implements cosine similarity for document retrieval
- Supports batch processing for large datasets
- Location: `utils/vector_store.py`

### 2.4 Model Fine-Tuning & Inference
#### 2.4.1 RAG Implementation
- Combines document retrieval with LLM generation
- Uses ConversationalRetrievalChain for context-aware responses
- Implements memory for conversation history
- Location: `models/llm.py`

### 2.5 Prompt Engineering
#### 2.5.1 Custom Prompt Template
- Banking-specific instructions
- Domain restriction enforcement
- Context integration guidelines
- Professional tone maintenance
- Location: `models/llm.py`

### 2.6 Real-Time Updates
#### 2.6.1 Document Management
- Supports dynamic document addition
- Automatic reindexing of new content
- Cache management for performance
- Location: `utils/vector_store.py`

### 2.7 Performance and Reliability
#### 2.7.1 Optimizations
- Batch processing for embeddings
- CPU-optimized model loading
- Efficient document chunking
- Caching mechanism for faster retrieval
- Location: `utils/vector_store.py`, `models/llm.py`

### 2.8 Version Control
#### 2.8.1 Git Implementation
- Private GitHub repository
- Regular commits for feature development
- Branch management for collaboration
- Documentation updates

### 2.9 System Interface
#### 2.9.1 Streamlit Web Application
- Clean, modern UI design
- Real-time chat interface
- Document upload functionality
- Progress indicators and loading states
- Location: `app/main.py`

### 2.10 Guard Rails Implementation
#### 2.10.1 Security Features
- Sensitive data detection
- Domain validation
- Inappropriate content filtering
- Security pattern detection
- Location: `utils/guardrails.py`

## 3. Technical Implementation
### 3.1 Data Processing
```python
class DataProcessor:
    def process_document(self, file_path):
        # Handles multiple file formats
        # Implements text cleaning
        # Performs chunking
        # Returns processed documents
```

### 3.2 Vector Store
```python
class VectorStore:
    def __init__(self):
        # Initializes FAISS index
        # Sets up sentence transformer
        # Manages document storage

    def search(self, query):
        # Performs semantic search
        # Returns relevant documents
```

### 3.3 LLM Model
```python
class LLMModel:
    def __init__(self):
        # Loads fine-tuned model
        # Sets up retrieval chain
        # Initializes memory

    def generate_response(self, query, context):
        # Generates context-aware responses
        # Handles conversation history
```

### 3.4 Guard Rails
```python
class GuardRails:
    def filter_response(self, response):
        # Filters sensitive information
        # Validates content

    def detect_out_of_domain(self, query):
        # Checks domain relevance
        # Enforces banking focus
```

## 4. Installation and Setup
### 4.1 Environment Setup
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Configuration
```powershell
# Set HuggingFace token
$env:HUGGINGFACE_API_TOKEN = "your_token_here"
```

### 4.3 Running the Application
```powershell
streamlit run app/main.py
```

## 5. User Guide
### 5.1 Starting the Application
- Run the Streamlit app
- Access through web browser
- Upload initial documents

### 5.2 Adding New Documents
- Use the sidebar upload interface
- Supported formats: Excel, CSV, JSON, Text
- Automatic processing and indexing

### 5.3 Asking Questions
- Type questions in the chat interface
- Receive context-aware responses
- View conversation history

### 5.4 Managing Knowledge Base
- Clear knowledge base when needed
- Upload new documents
- Monitor system statistics

## 6. Security Implementation
### 6.1 Content Filtering
- PII detection
- Sensitive data masking
- Inappropriate content blocking

### 6.2 Domain Enforcement
- Banking-specific responses
- Out-of-domain detection
- Query validation

### 6.3 Access Control
- Environment variable protection
- Secure token handling
- Error logging

## 7. Performance Optimization
### 7.1 System Optimizations
- CPU-optimized model loading
- Efficient document chunking
- Batch processing
- Caching mechanism

### 7.2 Scalability Features
- Modular architecture
- Efficient memory usage
- Batch processing support

## 8. Future Development
### 8.1 Planned Enhancements
- GPU support for faster processing
- Enhanced document preprocessing
- Improved context retrieval
- Advanced security features

### 8.2 Potential Additions
- User authentication
- Conversation export
- Analytics dashboard
- API integration

## 9. Troubleshooting Guide
### 9.1 Common Issues
- Model loading errors
- Document processing issues
- Response generation problems

### 9.2 Solutions
- Check environment variables
- Verify document formats
- Monitor system logs
- Clear cache when needed 