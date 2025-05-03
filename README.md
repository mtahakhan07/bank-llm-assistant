# NUST Bank LLM Assistant

This project implements a Large Language Model (LLM)-based solution to enhance customer service for NUST Bank. The system uses a retrieval-augmented generation (RAG) approach to answer customer queries accurately using a knowledge base of banking products and services.

## Features

- **Data Ingestion & Preprocessing**: Handles Excel, CSV, JSON, and plaintext files with automatic sanitization.
- **Embedding & Indexing**: Creates vector embeddings for efficient semantic search using FAISS.
- **LLM Integration**: Uses Llama-3.2-3B-Instruct for high-quality response generation.
- **Guard Rails**: Implements safety measures to prevent harmful content and protect sensitive information.
- **Real-Time Updates**: Allows adding new documents to the knowledge base through the UI.
- **Streamlit Interface**: Provides an intuitive chat interface for customers.

## Architecture

The system follows a retrieval-augmented generation (RAG) architecture:

1. **Data Processing**: Documents are preprocessed, chunked, and embedded.
2. **Vector Storage**: FAISS is used for efficient similarity search.
3. **Query Processing**: User questions are converted to embeddings and matched with relevant context.
4. **Response Generation**: The LLM generates responses based on retrieved context.
5. **Guard Rails**: Safety filters prevent harmful content and protect sensitive data.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bank-llm-assistant.git
   cd bank-llm-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to http://localhost:8501

## Usage

- Ask questions about NUST Bank products and services in the chat interface.
- Upload new documents via the sidebar to expand the knowledge base.
- Adjust response settings like length and temperature in the sidebar.

## Project Structure

```
bank-llm-assistant/
├── app/                    # Application components
├── data/                   # Data storage (FAISS indices)
├── models/                 # LLM model implementation
│   └── llm.py              # LLM model class
├── utils/                  # Utility functions
│   ├── data_processor.py   # Data preprocessing utilities
│   ├── guardrails.py       # Safety filters
│   └── vector_store.py     # FAISS vector store implementation
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Limitations

- The system is designed for a specific domain (banking) and may not perform well on out-of-domain queries.
- The LLM model requires significant computational resources, especially on CPU-only environments.
- Answers are limited to the information contained in the knowledge base.

## Future Improvements

- Multi-language support for diverse customer base
- Integration with bank's authentication system for personalized responses
- Enhanced context retrieval with hybrid search methods
- Support for additional document formats (PDF, DOC, etc.)

## License

This project is licensed for educational purposes only and is not intended for production use.

## Contributors

- Your Name
- Team Members 