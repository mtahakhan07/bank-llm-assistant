# Bank LLM Assistant Architecture

The Bank LLM Assistant follows a retrieval-augmented generation (RAG) architecture to provide accurate and context-aware responses to customer inquiries. This approach combines the power of a large language model with a vector-based retrieval system, allowing the assistant to leverage both general language understanding and domain-specific knowledge.

## High-Level Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│                 │     │                   │     │                    │
│  Data Ingestion │────▶│  Vector Database  │────▶│  Response Generation│
│                 │     │                   │     │                    │
└─────────────────┘     └───────────────────┘     └────────────────────┘
        │                         ▲                         │
        │                         │                         │
        │                         │                         │
        │                         │                         │
        ▼                         │                         ▼
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│                 │     │                   │     │                    │
│  Preprocessing  │     │  Query Processing │     │    Guard Rails     │
│                 │     │                   │     │                    │
└─────────────────┘     └───────────────────┘     └────────────────────┘
                                 ▲                         │
                                 │                         │
                                 │                         │
                                 │                         │
                                 └─────────────────────────┘
                                      User Interface
```

## Components

### 1. Data Ingestion & Preprocessing
- **File Format Handling**: Supports JSON, CSV, and plain text files
- **Sanitization**: Removes or masks personally identifiable information (PII)
- **Chunking**: Divides documents into smaller, meaningful chunks for embedding
- **Text Normalization**: Handles lowercasing, special character removal, etc.

### 2. Vector Database
- **Embedding Generation**: Uses sentence-transformers to create dense vector embeddings
- **FAISS Index**: Efficient similarity search with Facebook AI Similarity Search (FAISS)
- **Document Storage**: Maintains mapping between embeddings and document chunks

### 3. Query Processing
- **User Question Embedding**: Converts user questions into the same embedding space
- **Semantic Search**: Identifies the most relevant document chunks using cosine similarity
- **Context Compilation**: Assembles retrieved context for the language model

### 4. Response Generation
- **Llama-3.2-3B-Instruct Model**: Meta's 3 billion parameter LLM for text generation
- **Prompt Engineering**: Structured prompts to guide the model's response generation
- **Context Integration**: Incorporates retrieved information into the generation process

### 5. Guard Rails
- **Input Filtering**: Detects and blocks sensitive or inappropriate questions
- **Output Filtering**: Prevents disclosure of sensitive information
- **Safety Measures**: Refuses to provide information on harmful topics

### 6. User Interface (Streamlit)
- **Chat Interface**: Real-time conversation with the assistant
- **Document Upload**: Allows users to add new documents to the knowledge base
- **System Status**: Provides visibility into the state of various components

## Data Flow

1. **Document Processing Flow**:
   - User uploads a document → Preprocessing → Chunking → Vector Embedding → FAISS Index
   
2. **Query Processing Flow**:
   - User asks a question → Query Embedding → Vector Search → Context Retrieval → Prompt Construction → LLM Generation → Guard Rails → Response Display

## Key Technical Decisions

1. **Model Selection**: Llama-3.2-3B-Instruct offers a good balance between performance and resource requirements
2. **Embedding Model**: Sentence-transformers provides efficient and effective text embeddings
3. **Vector Index**: FAISS enables fast similarity search even with a large number of documents
4. **Retrieval Strategy**: Top-k most similar document chunks are retrieved for context
5. **Streamlit Frontend**: Provides an interactive, user-friendly interface for the assistant 