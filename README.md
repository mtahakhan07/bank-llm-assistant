# NUST Bank LLM Assistant

A banking assistant powered by a fine-tuned LLaMA 3.2 3B Instruct model, designed to provide accurate and secure responses to banking-related queries.

## Features

- 🤖 Fine-tuned LLaMA 3.2 3B Instruct model for banking domain
- 📚 Document processing and knowledge base management
- 🔒 Built-in guardrails for data privacy and security
- 💬 Interactive chat interface with Streamlit
- 📊 Support for multiple document formats (PDF, Excel, CSV, TXT)
- 🔍 Efficient vector search using FAISS
- 📝 Comprehensive logging and audit trails

## Project Structure

```
bank-llm-assistant/
├── app/
│   └── main.py              # Streamlit application
├── models/
│   ├── llm.py              # LLM model implementation
│   └── fine_tuning.ipynb   # Model fine-tuning notebook
├── utils/
│   ├── data_processor.py   # Document processing
│   ├── vector_store.py     # Vector search implementation
│   └── guardrails.py       # Security and privacy checks
├── cache/                  # Vector store cache
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

4. Run the application:
   ```bash
   streamlit run app/main.py
   ```

## Usage

1. **Starting the Application**
   - Run `streamlit run app/main.py`
   - Access the web interface at `http://localhost:8501`

2. **Uploading Documents**
   - Use the sidebar to upload banking documents
   - Supported formats: PDF, Excel, CSV, TXT
   - Documents are automatically processed and indexed

3. **Chat Interface**
   - Type your banking-related questions in the chat input
   - The assistant will provide relevant responses based on the uploaded documents
   - Responses are filtered for sensitive information

4. **Security Features**
   - Automatic detection of sensitive data
   - Domain-specific response filtering
   - Comprehensive audit logging

## Model Details

- Base Model: LLaMA 3.2 3B Instruct
- Fine-tuned on banking domain data
- Optimized for memory efficiency
- Supports context-aware responses

## Development

### Adding New Features

1. **Document Processing**
   - Add new file type support in `utils/data_processor.py`
   - Implement custom processing logic

2. **Guard Rails**
   - Extend patterns in `utils/guardrails.py`
   - Add new validation rules

3. **Vector Store**
   - Modify indexing strategy in `utils/vector_store.py`
   - Adjust chunk size and overlap

### Fine-tuning

1. Prepare your dataset
2. Use the `models/fine_tuning.ipynb` notebook
3. Follow the instructions in the notebook
4. Upload the fine-tuned model to Hugging Face

## Security Considerations

- All sensitive data is automatically detected and filtered
- Responses are validated against banking domain
- Comprehensive logging for audit trails
- No sensitive data is stored in the vector store

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for the LLaMA model
- Hugging Face for the Transformers library
- Streamlit for the web interface framework 