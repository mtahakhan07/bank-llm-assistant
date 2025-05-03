# Bank LLM Assistant

A Large Language Model (LLM)-based solution to enhance customer service for a local bank. This project uses advanced language models to create a responsive AI-driven assistant capable of handling customer inquiries accurately with context-aware responses.

## Project Structure

```
bank-llm-assistant/
├── app/                 # Streamlit web application
├── data/                # Dataset storage
├── models/              # Model scripts and fine-tuning code
├── utils/               # Utility functions
├── requirements.txt     # Required packages
├── .env                 # Environment configuration (create this file)
└── README.md            # Project overview
```

## Quick Start

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with the following content:
   ```
   # Model configuration
   MODEL_NAME=EleutherAI/gpt-neo-1.3B
   USE_MOCK_MODEL=True
   LOAD_8BIT=False
   
   # Hugging Face token for model access
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```
4. Process the sample data:
   ```
   python preprocess_sample_data.py
   ```
5. Start the application in mock mode:
   ```
   streamlit run app/streamlit_app.py
   ```
6. Access the web interface at http://localhost:8501

## Using LLM Models

This project supports various language models from Hugging Face, with a default configuration that balances performance and resource requirements.

### Setting Up Hugging Face Token

To use any actual model (not mock mode), you must:

1. Register/login at [huggingface.co](https://huggingface.co)
2. Go to Profile → Settings → Access Tokens
3. Create a new token with "read" access
4. Copy the token to your `.env` file:
   ```
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
   ```

### Choosing the Right Model

The project now defaults to **GPT-Neo 1.3B** which offers:
- Faster loading times
- Lower memory requirements
- Compatible with most systems
- No quantization required

You can still use **GPT-J 6B** if you have sufficient resources:
- Requires 16GB+ RAM or 8-bit quantization
- Higher quality responses
- Significantly larger resource footprint

### Running with Actual Models

Use the `run_with_model.py` script for the easiest setup:

```bash
# Run with the smaller GPT-Neo 1.3B model (recommended)
python run_with_model.py --small

# Run with the model specified in your .env file
python run_with_model.py

# Specify any Hugging Face model
python run_with_model.py --model=EleutherAI/gpt-neo-2.7B
```

### Mock Mode vs. Real Model

By default, the system runs in "mock mode" which simulates responses without loading any large model. This is ideal for testing and development.

To use a real model:
1. Use `run_with_model.py` with appropriate flags
2. Ensure your Hugging Face token is configured correctly
3. Be patient during initial model loading

## Features

- Data ingestion and preprocessing
- Vector embeddings for efficient retrieval
- Context-aware responses using advanced language models
- Basic guard rails implementation
- Streamlit-based chat interface for asking questions and uploading new documents

## Architecture

[Architecture Diagram](architecture.png)

## Troubleshooting

### Common Issues

1. **bitsandbytes errors with --load-8bit**:
   - This requires a separate installation: `pip install bitsandbytes`
   - Some Windows systems have compatibility issues with bitsandbytes
   - Use `--small` flag instead for more compatible models

2. **Out of memory errors**:
   - Use the smaller model: `python run_with_model.py --small` 
   - Close other memory-intensive applications
   - Try a different model with `--model` flag

3. **Authentication errors**:
   - Verify your Hugging Face token is correct
   - Check that you have permission to access the model
   - Ensure your internet connection is stable

## Note

This project is part of CS416: Large Language Models course (BESE-12) at the School of Electrical Engineering and Computer Science, National University of Sciences and Technology (NUST). 