import os
import json
import pandas as pd
import re
from typing import List, Dict, Any, Union, Tuple

class BankDataPreprocessor:
    """
    Preprocessor for bank customer service data.
    Handles data loading, cleaning, and preparation for embedding.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the bank dataset
        """
        self.data_dir = data_dir
        self.documents = []
        
    def load_data(self, file_path: str = None) -> List[Dict[str, Any]]:
        """
        Load data from JSON, CSV, or TXT files.
        
        Args:
            file_path: Path to the data file (if None, will search data_dir)
            
        Returns:
            List of documents as dictionaries
        """
        if file_path is None:
            # Search for data files in the data directory
            for file in os.listdir(self.data_dir):
                if file.endswith(('.json', '.csv', '.txt')):
                    file_path = os.path.join(self.data_dir, file)
                    break
        
        if file_path is None:
            raise FileNotFoundError("No data file found in the data directory")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.documents = data
            elif isinstance(data, dict):
                if 'documents' in data:
                    self.documents = data['documents']
                else:
                    # Convert flat dict to list of dicts
                    self.documents = [data]
        
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            self.documents = df.to_dict('records')
            
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Simple parsing: each document is separated by blank lines
            current_doc = ""
            for line in lines:
                if line.strip():
                    current_doc += line
                elif current_doc:
                    self.documents.append({"text": current_doc.strip()})
                    current_doc = ""
            
            # Add the last document if it exists
            if current_doc:
                self.documents.append({"text": current_doc.strip()})
        
        print(f"Loaded {len(self.documents)} documents from {file_path}")
        return self.documents
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,?!:;-]', '', text)
        
        return text.strip()
    
    def sanitize_pii(self, text: str) -> str:
        """
        Remove or mask personally identifiable information (PII).
        
        Args:
            text: Text containing potential PII
            
        Returns:
            Text with PII removed or masked
        """
        # Replace email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        
        # Replace phone numbers (various formats)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        # Replace credit card numbers
        text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]', text)
        
        # Replace account numbers (assuming 8-12 digits)
        text = re.sub(r'\b\d{8,12}\b', '[ACCOUNT_NUMBER]', text)
        
        # Replace SSN/ID numbers
        text = re.sub(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]', text)
        
        # Replace names (more complex, would need NER in production)
        # This is a simple approximation
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        return text
    
    def chunk_documents(self, max_chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks suitable for embedding.
        
        Args:
            max_chunk_size: Maximum number of tokens per chunk
            
        Returns:
            List of document chunks
        """
        chunked_docs = []
        
        for doc in self.documents:
            # Get the text field or the first available text-like field
            text = None
            if 'text' in doc:
                text = doc['text']
            elif 'content' in doc:
                text = doc['content']
            elif 'body' in doc:
                text = doc['body']
            else:
                # Find the first string value that could be text
                for key, value in doc.items():
                    if isinstance(value, str) and len(value) > 50:
                        text = value
                        break
            
            if not text:
                continue
                
            # Clean and sanitize the text
            text = self.clean_text(text)
            text = self.sanitize_pii(text)
            
            # Split into sentences (simple approach)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Create chunks of sentences that fit within max_chunk_size
            current_chunk = ""
            for sentence in sentences:
                # Approximate token count (words + punctuation)
                sentence_tokens = len(re.findall(r'\b\w+\b', sentence)) + len(re.findall(r'[.,!?;:]', sentence))
                
                if len(current_chunk) == 0:
                    current_chunk = sentence
                elif len(re.findall(r'\b\w+\b', current_chunk)) + sentence_tokens <= max_chunk_size:
                    current_chunk += " " + sentence
                else:
                    # Store the current chunk and start a new one
                    chunk_doc = doc.copy()
                    chunk_doc['text'] = current_chunk
                    chunked_docs.append(chunk_doc)
                    current_chunk = sentence
            
            # Add the last chunk if it exists
            if current_chunk:
                chunk_doc = doc.copy()
                chunk_doc['text'] = current_chunk
                chunked_docs.append(chunk_doc)
                
        print(f"Created {len(chunked_docs)} chunks from {len(self.documents)} documents")
        return chunked_docs
    
    def save_processed_data(self, output_path: str = None) -> str:
        """
        Save the processed documents to a file.
        
        Args:
            output_path: Path to save the processed data
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, "processed_data.json")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
        print(f"Saved processed data to {output_path}")
        return output_path
    
    def process_pipeline(self, file_path: str = None, max_chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Run the full preprocessing pipeline.
        
        Args:
            file_path: Path to the input data file
            max_chunk_size: Maximum chunk size for document splitting
            
        Returns:
            List of processed document chunks
        """
        self.load_data(file_path)
        processed_chunks = self.chunk_documents(max_chunk_size)
        return processed_chunks


if __name__ == "__main__":
    # Example usage
    preprocessor = BankDataPreprocessor()
    try:
        processed_data = preprocessor.process_pipeline()
        preprocessor.save_processed_data()
        print("Data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}") 