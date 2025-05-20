import pandas as pd
import os
import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import tempfile

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processor for handling document processing and anonymization.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Regular expressions for sensitive data
        self.patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'account': r'\b\d{4}[-]?\d{4}[-]?\d{4}[-]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'address': r'\b\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way)\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        }
        
        # Text cleaning patterns
        self.cleaning_patterns = {
            'urls': r'https?://\S+|www\.\S+',
            'special_chars': r'[^\w\s.,!?-]',
            'extra_spaces': r'\s+',
            'html_tags': r'<[^>]+>'
        }
        
        self.processed_files = set()  # Track processed files
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
    
    def process_document(self, file_path: str) -> List[str]:
        """
        Process a document and return a list of text chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of text chunks
        """
        try:
            # Check if file has already been processed
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                logger.info(f"File {file_path} already processed, skipping")
                return []
            
            # Get file extension
            ext = Path(file_path).suffix.lower()
            
            # Process based on file type
            if ext in ['.xlsx', '.xls']:
                chunks = self._process_excel(file_path)
            elif ext == '.csv':
                chunks = self._process_csv(file_path)
            elif ext == '.json':
                chunks = self._process_json(file_path)
            elif ext == '.txt':
                chunks = self._process_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return []
            
            # Mark file as processed
            self.processed_files.add(file_hash)
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return []
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to track processed files."""
        try:
            stat = os.stat(file_path)
            return f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        except:
            return file_path
    
    def _process_excel(self, file_path: str) -> List[str]:
        """Process Excel file into text chunks."""
        try:
            logger.info(f"Processing Excel file: {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Convert to text chunks
            chunks = []
            for _, row in df.iterrows():
                # Convert row to string, handling NaN values
                text = " ".join(str(val) for val in row.values if pd.notna(val))
                if text.strip():
                    chunks.append(text)
            
            logger.info(f"Extracted {len(chunks)} chunks from Excel file")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {str(e)}")
            return []
    
    def _process_csv(self, file_path: str) -> List[str]:
        """Process CSV file into text chunks."""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            for _, row in df.iterrows():
                text = " ".join(str(val) for val in row.values if pd.notna(val))
                if text.strip():
                    chunks.append(text)
            return chunks
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            return []
    
    def _process_json(self, file_path: str) -> List[str]:
        """Process JSON file into text chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._flatten_json(data)
        except Exception as e:
            logger.error(f"Error processing JSON {file_path}: {str(e)}")
            return []
    
    def _process_text(self, file_path: str) -> List[str]:
        """Process text file into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Split into paragraphs
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
            return chunks
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def _flatten_json(self, data: Any, prefix: str = "") -> List[str]:
        """Flatten JSON data into text chunks."""
        chunks = []
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                chunks.extend(self._flatten_json(value, new_prefix))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                chunks.extend(self._flatten_json(item, new_prefix))
        else:
            if data is not None:
                chunks.append(f"{prefix}: {data}")
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(self.cleaning_patterns['urls'], '', text)
        
        # Remove HTML tags
        text = re.sub(self.cleaning_patterns['html_tags'], '', text)
        
        # Remove special characters
        text = re.sub(self.cleaning_patterns['special_chars'], ' ', text)
        
        # Remove extra spaces
        text = re.sub(self.cleaning_patterns['extra_spaces'], ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Split into words
        tokens = text.split()
        
        # Remove stopwords (optional)
        # tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text: Text to split
            chunk_size: Target size of each chunk
            
        Returns:
            List of text chunks
        """
        # Clean and anonymize text
        text = self.clean_text(text)
        text = self.anonymize_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive information in text.
        
        Args:
            text: Text to anonymize
            
        Returns:
            Anonymized text
        """
        # Replace sensitive patterns with hashed values
        for pattern_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                sensitive_data = match.group()
                # Create a deterministic hash
                hash_value = hashlib.sha256(sensitive_data.encode()).hexdigest()[:8]
                replacement = f"[{pattern_type.upper()}_{hash_value}]"
                text = text.replace(sensitive_data, replacement)
        
        return text
    
    def validate_data(self, data: List[str]) -> bool:
        """
        Validate processed data.
        
        Args:
            data: List of processed text chunks
            
        Returns:
            True if data is valid, False otherwise
        """
        if not data:
            return False
        
        # Check for minimum content
        total_length = sum(len(chunk) for chunk in data)
        if total_length < 100:  # Minimum 100 characters
            return False
        
        # Check for sensitive data
        for chunk in data:
            for pattern in self.patterns.values():
                if re.search(pattern, chunk):
                    self.logger.warning("Sensitive data found in processed chunks")
                    return False
        
        return True

def preprocess_excel(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    Preprocesses an Excel file and returns a list of text chunks.
    
    Args:
        file_path: Path to the Excel file
        chunk_size: The target size of each text chunk in characters
        
    Returns:
        A list of text chunks suitable for embedding
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names and replace NaN values
        df = df.fillna("")
        
        # Extract information from the dataframe
        chunks = []
        
        # Process each row as a potential product or service
        for _, row in df.iterrows():
            # Skip completely empty rows
            if all(not str(cell).strip() for cell in row):
                continue
                
            # Convert the row to a string representation
            row_text = ""
            for col, value in row.items():
                if str(value).strip():
                    # Check if the column has a meaningful name
                    col_name = str(col).strip()
                    if col_name.startswith("Unnamed"):
                        # Just add the value without the column name
                        row_text += f"{value}\n"
                    else:
                        # Add both column name and value
                        row_text += f"{col}: {value}\n"
            
            # Only add non-empty row texts
            if row_text.strip():
                # Apply sanitization
                row_text = sanitize_text(row_text)
                
                # Add as a chunk
                chunks.append(row_text)
        
        # For larger texts, split into smaller chunks
        result_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split into smaller chunks
                for i in range(0, len(chunk), chunk_size):
                    sub_chunk = chunk[i:i+chunk_size]
                    if sub_chunk.strip():
                        result_chunks.append(sub_chunk)
            else:
                if chunk.strip():
                    result_chunks.append(chunk)
        
        return result_chunks
    
    except Exception as e:
        print(f"Error preprocessing Excel file: {str(e)}")
        return []

def preprocess_csv(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    Preprocesses a CSV file and returns a list of text chunks.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: The target size of each text chunk in characters
        
    Returns:
        A list of text chunks suitable for embedding
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names and replace NaN values
        df = df.fillna("")
        
        # Extract information from the dataframe
        chunks = []
        
        # Process each row as a potential product or service
        for _, row in df.iterrows():
            # Skip completely empty rows
            if all(not str(cell).strip() for cell in row):
                continue
                
            # Convert the row to a string representation
            row_text = ""
            for col, value in row.items():
                if str(value).strip():
                    # Check if the column has a meaningful name
                    col_name = str(col).strip()
                    if col_name.startswith("Unnamed"):
                        # Just add the value without the column name
                        row_text += f"{value}\n"
                    else:
                        # Add both column name and value
                        row_text += f"{col}: {value}\n"
            
            # Only add non-empty row texts
            if row_text.strip():
                # Apply sanitization
                row_text = sanitize_text(row_text)
                
                # Add as a chunk
                chunks.append(row_text)
        
        # For larger texts, split into smaller chunks
        result_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split into smaller chunks
                for i in range(0, len(chunk), chunk_size):
                    sub_chunk = chunk[i:i+chunk_size]
                    if sub_chunk.strip():
                        result_chunks.append(sub_chunk)
            else:
                if chunk.strip():
                    result_chunks.append(chunk)
        
        return result_chunks
    
    except Exception as e:
        print(f"Error preprocessing CSV file: {str(e)}")
        return []

def preprocess_json(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    Preprocesses a JSON file and returns a list of text chunks.
    
    Args:
        file_path: Path to the JSON file
        chunk_size: The target size of each text chunk in characters
        
    Returns:
        A list of text chunks suitable for embedding
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = []
        
        # Process the JSON data based on its structure
        if isinstance(data, list):
            # If it's a list of records
            for item in data:
                if isinstance(item, dict):
                    # Convert each record to a string
                    item_text = "\n".join([f"{k}: {v}" for k, v in item.items() if v])
                    if item_text.strip():
                        chunks.append(sanitize_text(item_text))
        elif isinstance(data, dict):
            # If it's a single record or nested structure
            def process_dict(d, prefix=""):
                text = ""
                for k, v in d.items():
                    if isinstance(v, dict):
                        text += process_dict(v, f"{prefix}{k}.")
                    elif isinstance(v, list):
                        if all(isinstance(i, dict) for i in v):
                            for i, item in enumerate(v):
                                text += process_dict(item, f"{prefix}{k}[{i}].")
                        else:
                            text += f"{prefix}{k}: {v}\n"
                    else:
                        text += f"{prefix}{k}: {v}\n"
                return text
            
            text = process_dict(data)
            if text.strip():
                chunks.append(sanitize_text(text))
        
        # Split large chunks
        result_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split into smaller chunks
                for i in range(0, len(chunk), chunk_size):
                    sub_chunk = chunk[i:i+chunk_size]
                    if sub_chunk.strip():
                        result_chunks.append(sub_chunk)
            else:
                if chunk.strip():
                    result_chunks.append(chunk)
        
        return result_chunks
    
    except Exception as e:
        print(f"Error preprocessing JSON file: {str(e)}")
        return []

def preprocess_text(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    Preprocesses a text file and returns a list of text chunks.
    
    Args:
        file_path: Path to the text file
        chunk_size: The target size of each text chunk in characters
        
    Returns:
        A list of text chunks suitable for embedding
    """
    try:
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Sanitize the text
        text = sanitize_text(text)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        print(f"Error preprocessing text file: {str(e)}")
        return []

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by removing sensitive information.
    
    Args:
        text: The text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove potential PII (credit card numbers, SSNs, etc.)
    # Credit card pattern: 16 digits, may have spaces or dashes
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]', text)
    
    # SSN pattern: 9 digits, may have dashes
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Email pattern
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Phone number patterns
    text = re.sub(r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    
    # Basic text normalization
    # Convert to lowercase
    text = text.lower()
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def add_document_to_index(file_path: str, vector_store) -> bool:
    """
    Processes a document and adds it to the vector store.
    
    Args:
        file_path: Path to the document
        vector_store: The vector store instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        chunks = []
        if file_extension == '.xlsx' or file_extension == '.xls':
            chunks = preprocess_excel(file_path)
        elif file_extension == '.csv':
            chunks = preprocess_csv(file_path)
        elif file_extension == '.json':
            chunks = preprocess_json(file_path)
        elif file_extension == '.txt':
            chunks = preprocess_text(file_path)
        else:
            print(f"Unsupported file format: {file_extension}")
            return False
        
        if not chunks:
            print(f"No valid chunks found in the document: {file_path}")
            return False
        
        # Add chunks to the vector store
        vector_store.add_texts(chunks)
        
        return True
    
    except Exception as e:
        print(f"Error adding document to index: {str(e)}")
        return False 