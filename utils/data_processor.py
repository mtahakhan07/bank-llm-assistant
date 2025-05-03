import pandas as pd
import os
import json
import re
from typing import List, Dict, Any, Optional

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