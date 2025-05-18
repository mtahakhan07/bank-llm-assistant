"""
Preprocess the NUST Bank knowledge base and build a FAISS index.
Run this script to initialize the system before starting the main application.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import preprocess_excel
from utils.vector_store import VectorStore

def main():
    """Main preprocessing function."""
    print("Starting preprocessing...")
    
    # Check if the Excel file exists
    excel_path = "NUST Bank-Product-Knowledge.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return False
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Reset existing index if it exists
    if os.path.exists("data/faiss_index.faiss"):
        print("Removing existing index...")
        try:
            os.remove("data/faiss_index.faiss")
        except Exception as e:
            print(f"Error removing index: {str(e)}")
    
    if os.path.exists("data/faiss_index_mapping.pkl"):
        print("Removing existing mapping...")
        try:
            os.remove("data/faiss_index_mapping.pkl")
        except Exception as e:
            print(f"Error removing mapping: {str(e)}")
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore()
    
    # Process the Excel file
    print(f"Processing {excel_path}...")
    try:
        chunks = preprocess_excel(excel_path)
        
        if not chunks:
            print("Error: Failed to extract content from the Excel file.")
            return False
        
        print(f"Extracted {len(chunks)} text chunks.")
        
        # Add chunks to the vector store
        print("Building vector index...")
        vector_store.add_texts(chunks)
        
        # Verify the index was created
        if vector_store.is_initialized():
            print("Preprocessing completed successfully!")
            print(f"Index saved to {vector_store.index_path}")
            print(f"Mapping saved to {vector_store.mapping_path}")
            return True
        else:
            print("Error: Failed to initialize vector store.")
            return False
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 