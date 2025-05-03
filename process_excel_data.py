import os
import sys
import json
import logging
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import BankDataPreprocessor
from utils.embeddings import EmbeddingManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_excel_data(file_path):
    """
    Process the bank product knowledge Excel file.
    """
    logger.info(f"Loading Excel file from {file_path}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        logger.info(f"Excel file loaded successfully with {len(df)} rows")
        
        # Convert DataFrame to list of documents
        processed_docs = []
        
        # Iterate through each row
        for index, row in df.iterrows():
            # Convert row to dictionary and handle NaN values
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = ""
            
            # Create document with text suitable for embedding
            # Adjust the column names based on your Excel structure
            text = ""
            for key, value in row_dict.items():
                if value:
                    text += f"{key}: {value}\n"
            
            if text:
                doc = {
                    'text': text,
                    'source': 'excel_product_knowledge',
                    'row_index': index,
                    **row_dict  # Include all original columns
                }
                processed_docs.append(doc)
        
        logger.info(f"Processed {len(processed_docs)} documents from Excel file")
        return processed_docs
    
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        raise

def main():
    """
    Process the Excel file and create embeddings.
    """
    logger.info("Starting processing of Excel bank data")
    
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    excel_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NUST Bank-Product-Knowledge.xlsx')
    processed_data_path = os.path.join(data_dir, 'processed_data.json')
    
    # Check if Excel file exists
    if not os.path.exists(excel_file_path):
        logger.error(f"Excel file not found at {excel_file_path}")
        print(f"Error: Excel file not found at {excel_file_path}")
        return
    
    try:
        # Process the Excel data
        processed_excel_docs = process_excel_data(excel_file_path)
        
        # Load existing processed data if available
        existing_docs = []
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'r', encoding='utf-8') as f:
                existing_docs = json.load(f)
                logger.info(f"Loaded {len(existing_docs)} existing documents")
        
        # Combine existing and new documents
        combined_docs = existing_docs + processed_excel_docs
        logger.info(f"Combined total of {len(combined_docs)} documents")
        
        # Save processed data
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(combined_docs, f, indent=2)
        
        logger.info(f"Saved combined processed data to {processed_data_path}")
        
        # Generate embeddings and build index
        logger.info("Initializing embedding manager")
        embedding_manager = EmbeddingManager(
            index_path=os.path.join(data_dir, 'faiss_index'),
            docs_path=processed_data_path
        )
        
        logger.info("Loading documents for embedding")
        embedding_manager.load_documents()
        
        logger.info("Generating embeddings")
        embedding_manager.generate_embeddings()
        
        logger.info("Building FAISS index")
        embedding_manager.build_faiss_index()
        
        logger.info("Saving index")
        embedding_manager.save_index()
        
        logger.info("Embedding and indexing complete")
        print("Excel data processing, embedding and indexing complete. The system is ready to use.")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 