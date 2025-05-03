import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    A vector store for text similarity search using FAISS.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", 
                 index_path: str = "data/faiss_index.faiss",
                 mapping_path: str = "data/faiss_index_mapping.pkl"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
            index_path: Path where the FAISS index is stored
            mapping_path: Path where the mapping of index to text is stored
        """
        # Load the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Paths for storing the index and mapping
        self.index_path = index_path
        self.mapping_path = mapping_path
        
        # Initialize or load the FAISS index and mapping
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            self.load_index()
        else:
            # Create a new simple flat index (not IDMap)
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.text_mapping = []
    
    def is_initialized(self) -> bool:
        """
        Check if the vector store is initialized with data.
        
        Returns:
            True if the vector store contains embeddings, False otherwise
        """
        return hasattr(self, 'text_mapping') and len(self.text_mapping) > 0
    
    def add_texts(self, texts: List[str]) -> None:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
        """
        if not texts:
            return
            
        try:
            # Generate embeddings for the texts
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Make sure embeddings are in the right format (float32)
            embeddings = embeddings.astype(np.float32)
            
            # Add embeddings to the index
            self.index.add(embeddings)
            
            # Update the text mapping
            for text in texts:
                self.text_mapping.append(text)
            
            # Save the updated index
            self.save_index()
        except Exception as e:
            print(f"Error adding texts to index: {str(e)}")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search for similar texts to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of similar texts
        """
        # Check if index is empty
        if not hasattr(self, 'index') or self.index.ntotal == 0:
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Make sure embeddings are in the right format (float32)
            query_embedding = query_embedding.astype(np.float32)
            
            # Search the index
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)
            
            # Get the corresponding texts
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.text_mapping):
                    results.append(self.text_mapping[idx])
            
            return results
        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []
    
    def save_index(self) -> None:
        """
        Save the FAISS index and text mapping to disk.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save the FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save the text mapping
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(self.text_mapping, f)
        except Exception as e:
            print(f"Error saving index: {str(e)}")
    
    def load_index(self) -> None:
        """
        Load the FAISS index and text mapping from disk.
        """
        try:
            # Load the FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load the text mapping
            with open(self.mapping_path, 'rb') as f:
                self.text_mapping = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            # Create a new index
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.text_mapping = []
    
    def reset_index(self) -> None:
        """
        Reset the vector store by creating a new index.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.text_mapping = []
        
        # Delete existing files if they exist
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.mapping_path):
                os.remove(self.mapping_path)
        except Exception as e:
            print(f"Error resetting index: {str(e)}") 