import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle

class EmbeddingManager:
    """
    Manages the creation, storage, and retrieval of document embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        embedding_dim: int = 384, 
        index_path: str = "../data/faiss_index",
        docs_path: str = "../data/processed_data.json"
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            embedding_dim: Dimension of the embeddings
            index_path: Path to save/load the FAISS index
            docs_path: Path to the processed documents
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.docs_path = docs_path
        self.embeddings = None
        self.documents = None
        self.index = None
        self.model = None
        
    def load_model(self):
        """
        Load the sentence transformer model.
        """
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def load_documents(self, docs_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load the processed documents.
        
        Args:
            docs_path: Path to the processed documents JSON file
            
        Returns:
            List of document dictionaries
        """
        if docs_path is not None:
            self.docs_path = docs_path
            
        with open(self.docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        print(f"Loaded {len(self.documents)} documents from {self.docs_path}")
        return self.documents
    
    def generate_embeddings(self, force_recompute: bool = False) -> np.ndarray:
        """
        Generate embeddings for all documents.
        
        Args:
            force_recompute: Whether to force recomputation even if embeddings exist
            
        Returns:
            NumPy array of embeddings
        """
        # Check if we already have the documents loaded
        if self.documents is None:
            self.load_documents()
            
        # Check if we already have the model loaded
        if self.model is None:
            self.load_model()
        
        # Extract text from documents
        texts = []
        for doc in self.documents:
            if 'text' in doc:
                texts.append(doc['text'])
            elif 'content' in doc:
                texts.append(doc['content'])
            elif 'body' in doc:
                texts.append(doc['body'])
        
        print(f"Generating embeddings for {len(texts)} documents...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return self.embeddings
    
    def build_faiss_index(self, force_rebuild: bool = False) -> faiss.Index:
        """
        Build a FAISS index for efficient similarity search.
        
        Args:
            force_rebuild: Whether to force rebuilding the index
            
        Returns:
            FAISS index
        """
        # Check if we need to generate embeddings first
        if self.embeddings is None:
            self.generate_embeddings()
        
        print("Building FAISS index...")
        # Create a flat index (exact search)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add the embeddings to the index
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(self.embeddings, np.array(range(len(self.embeddings))))
        
        return self.index
    
    def save_index(self, index_path: Optional[str] = None) -> str:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            Path to the saved index
        """
        if index_path is not None:
            self.index_path = index_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save mapping of index to document
        with open(f"{self.index_path}_mapping.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Saved FAISS index to {self.index_path}.faiss")
        return self.index_path
    
    def load_index(self, index_path: Optional[str] = None) -> faiss.Index:
        """
        Load the FAISS index from disk.
        
        Args:
            index_path: Path to the saved index
            
        Returns:
            FAISS index
        """
        if index_path is not None:
            self.index_path = index_path
            
        # Load the index
        self.index = faiss.read_index(f"{self.index_path}.faiss")
        
        # Load mapping of index to document
        with open(f"{self.index_path}_mapping.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Loaded FAISS index from {self.index_path}.faiss")
        return self.index
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        # Check if we have the model loaded
        if self.model is None:
            self.load_model()
            
        # Check if we have the index loaded
        if self.index is None:
            try:
                self.load_index()
            except:
                print("Index not found, building a new one...")
                self.load_documents()
                self.generate_embeddings()
                self.build_faiss_index()
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Valid index
                doc = self.documents[idx].copy()
                doc['score'] = float(1.0 - distances[0][i])  # Convert distance to similarity score
                results.append(doc)
        
        return results


if __name__ == "__main__":
    # Example usage
    embedding_manager = EmbeddingManager()
    
    try:
        # Try to load existing index
        embedding_manager.load_index()
    except:
        print("Building new embeddings and index...")
        embedding_manager.load_documents()
        embedding_manager.generate_embeddings()
        embedding_manager.build_faiss_index()
        embedding_manager.save_index()
    
    # Test search
    results = embedding_manager.search("How do I check my account balance?")
    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result.get('text', '')[:100]}...") 