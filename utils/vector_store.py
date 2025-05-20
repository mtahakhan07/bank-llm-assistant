import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
from langchain.schema import Document
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for efficient similarity search using FAISS.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Store documents and metadata
        self.documents = []
        self.metadata = []
        
        # Cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Batch size for processing
        self.batch_size = 32
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing index and documents from cache if available."""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            index_path = self.cache_dir / "faiss_index.bin"
            docs_path = self.cache_dir / "documents.pkl"
            meta_path = self.cache_dir / "metadata.pkl"
            
            if index_path.exists() and docs_path.exists() and meta_path.exists():
                self.index = faiss.read_index(str(index_path))
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from cache")
            else:
                # Initialize empty index if no cache exists
                self.index = faiss.IndexFlatIP(self.dimension)
                self.documents = []
                self.metadata = []
                
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            # Initialize empty index on error
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.metadata = []
    
    def _save_index(self):
        """Save index and documents to cache."""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = self.cache_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents and metadata
            docs_path = self.cache_dir / "documents.pkl"
            meta_path = self.cache_dir / "metadata.pkl"
            
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            
            print(f"Saved {len(self.documents)} documents to cache")
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
    
    def add_documents(self, documents: List[Document], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            metadata: Optional list of metadata dictionaries for each document
        """
        if not documents:
            return
        
        try:
            # Extract text content from Document objects
            texts = [doc.page_content for doc in documents]
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Generate embeddings
                embeddings = self.model.encode(batch, show_progress_bar=True)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.index.add(np.array(embeddings).astype('float32'))
                
                # Store documents and metadata
                self.documents.extend(batch)
                if metadata:
                    self.metadata.extend(metadata[i:i + self.batch_size])
                else:
                    self.metadata.extend([{} for _ in batch])
            
            # Save to cache
            self._save_index()
            print(f"Successfully added {len(documents)} documents to the vector store")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
    
    def search(self, query: str, k: int = 3, score_threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing document text, metadata, and similarity score
        """
        if not self.documents:
            logger.warning("No documents available for search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(
                np.array(query_embedding).astype('float32'),
                min(k, len(self.documents))
            )
            
            # Return relevant documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= score_threshold:
                    # Log the similarity score for debugging
                    logger.info(f"Document {idx} similarity score: {score:.4f}")
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(score)
                    })
            
            if not results:
                logger.warning(f"No documents found above threshold {score_threshold}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}", exc_info=True)
            return []
    
    def clear(self):
        """Clear all documents and reset the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        self._save_index()

    def is_initialized(self) -> bool:
        """
        Check if the vector store is initialized with data.
        
        Returns:
            True if the vector store contains embeddings, False otherwise
        """
        return hasattr(self, 'documents') and len(self.documents) > 0
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dictionaries for each text
        """
        if not texts:
            return
            
        try:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Generate embeddings for the batch
                embeddings = self.model.encode(batch, convert_to_numpy=True)
                
                # Normalize embeddings
                faiss.normalize_L2(embeddings)
                
                # Add embeddings to the index
                self.index.add(embeddings.astype(np.float32))
                
                # Update the text mapping and metadata
                self.documents.extend(batch)
                if metadata:
                    self.metadata.extend(metadata[i:i + self.batch_size])
                else:
                    self.metadata.extend([{} for _ in batch])
            
            # Save to cache
            self._save_index()
        except Exception as e:
            print(f"Error adding texts to index: {str(e)}")
    
    def reset_index(self) -> None:
        """
        Reset the vector store by creating a new index.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        
        # Delete existing files if they exist
        try:
            if self.cache_dir.exists():
                for file in self.cache_dir.glob("*"):
                    file.unlink()
        except Exception as e:
            print(f"Error resetting index: {str(e)}")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:", key="user_input")
    submit = st.form_submit_button("Send")
if submit and user_input:
    # process the input as before
    ... 