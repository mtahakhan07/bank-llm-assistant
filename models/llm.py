import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from utils.guardrails import filter_response, detect_out_of_domain
import logging
import streamlit as st
from langchain.memory import ConversationBufferMemory
from pydantic import Field, BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import tempfile
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFacePipeline

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Force UTF-8 encoding
)
logger = logging.getLogger(__name__)

# Filter out FAISS GPU messages
logging.getLogger('faiss').setLevel(logging.ERROR)

def convert_to_documents(data: Any) -> List[Document]:
    """
    Convert various data types to a list of Document objects.
    
    Args:
        data: Input data (DataFrame, list, or string)
        
    Returns:
        List of Document objects
    """
    try:
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to documents
            documents = []
            for _, row in data.iterrows():
                # Convert row to string and create document
                content = " ".join(str(value) for value in row.values if pd.notna(value))
                if content.strip():  # Only add non-empty documents
                    documents.append(Document(page_content=content))
            return documents
            
        elif isinstance(data, list):
            # Convert list to documents
            return [Document(page_content=str(item)) for item in data if str(item).strip()]
            
        elif isinstance(data, str):
            # Convert string to single document
            return [Document(page_content=data)] if data.strip() else []
            
        else:
            logger.warning(f"Unsupported data type: {type(data)}")
            return []
            
    except Exception as e:
        logger.error(f"Error converting data to documents: {str(e)}", exc_info=True)
        return []

class SimpleRetriever(BaseRetriever, BaseModel):
    """Simple retriever for document search using semantic similarity."""
    
    documents: List[Document] = Field(default_factory=list)
    encoder: Optional[SentenceTransformer] = Field(default=None)
    
    def __init__(self, documents: Optional[List[Document]] = None):
        """Initialize with optional documents."""
        # Convert input to Document objects if needed
        if documents is not None:
            documents = convert_to_documents(documents)
        super().__init__(documents=documents or [])
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Initialized SimpleRetriever with {len(self.documents)} documents")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return relevant documents for the query using semantic similarity."""
        try:
            if not self.documents:
                logger.warning("No documents available for retrieval")
                return []
            
            # Encode query and documents
            query_embedding = self.encoder.encode([query])[0]
            doc_embeddings = self.encoder.encode([doc.page_content for doc in self.documents])
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Get top 3 most similar documents
            top_k = min(3, len(self.documents))
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Log retrieval results
            logger.info(f"Retrieved {top_k} documents for query: {query}")
            for idx in top_indices:
                logger.debug(f"Document similarity: {similarities[idx]:.4f}")
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}", exc_info=True)
            return []

class StreamlitChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in Streamlit session state."""
    
    def __init__(self, key: str = "chat_history"):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = []
    
    @property
    def messages(self):
        return st.session_state[self.key]
    
    def add_user_message(self, message: str) -> None:
        st.session_state[self.key].append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        st.session_state[self.key].append(AIMessage(content=message))
    
    def clear(self) -> None:
        st.session_state[self.key] = []

class LLMModel:
    """
    Large Language Model for response generation using Hugging Face Transformers with LangChain.
    """
    
    def __init__(self):
        """
        Initialize the model using Hugging Face Transformers with LangChain.
        """
        self.model_name = "taha99/llama3-3b-instruct-bank-query-10k"
        
        try:
            logger.info("Initializing LLM model...")
            
            # Check for HuggingFace token
            if not os.getenv("HUGGINGFACE_API_TOKEN"):
                raise ValueError("HUGGINGFACE_API_TOKEN environment variable not set")
            
            # Initialize tokenizer and model
            logger.info("Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                token=os.getenv("HUGGINGFACE_API_TOKEN")
            )
            logger.info("Tokenizer loaded successfully")
            
            # Load model in CPU mode with memory optimizations
            logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                token=os.getenv("HUGGINGFACE_API_TOKEN")
            )
            logger.info("Model loaded successfully")
            
            # Create pipeline with optimized settings
            logger.info("Creating text generation pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                device_map="cpu",
                pad_token_id=tokenizer.eos_token_id
            )
            logger.info("Pipeline created successfully")
            
            # Initialize LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("LangChain wrapper initialized")
            
            # Initialize retriever with empty documents
            self.retriever = SimpleRetriever()
            logger.info("Retriever initialized")
            
            # Initialize memory with updated configuration
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            logger.info("Memory initialized")
            
            # Create prompt template with better instructions
            self.prompt = PromptTemplate(
                input_variables=["chat_history", "question", "context"],
                template="""You are a helpful and knowledgeable banking assistant for NUST Bank. Your goal is to provide accurate, helpful, and friendly responses to customer inquiries about banking products and services.

IMPORTANT GUIDELINES:
1. ONLY answer questions related to banking, finance, and NUST Bank's products/services
2. If the question is NOT about banking or finance, respond with: "I apologize, but I can only assist with banking-related questions. Please ask about NUST Bank's products, services, or banking operations."
3. Use the provided context to answer questions accurately
4. If you don't know the answer, say so politely
5. Keep responses concise but informative
6. Maintain a professional and friendly tone
7. Do not share sensitive information
8. Use examples when helpful
9. Break down complex information into digestible parts
10. Always prioritize customer security and privacy

Context: {context}

Chat History:
{chat_history}

Human: {question}
Assistant:"""
            )
            logger.info("Prompt template created")
            
            # Create chain with better configuration
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={
                    "prompt": self.prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=True,
                return_generated_question=True,
                max_tokens_limit=4000,
                verbose=True
            )
            logger.info("Chain created successfully")
            
            logger.info("Successfully initialized model via Transformers")
            
        except Exception as e:
            logger.error(f"Error during LLM initialization: {str(e)}", exc_info=True)
            raise

    def generate_response(self, query: str, context: Optional[List[Document]] = None, **kwargs) -> str:
        """Generate a response using the LLM model."""
        try:
            logger.info(f"Starting response generation for query: {query}")
            
            # Only update retriever if new context is provided
            if context and len(context) > 0:
                logger.info(f"Updating retriever with {len(context)} documents")
                # Convert context to Document objects if needed
                context = convert_to_documents(context)
                if context:  # Only update if we have valid documents
                    # Log the context being used
                    for i, doc in enumerate(context):
                        logger.info(f"Context {i}: {doc.page_content[:200]}...")
                    
                    self.retriever = SimpleRetriever(documents=context)
                    self.chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,
                        memory=self.memory,
                        combine_docs_chain_kwargs={
                            "prompt": self.prompt,
                            "document_variable_name": "context"
                        },
                        return_source_documents=True,
                        return_generated_question=True,
                        verbose=True
                    )
            
            logger.info("Preparing to invoke chain for response generation...")
            # Generate response using invoke instead of __call__
            response = self.chain.invoke({"question": query})
            logger.info(f"Raw chain response: {response}")
            
            if not response:
                logger.error("Empty response from chain")
                return "I apologize, but I was unable to generate a response. Please try again."
            
            if isinstance(response, dict):
                if "answer" in response:
                    answer = response["answer"]
                    # Log source documents if available
                    if "source_documents" in response:
                        logger.info("Source documents used:")
                        for i, doc in enumerate(response["source_documents"]):
                            logger.info(f"Source {i}: {doc.page_content[:200]}...")
                elif "text" in response:
                    answer = response["text"]
                else:
                    logger.error(f"Unexpected response format: {response}")
                    return "I apologize, but I was unable to generate a response. Please try again."
            else:
                answer = str(response)
            
            # Clean up the response
            answer = answer.strip()
            if "Assistant:" in answer:
                answer = answer.split("Assistant:")[-1].strip()
            
            # Filter response for sensitive information
            answer = filter_response(answer)
            
            logger.info(f"Successfully generated response: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model being used.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "api_type": "Hugging Face Transformers with LangChain",
            "framework": "LangChain"
        } 