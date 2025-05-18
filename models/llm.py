import os
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from utils.guardrails import filter_response, detect_out_of_domain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import login
from transformers import BitsAndBytesConfig

class LLMModel:
    """
    Large Language Model for response generation using Llama 3.2 3B Instruct.
    """
    
    def __init__(self):
        """
        Initialize the Llama 3.2 3B Instruct model with optimizations.
        """
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Check if CUDA is available and set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the model and tokenizer with optimizations
        try:
            print(f"Loading model: {self.model_name}...")
            
            # Initialize tokenizer with padding
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True
            )
            
            # Set default padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for better memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                use_cache=True,
                low_cpu_mem_usage=True
            )
            
            # Create optimized pipeline with chat template
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                device_map="auto" if self.device == "cuda" else None,
                batch_size=1,
                return_full_text=False
            )
            
            # Create LangChain pipeline
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            # Initialize conversation memory with improved context handling
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            print(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load Llama 3.2 3B Instruct model: {str(e)}")
    
    def generate_response(self, query: str, context: List[str], max_length: int = 250, temperature: float = 0.7) -> str:
        """
        Generate a response based on the query and context with improved prompt handling.
        
        Args:
            query: The user's query
            context: List of relevant context chunks
            max_length: Maximum length of the generated response
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Generated response
        """
        # Check if query is out of domain
        if detect_out_of_domain(query):
            return "I'm sorry, but your question appears to be outside the domain of banking and financial services. I'm specifically trained to help with questions about NUST Bank's products and services. How can I assist you with your banking needs today?"
        
        try:
            # Create an improved prompt template using LLaMA chat format
            messages = [
                {"role": "system", "content": """You are a helpful, accurate, and friendly bank assistant for NUST Bank. 
                Your role is to provide accurate information about banking products and services.
                
                Guidelines:
                1. Answer based ONLY on the provided context
                2. If information is not in the context, politely say so
                3. Keep responses clear and concise
                4. Maintain a professional and helpful tone
                5. Use bullet points for multiple items
                6. Format numbers and percentages appropriately"""},
                {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"}
            ]
            
            # Format messages for the model
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response with improved parameters
            response = self.llm(
                formatted_prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2
            )
            
            # Update memory with improved context
            self.memory.save_context(
                {"input": query, "context": " ".join(context)},
                {"output": response}
            )
            
            # Apply response filtering with improved handling
            filtered_response = filter_response(response)
            
            return filtered_response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I am currently experiencing technical difficulties. Please try again later."
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1_000_000,  # In millions
            "optimizations": {
                "half_precision": self.model.dtype == torch.float16,
                "quantization": hasattr(self.model, "quantization_config") and self.model.quantization_config is not None
            }
        } 