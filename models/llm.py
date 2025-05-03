import os
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from utils.guardrails import filter_response, detect_out_of_domain

class LLMModel:
    """
    Large Language Model for response generation.
    """
    
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the model and tokenizer
        try:
            print(f"Loading model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # For models that need trust_remote_code (like Llama-3.2-3B-Instruct)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move model to device if CPU
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            print("Falling back to a more available model...")
            
            # List of fallback models to try
            fallback_models = [
                "google/gemma-2b-it", 
                "stabilityai/stablelm-3b-4e1t", 
                "microsoft/phi-2",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ]
            
            for fallback_model in fallback_models:
                try:
                    print(f"Trying {fallback_model}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                    
                    # Move model to device if CPU
                    if self.device == "cpu":
                        self.model = self.model.to(self.device)
                        
                    self.model_name = fallback_model
                    print(f"Successfully loaded {fallback_model}")
                    break
                except Exception as e2:
                    print(f"Failed to load {fallback_model}: {str(e2)}")
            
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                print("All model loading attempts failed. Using text-generation pipeline with a default model.")
                try:
                    self.pipe = pipeline("text-generation", model="gpt2")
                    self.model_name = "gpt2"
                    self.using_pipeline = True
                except Exception as e3:
                    print(f"Failed to load even the simplest model: {str(e3)}")
                    raise RuntimeError("Failed to load any LLM model")
            else:
                self.using_pipeline = False
    
    def generate_response(self, query: str, context: List[str], max_length: int = 250, temperature: float = 0.7) -> str:
        """
        Generate a response based on the query and context.
        
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
        
        # Create a prompt with the context and query
        system_prompt = "You are a helpful, accurate, and friendly bank assistant for NUST Bank. Answer questions accurately based on the given context. If the answer is not in the context, politely indicate that you don't have that information rather than making up an answer. Always maintain a professional and helpful tone."
        
        # Join context into a single string
        context_text = "\n\n".join(context)
        
        # Construct the full prompt
        full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            if hasattr(self, 'using_pipeline') and self.using_pipeline:
                # Generate with pipeline
                result = self.pipe(
                    full_prompt, 
                    max_length=len(full_prompt.split()) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2
                )
                generated_text = result[0]['generated_text']
            else:
                # Generate with model and tokenizer
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                
                # Generate with the model
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.2
                    )
                
                # Decode the response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            answer = generated_text[len(full_prompt):].strip()
            
            # If the answer is empty, provide a fallback response
            if not answer:
                answer = "I don't have specific information on that topic in my knowledge base. Is there something else about NUST Bank's products or services that I can help you with?"
            
            # Apply response filtering
            filtered_answer = filter_response(answer)
            
            return filtered_answer
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I am currently experiencing technical difficulties. Please try again later."
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if hasattr(self, 'using_pipeline') and self.using_pipeline:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "using_pipeline": True
            }
        else:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "parameters": sum(p.numel() for p in self.model.parameters()) / 1_000_000,  # In millions
                "using_pipeline": False
            } 