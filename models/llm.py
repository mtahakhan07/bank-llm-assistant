import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaModel:
    """
    Wrapper for the Llama-3.2-3B-Instruct model from Meta.
    Handles loading, generation, and context management.
    """
    
    def __init__(
        self, 
        model_name: str = None,
        device: str = None,
        load_in_8bit: bool = False,
        mock_mode: bool = True  # Set to True by default for testing
    ):
        """
        Initialize the Llama model.
        
        Args:
            model_name: Name or path of the model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            load_in_8bit: Whether to load the model in 8-bit quantized format
            mock_mode: Whether to use a mock mode for testing instead of loading the model
        """
        # Get model name from environment variables if not provided
        if model_name is None:
            self.model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
        else:
            self.model_name = model_name
            
        # Get Hugging Face token from environment
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Determine the device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.load_in_8bit = load_in_8bit
        self.tokenizer = None
        self.model = None
        self.mock_mode = mock_mode
        
        if self.mock_mode:
            logger.info("Using mock mode for testing - no model will be loaded")
            # Initialize a small tokenizer for the prompt construction
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        else:
            # Always initialize the tokenizer for non-mock mode
            logger.info("Initializing tokenizer for actual model use")
            self.load_model()
        
    def load_model(self):
        """
        Load the Llama model and tokenizer.
        """
        if self.mock_mode:
            logger.info("Mock mode active - not loading the actual model")
            return
            
        logger.info(f"Loading Llama model: {self.model_name}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=self.hf_token
        )
        
        # Load the model with appropriate quantization if needed
        if self.load_in_8bit:
            logger.info("Loading model in 8-bit quantized format")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_8bit=True,
                token=self.hf_token
            )
        else:
            logger.info(f"Loading model on {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token
            ).to(self.device)
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text using the Llama model.
        
        Args:
            prompt: Input text to condition the generation
            max_length: Maximum length of the generated text
            temperature: Controls randomness (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalize repetition
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated text sequences
        """
        # For mock mode, just return a placeholder response
        if self.mock_mode:
            return [prompt + "\n\nThis is a mock response for testing purposes."]
            
        # Ensure model is loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Set pad token id to eos token id if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Create attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Configure generation parameters
        gen_config = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        logger.info(f"Generating text with parameters: {gen_config}")
        
        # Generate text
        with torch.no_grad():
            output_sequences = self.model.generate(**gen_config)
        
        # Decode and clean up the generated sequences
        generated_texts = []
        for output in output_sequences:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_answer(
        self,
        question: str,
        context: List[Dict[str, Any]],
        max_context_length: int = 1500,
        max_length: int = 200
    ) -> str:
        """
        Generate an answer based on a question and relevant context.
        
        Args:
            question: The user's question
            context: List of context documents with text and metadata
            max_context_length: Maximum number of tokens to use from context
            max_length: Maximum answer length
            
        Returns:
            Generated answer
        """
        # Construct a prompt with the question and context
        prompt = self._construct_prompt(question, context, max_context_length)
        
        # In mock mode, use the context to create a simple answer
        if self.mock_mode:
            for doc in context:
                if 'text' in doc:
                    text = doc['text']
                    if "Question: " in text and "Answer: " in text:
                        q_part = text.split("Question: ")[1].split("\n")[0]
                        a_part = text.split("Answer: ")[1]
                        
                        # If this context contains the exact question
                        if question.lower() in q_part.lower():
                            return a_part
            
            # If no exact match, return the first answer from context
            if context and len(context) > 0:
                if 'text' in context[0]:
                    text = context[0]['text']
                    if "Answer: " in text:
                        return text.split("Answer: ")[1]
                
            return "Based on the available information, I don't have a specific answer to that question."
        
        # Generate answer using the model
        answers = self.generate(
            prompt=prompt,
            max_length=len(self.tokenizer.encode(prompt)) + max_length,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Extract the answer part
        answer = answers[0][len(prompt):].strip()
        
        # Apply guardrails
        answer = self._apply_guardrails(question, answer)
        
        return answer
    
    def _construct_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        max_context_length: int
    ) -> str:
        """
        Construct a prompt from the question and context.
        
        Args:
            question: User's question
            context: List of context documents
            max_context_length: Maximum context length in tokens
            
        Returns:
            Formatted prompt string
        """
        # Format the context
        context_texts = []
        for doc in context:
            if 'text' in doc:
                context_texts.append(doc['text'])
            elif 'content' in doc:
                context_texts.append(doc['content'])
        
        # Create a prompt template for Llama models
        prompt_template = """<|system|>
You are a helpful, accurate, and friendly bank assistant. Answer the user's question based on the context provided below. 
If the answer cannot be determined from the context, politely say you don't have information about that topic.
Do not make up information or provide personal opinions.

Context:
{context}</|system|>

<|user|>
{question}</|user|>

<|assistant|>
"""
        
        # Join context, limited by max_context_length
        joined_context = "\n".join(context_texts)
        
        # Check the token length and truncate if necessary
        if self.tokenizer:
            context_tokens = self.tokenizer.encode(joined_context)
            if len(context_tokens) > max_context_length:
                joined_context = self.tokenizer.decode(context_tokens[:max_context_length])
        
        # Format the final prompt
        prompt = prompt_template.format(
            context=joined_context,
            question=question
        )
        
        return prompt
    
    def _apply_guardrails(self, question: str, answer: str) -> str:
        """
        Apply guardrails to the generated answer to prevent harmful content.
        
        Args:
            question: The original question
            answer: The generated answer
            
        Returns:
            Filtered answer
        """
        # List of sensitive topics for a bank assistant
        sensitive_topics = [
            "password", "account number", "social security", "credit card number",
            "pin", "cvv", "full card details", "routing number", "fraud",
            "hack", "illegal", "money laundering", "bypass security"
        ]
        
        # Check if the question asks about sensitive topics
        question_lower = question.lower()
        for topic in sensitive_topics:
            if topic in question_lower:
                return "I cannot provide information on this topic as it may involve sensitive or private data. For assistance with sensitive account information, please contact our customer service directly."
        
        # Check if the answer contains suspicious statements
        suspicious_phrases = [
            "I'll help you bypass", "Here's how to hack", "I can give you access",
            "workaround security", "avoid detection", "unauthorized access",
            "manipulate the system"
        ]
        
        answer_lower = answer.lower()
        for phrase in suspicious_phrases:
            if phrase in answer_lower:
                return "I cannot provide that information as it appears to involve security-sensitive operations. Please contact our customer service for legitimate assistance with your banking needs."
        
        return answer


if __name__ == "__main__":
    # Test the model with a simple example
    llm = LlamaModel(mock_mode=True)
    test_context = [
        {"text": "Customers can check their account balance through our mobile app, online banking platform, or by calling our 24/7 customer service line at 1-800-BANK."},
        {"text": "Our savings accounts offer competitive interest rates starting at 2.5% APY for balances over $1,000."}
    ]
    
    answer = llm.generate_answer(
        "How can I check my account balance?",
        test_context
    )
    
    print(f"Question: How can I check my account balance?")
    print(f"Answer: {answer}") 