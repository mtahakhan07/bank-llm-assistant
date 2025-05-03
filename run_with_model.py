import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Run the bank assistant with the actual LLM model instead of mock mode.
    """
    parser = argparse.ArgumentParser(description="Run Bank LLM Assistant with an LLM model")
    parser.add_argument("--mock", action="store_true", help="Use mock mode instead of real model (RECOMMENDED for testing)")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit quantization (requires bitsandbytes)")
    parser.add_argument("--model", type=str, help="Custom model name to use (overrides .env)")
    parser.add_argument("--small", action="store_true", help="Use Llama-3.2-1B-Instruct (smaller, faster model)")
    args = parser.parse_args()

    # Determine which model to use
    if args.model:
        model_name = args.model
    elif args.small:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")  # Default to 3B model
    
    # Mode message
    mode = "mock" if args.mock else "actual model"
    quant = "with 8-bit quantization" if args.load_8bit else "without quantization"
    
    if not args.mock:
        logger.info("WARNING: Running in actual model mode requires downloading and loading the model.")
        logger.info("         This may take a few minutes and requires sufficient resources.")
        logger.info("         For testing purposes, consider using --mock mode.")
        logger.info(f"Starting Bank LLM Assistant in {mode} mode {quant}")
        logger.info(f"Using model: {model_name}")
    else:
        logger.info(f"Starting Bank LLM Assistant in {mode} mode (safe for testing)")
    
    # Import here to ensure environment variables are loaded first
    import subprocess
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["USE_MOCK_MODEL"] = "True" if args.mock else "False"
    env["LOAD_8BIT"] = "True" if args.load_8bit else "False"
    env["MODEL_NAME"] = model_name
    
    # Run Streamlit
    logger.info("Starting Streamlit...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"], env=env)

if __name__ == "__main__":
    main() 