"""
Run script for the NUST Bank LLM Assistant.
This script will preprocess the data if needed and then start the Streamlit application.
"""

import os
import subprocess
import sys

def main():
    """Main function to check data and start the application."""
    # Check if the FAISS index exists
    if not os.path.exists("data/faiss_index.faiss") or not os.path.exists("data/faiss_index_mapping.pkl"):
        print("FAISS index not found. Running preprocessing...")
        
        # Run preprocessing
        result = subprocess.run([sys.executable, "preprocess.py"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Preprocessing failed:")
            print(result.stderr)
            return False
        
        print(result.stdout)
    
    # Run the Streamlit application
    print("Starting Streamlit application...")
    subprocess.run(["streamlit", "run", "main.py"])
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 