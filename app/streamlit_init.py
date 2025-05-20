import os
import asyncio
import nest_asyncio
import warnings

# Configure environment variables
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._classes")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Apply nest_asyncio
nest_asyncio.apply()

# Set up event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Prevent Streamlit from watching torch._classes
import torch
torch._classes = None 