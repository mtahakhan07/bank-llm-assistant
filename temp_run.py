
import os
import sys
from app.streamlit_app import main as app_main

# Override the environment to use real model
os.environ['USE_MOCK_MODEL'] = 'False'
os.environ['LOAD_8BIT'] = 'False'

if __name__ == '__main__':
    app_main()
        