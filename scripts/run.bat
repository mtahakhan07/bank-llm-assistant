@echo off
echo Starting NUST Bank LLM Assistant...
echo.

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run:
    echo python -m venv venv
    echo venv\Scripts\activate.bat
    echo pip install -r requirements.txt
    exit /b 1
)

REM Check if the index exists, if not run preprocessing
if not exist data\faiss_index.faiss (
    echo FAISS index not found. Running preprocessing...
    python preprocess.py
    if %ERRORLEVEL% neq 0 (
        echo Preprocessing failed. Please check the error messages above.
        exit /b 1
    )
)

REM Start the Streamlit application
echo Starting Streamlit application...
streamlit run main.py

REM Deactivate the virtual environment when done
call venv\Scripts\deactivate.bat 