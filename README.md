# FixMyData: AI-Powered Data Quality Assistant

## Project Description

FixMyData is a data quality assistant that helps identify and suggest fixes for issues in datasets using AI (specifically, a local Ollama instance). The project demonstrates a modular architecture with a Python backend (FastAPI) for data processing and AI interaction, and includes examples of two different frontends: a simple HTML/JavaScript frontend and a Streamlit application.

This project is a great example for a portfolio, showcasing skills in data handling, API development, AI integration (local LLMs), and different frontend approaches.

## Components

*   **FastAPI Backend (`api.py`):** Provides API endpoints for:
    *   Uploading data files (CSV/Excel).
    *   Generating a data profile and extracting data quality issues.
    *   Getting AI recommendations for fixes using Ollama.
    *   Applying code-based fixes to the data.
*   **Simple HTML/JavaScript Frontend (`index.html`):** A basic static HTML file that demonstrates how to interact with the FastAPI backend using JavaScript to upload files, trigger profiling, get AI recommendations, and conceptually apply fixes. **Note:** The data handling for applying fixes in this simple frontend is conceptual due to the complexity of managing large DataFrame states in the browser; a real-world application would require a more robust data management strategy.
*   **Streamlit App (`app.py`):** The original Streamlit application, offering a single-file, interactive interface for data loading, profiling, and AI-assisted fixing.
*   **AI Integration (`ai_integration.py`):** Contains helper functions for interacting with the Ollama API, used by the Streamlit app.
*   **Requirements (`requirements.txt`):** Lists the Python packages required to run the backend and the Streamlit app.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd FixMyData-AI
    ```
2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Ollama:**
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/). Follow the installation instructions for your operating system.
5.  **Pull the Mistral model (or another compatible model):**
    *   Open a **new** terminal window (to ensure Ollama is in your PATH after installation).
    *   Run the command:
        ```bash
        ollama pull mistral
        ```
    *   Ensure the Ollama service is running (usually starts automatically after installation, or run `ollama serve`).

## How to Run

### Running the FastAPI Backend

1.  Make sure your Python virtual environment is activated.
2.  Make sure the Ollama service is running.
3.  Run the following command in the terminal:
    ```bash
    uvicorn api:app --reload
    ```
4.  The API will be running at `http://127.0.0.1:8000`. You can access the interactive documentation at `http://127.0.0.1:8000/docs`.

### Running the Simple HTML/JavaScript Frontend

1.  Ensure the FastAPI backend is running (as described above).
2.  Open the `index.html` file directly in your web browser.

### Running the Streamlit App

1.  Make sure your Python virtual environment is activated.
2.  Make sure the Ollama service is running and the `mistral` model is pulled.
3.  Run the following command in the terminal:
    ```bash
    streamlit run app.py
    ```
4.  The Streamlit app will open in your web browser (usually at `http://localhost:8501` or `8502`).

## Using the Streamlit App

Once the Streamlit app is running, follow these steps:

1.  **Upload your Data:** Use the file uploader in the sidebar to upload your CSV or Excel file.
2.  **Generate Data Profile:** Click the "Generate Data Profile" button in the sidebar. This will analyze your data and generate a detailed report.
3.  **Review Profile & Issues:** Navigate to the "Profiling Report" tab to view the comprehensive data quality report. The "AI Diagnosis & Fixes" tab will show a summary of extracted issues and provide AI-suggested fixes.
4.  **Apply AI Fixes:** In the "AI Diagnosis & Fixes" tab, review the recommended fixes and click the "Apply Fix" button for any fix you wish to apply. The "Modified Data" tab will appear, showing the dataset after applying fixes.
5.  **Download Data/Report:** You can download the modified data as a CSV from the "Modified Data" tab or download the full profiling report as HTML or PDF from the "Profiling Report" tab.

**Note on PDF Downloads:** The PDF download feature requires the external command-line tool **wkhtmltopdf** to be installed on your system and added to your system's PATH. You can download wkhtmltopdf from [https://wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html).

## Future Improvements

*   Implement robust data state management in the simple HTML/JS frontend to make the "Apply Fix" functionality fully working.
*   Build a more sophisticated and visually appealing frontend using a modern JavaScript framework (e.g., React, Vue, Svelte) with advanced CSS and animation libraries.
*   Add more data quality checks and corresponding AI fix recommendations.
*   Improve the prompting for the AI to potentially get better-formatted or more accurate code snippets.
*   Add unit tests for the backend API.
*   Containerize the application using Docker. 