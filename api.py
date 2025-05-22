from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np # Needed for exec namespace
from ydata_profiling import ProfileReport
import json
import os
import requests # For Ollama interaction
from typing import List, Dict, Any, Union
from io import StringIO, BytesIO
# We don't need tenacity in the API for a simple example, error handling is sufficient
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Ollama API configuration (assuming it runs on localhost:11434)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Initialize FastAPI app
app = FastAPI()

# --- Helper Function for Data Issue Extraction (copied from app.py) ---
def extract_data_issues(profile_json_data, df_original):
    issues = {"general": {}, "variables": {}}
    try:
        # General stats
        issues["general"]["n_variables"] = profile_json_data["table"]["n_var"]
        issues["general"]["n_observations"] = profile_json_data["table"]["n"]
        issues["general"]["missing_cells"] = profile_json_data["table"]["n_cells_missing"]
        issues["general"]["missing_cells_perc"] = profile_json_data["table"]["p_cells_missing"]
        issues["general"]["duplicate_rows"] = profile_json_data["table"]["n_duplicates"]
        issues["general"]["duplicate_rows_perc"] = profile_json_data["table"]["p_duplicates"]

        # Per-variable issues
        for var_name, var_details in profile_json_data["variables"].items():
            var_issues = {}
            if var_details.get("p_missing", 0) > 0.01 : # Report if > 1% missing
                var_issues["missing_percentage"] = round(var_details["p_missing"] * 100, 2)
            if var_details.get("p_zeros", 0) > 0.6:  # If > 60% are zeros
                var_issues["high_zeros_percentage"] = round(var_details["p_zeros"] * 100, 2)
            skewness = var_details.get("skewness")
            if skewness is not None and (skewness > 2 or skewness < -2): # Higher threshold for skewness
                var_issues["high_skewness"] = round(skewness, 2)

            # Basic outlier heuristic (can be expanded)
            if "iqr" in var_details: # Check if IQR is calculated (numeric types)
                q1 = var_details.get("q_25")
                q3 = var_details.get("q_75")
                iqr = var_details.get("iqr")
                min_val = var_details.get("min")
                max_val = var_details.get("max")
                if q1 is not None and q3 is not None and iqr is not None and min_val is not None and max_val is not None:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    # This is a simplified check; ydata-profiling itself flags outliers in its report
                    if min_val < lower_bound or max_val > upper_bound:
                         var_issues["potential_outliers_detected_by_iqr_rule"] = True

            profile_type = var_details.get("type")
            # We need access to the actual dataframe dtype here, which is tricky in a stateless API
            # For now, skipping the potential_type_mismatch check as it requires the original df
            # if var_name in df_original.columns: # Ensure column exists in DataFrame
            #     pandas_type = str(df_original[var_name].dtype)
            #     if profile_type == "Categorical" and not pd.api.types.is_string_dtype(df_original[var_name]) and not pd.api.types.is_categorical_dtype(df_original[var_name]):
            #         var_issues["potential_type_mismatch"] = f"Profiled as {profile_type} but pandas dtype is {pandas_type}"
            #     elif profile_type == "Numeric" and not pd.api.types.is_numeric_dtype(df_original[var_name]):
            #         var_issues["potential_type_mismatch"] = f"Profiled as {profile_type} but pandas dtype is {pandas_type}"

            if profile_type == "Unsupported" and var_details.get("check_mixed_type", {}).get("mixed_type", False):
                 var_issues["mixed_data_types"] = True

            if var_issues:
                issues["variables"][var_name] = var_issues

        if not issues["general"] and not issues["variables"]:
            return {"message": "No major data quality issues automatically flagged by basic checks. Review the full profile for details."}

        return issues
    except KeyError as e:
        # In an API, we log this server-side, maybe return a generic error
        print(f"KeyError while parsing profile JSON: {e}. Structure might have changed or key is missing. Some issues may not be extracted.")
        issues["error"] = f"Could not fully parse profiling report due to KeyError: {e}"
        return issues
    except Exception as e:
        print(f"Unexpected error extracting data issues: {e}")
        issues["error"] = f"Unexpected error: {e}"
        return issues

# Define a simple root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the FixMyData API"}

# Endpoint for checking Ollama status
@app.get("/ollama-status", tags=["Ollama"])
async def get_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return {"status": "Ollama is running", "models": response.json().get("models", [])}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama is not running or accessible: {e}")

# Endpoint for uploading a file and getting initial profile summary
@app.post("/upload-and-profile", tags=["Data"])
async def upload_file_and_profile(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xls', '.xlsx')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")

    try:
        # Read the file into a pandas DataFrame
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(contents.decode('utf-8', errors='ignore')), on_bad_lines='skip', low_memory=False)
        else:
            df = pd.read_excel(BytesIO(contents), engine='openpyxl')

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Generate profile report JSON
        profile = ProfileReport(df, title=f"Profiling Report for {file.filename}", explorative=True, lazy=False)
        profile_json_data = json.loads(profile.to_json())

        # Extract data issues
        issues_summary = extract_data_issues(profile_json_data, df) # Pass df if needed for type checks later

        # In a real API, you might save the dataframe temporarily or process it
        # and return an ID or a summary.

        # For this example, we'll just return the summary and expect the client
        # to send the data back for applying fixes.

        return {
            "filename": file.filename,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "issues_summary": issues_summary,
            "message": "File uploaded and processed successfully. Data profile summary extracted."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# Endpoint for getting AI recommendations
@app.post("/get-recommendations", tags=["AI"])
async def get_recommendations(issues_summary: Dict[str, Any]):
    """
    Sends data issues summary to Ollama and gets structured recommendations.
    """
    if "error" in issues_summary or "message" in issues_summary:
         raise HTTPException(status_code=400, detail="Invalid issue summary provided.")

    issues_summary_str = json.dumps(issues_summary, indent=2)

    prompt = f"""
You are an expert Data Quality Analyst AI. I have a dataset with the following issues:

{issues_summary_str}

For each distinct issue, please provide a clear and detailed response with:
- "title": A concise, descriptive title summarizing the issue and the suggested action (e.g., "Handle Missing Values in 'Age' Column").
- "explanation": A helpful explanation in a conversational tone about why this specific issue is a problem for data analysis or modeling.
- "fix_description": A recommended fix written in plain English that a non-technical person can understand. Clearly state any assumptions made (e.g., "filling missing ages with the median").
- "code": A Python code snippet using pandas to implement the fix. The DataFrame is always named `df`. Ensure the code is self-contained and directly applicable. For example, if a column needs to be targeted, include it in the code (e.g., `df['column_name'] = ...`). If the fix involves calculations like median or mean, show how to calculate it and apply it.
- "affected_columns": A list of column names that this fix would primarily affect. If it's a general fix (like dropping duplicates), this can be an empty list or `["all"]`.

It is CRUCIAL that your output is a valid JSON list of dictionaries, like this:
[
  {{
    "title": "Example: Impute Missing Age Data",
    "explanation": "Missing age data can skew analysis results and prevent some machine learning models from working correctly. It's important to handle these missing values appropriately.",
    "fix_description": "We can fill the missing 'Age' values with the median age of the dataset. The median is often a good choice as it's less sensitive to outliers than the mean.",
    "code": "median_age = df['Age'].median()\\ndf['Age'].fillna(median_age, inplace=True)",
    "affected_columns": ["Age"]
  }},
  {{
    "title": "Example: Remove Duplicate Rows",
    "explanation": "Duplicate rows can inflate dataset size and lead to incorrect statistical summaries and biased model training.",
    "fix_description": "We will remove rows that are exact duplicates across all columns, keeping the first occurrence.",
    "code": "df.drop_duplicates(inplace=True)",
    "affected_columns": ["all"]
  }}
]
Ensure that the JSON is well-formed and contains no extra text before or after the list.
Try to maintain context. If a previous fix might affect a subsequent one, briefly mention it if relevant.
"""

    try:
        payload = {
            "model": "mistral", # Or another suitable model you have pulled
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        response_data = response.json()
        response_text = response_data.get("response", "").strip()

        # Attempt to parse the JSON (similar logic from ai_integration.py)
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
        response_text = response_text.strip()

        parsed_response = json.loads(response_text)

        if isinstance(parsed_response, dict) and len(parsed_response.keys()) == 1:
            potential_list = next(iter(parsed_response.values()))
            if isinstance(potential_list, list):
                return potential_list
        elif isinstance(parsed_response, list):
             return parsed_response
        else:
            # Log the unexpected format server-side
            print(f"AI response was valid JSON, but not in the expected list format. Received: {parsed_response}")
            raise HTTPException(status_code=500, detail="AI response was not in the expected format.")

    except json.JSONDecodeError:
        print(f"Error decoding JSON response from AI. Raw response: {response_text}")
        raise HTTPException(status_code=500, detail="Could not decode JSON response from AI.")
    except requests.exceptions.RequestException as e:
        print(f"Ollama API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while getting AI recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Endpoint for applying fixes
@app.post("/apply-fix", tags=["Data"])
async def apply_fix(data: Dict[str, Any], fix_code: str):
    """
    Applies a given fix (Pandas code) to the provided data.
    Data is expected as a dictionary that can be loaded into a DataFrame.
    """
    try:
        # Load data from the request body into a DataFrame
        df = pd.DataFrame.from_dict(data)

        # Create a namespace for executing the code
        namespace = {
            'pd': pd,
            'np': np, # Include numpy as it's often used with pandas
            'df': df  # The DataFrame to operate on
        }

        # Execute the code snippet
        exec(fix_code, namespace)

        # Retrieve the modified DataFrame from the namespace
        modified_df = namespace['df']

        # Return the modified DataFrame as a dictionary
        return modified_df.to_dict(orient='records') # Or 'list', 'series', 'split', 'index', 'tight'

    except Exception as e:
        # Log the error server-side and return an error response
        print(f"Error applying fix: {e}")
        raise HTTPException(status_code=500, detail=f"Error applying fix: {e}") 