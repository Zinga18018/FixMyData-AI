import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Define a session state key for applied fix tracking if not already defined globally
if 'applied_fixes_log' not in st.session_state:
    st.session_state.applied_fixes_log = []

def get_llm_recommendations_structured(issues_summary_str: Union[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Sends data issues to Ollama and gets structured recommendations.
    Args:
        issues_summary_str: Dictionary or string containing data quality issues
    Returns:
        List of dictionaries containing recommendations
    """
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}. Please ensure Ollama is running.")
        return []

    if isinstance(issues_summary_str, dict):
        issues_summary_str = json.dumps(issues_summary_str, indent=2)

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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException))
    )
    def call_ollama_api():
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]

    try:
        response_text = call_ollama_api().strip()

        with st.expander("View AI Interaction Details (Debug)", expanded=False):
            st.write("**Prompt sent to AI:**")
            st.code(prompt, language='markdown')
            st.write("**Raw response from AI:**")
            st.code(response_text, language='json')

        # Attempt to parse the JSON
        # Sometimes the model might still wrap the JSON in ```json ... ``` or have introductory text.
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
        response_text = response_text.strip()

        parsed_response = json.loads(response_text)

        # The prompt asks for a list directly, but some models might wrap it in a key like "recommendations"
        if isinstance(parsed_response, dict) and len(parsed_response.keys()) == 1:
            # If it's a dict with one key, assume the list is the value of that key
            # This handles cases where the model might output {"recommendations": [...]}
            potential_list = next(iter(parsed_response.values()))
            if isinstance(potential_list, list):
                return potential_list
        elif isinstance(parsed_response, list):
             return parsed_response # Expected format
        else:
            st.error("AI response was valid JSON, but not in the expected list format. Trying to adapt...")
            # Attempt to find a list within the JSON structure if possible (heuristic)
            if isinstance(parsed_response, dict):
                for key, value in parsed_response.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        st.warning(f"Found a list under key '{key}'. Using this list for recommendations.")
                        return value
            st.error("Could not adapt the AI's JSON response to the expected list of recommendations. Please check the debug output.")
            return []

    except json.JSONDecodeError as json_e:
        st.error(f"Error decoding JSON response from AI: {json_e}")
        st.error("The AI's response was not valid JSON. Please check the raw response in the debug section above.")
        return []
    except requests.exceptions.RequestException as req_e:
        st.error(f"Ollama API Error: {req_e}. This might be due to server issues or invalid requests.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while calling Ollama API: {e}")
        st.error(f"Type of error: {type(e)}")
        return []


def display_and_apply_fixes_structured(fixes_list: List[Dict[str, str]], fix_key_prefix: str) -> None:
    """
    Displays the AI recommendations and provides UI for applying fixes.
    Args:
        fixes_list: List of dictionaries containing fix recommendations.
        fix_key_prefix: A unique prefix for widget keys to avoid conflicts.
    """
    if not fixes_list:
        st.warning("No AI recommendations available to display or recommendations are malformed.")
        return

    st.info("⚠️ **Important:** AI-generated code can sometimes be imperfect. Always review the code and understand its implications before applying it to your data. Applied fixes modify a copy of your data for this session.")

    for i, fix in enumerate(fixes_list):
        # Validate fix structure
        if not all(key in fix for key in ["title", "explanation", "fix_description", "code", "affected_columns"]):
            st.warning(f"Recommendation {i+1} is malformed and will be skipped. Received: {fix}")
            continue

        with st.expander(f"Recommendation {i+1}: {fix['title']}"):
            st.markdown(f"**🤔 Why this is an issue:** {fix['explanation']}")
            st.markdown(f"**💡 Suggested Fix:** {fix['fix_description']}")
            st.markdown(f"**🎯 Primarily Affects Columns:** `{fix.get('affected_columns', 'N/A')}`")

            st.code(fix['code'], language='python')

            button_key = f"{fix_key_prefix}_apply_fix_{i}"
            apply_button_placeholder = st.empty()

            if apply_button_placeholder.button(f"Apply Fix: {fix['title']}", key=button_key, help="Review code before applying!"):
                with st.spinner(f"Applying fix: {fix['title']}..."):
                    original_df_head = st.session_state.df_modified.head().copy()
                    original_df_shape = st.session_state.df_modified.shape
                    original_missing_sum = st.session_state.df_modified.isna().sum().sum()

                    # Store current df_modified for potential undo
                    if 'undo_history' not in st.session_state:
                        st.session_state.undo_history = []
                    st.session_state.undo_history.append(st.session_state.df_modified.copy())
                    if len(st.session_state.undo_history) > 5: # Keep last 5 undo states
                        st.session_state.undo_history.pop(0)


                    # Apply fix using safe evaluation
                    temp_df = st.session_state.df_modified.copy() # Work on a temporary copy for this specific fix
                    
                    # Safe evaluation environment
                    namespace = {
                        'pd': pd,
                        'np': np,
                        'df': temp_df  # Operate on the temporary DataFrame
                    }
                    
                    # Capture stdout/stderr during execution
                    from io import StringIO
                    import sys
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    redirected_output = sys.stdout = StringIO()
                    redirected_error = sys.stderr = StringIO()

                    try:
                        # Clean up potential indentation issues from AI
                        cleaned_code = "\n".join([line.lstrip() for line in fix['code'].split('\n')])
                        print(f"\n--- Executing Code ---\n{cleaned_code}\n---", file=old_stdout) # Print code being executed to original stdout
                        
                        exec(cleaned_code, namespace)

                        # Restore stdout/stderr
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

                        captured_output = redirected_output.getvalue()
                        captured_error = redirected_error.getvalue()

                        modified_df_intermediate = namespace['df'] # Retrieve the modified df from the namespace

                        if not modified_df_intermediate.equals(st.session_state.df_modified):
                            st.session_state.df_modified = modified_df_intermediate # Update the main modified df
                            st.session_state.applied_fixes_log.append({
                                "title": fix['title'],
                                "code": fix['code']
                            })
                            st.success(f"Fix '{fix['title']}' applied successfully!")

                            if captured_output:
                                st.markdown("##### Output from fix code:")
                                st.code(captured_output)
                            if captured_error:
                                st.error("##### Error during fix code execution:")
                                st.code(captured_error)

                            # Show before/after comparison more effectively
                            st.markdown("##### Data Preview (First 5 Rows):")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Before Fix:**")
                                st.data_editor(original_df_head, height=200, key=f"{button_key}_before_data_editor", disabled=True, use_container_width=True)
                            with col2:
                                st.write("**After Fix:**")
                                st.data_editor(st.session_state.df_modified.head(), height=200, key=f"{button_key}_after_data_editor", disabled=True, use_container_width=True)

                            # Show impact statistics
                            st.markdown("##### Impact Statistics:")
                            impact_data = {
                                "Metric": ["Rows", "Columns", "Missing Values"],
                                "Before": [original_df_shape[0], original_df_shape[1], original_missing_sum],
                                "After": [st.session_state.df_modified.shape[0], st.session_state.df_modified.shape[1], st.session_state.df_modified.isna().sum().sum()]
                            }
                            st.table(pd.DataFrame(impact_data))
                            st.session_state.fix_applied_once = True # Flag to allow rerun for UI update
                        else:
                            st.warning("Applying this fix resulted in no changes to the dataset.")
                            if captured_output:
                                st.markdown("##### Output from fix code:")
                                st.code(captured_output)
                            if captured_error:
                                st.error("##### Error during fix code execution:")
                                st.code(captured_error)
                            # Remove the last item from undo history as no change occurred
                            if st.session_state.undo_history:
                                st.session_state.undo_history.pop()

                    except Exception as e:
                        # Ensure stdout/stderr are restored even if an error occurs
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

                        captured_output = redirected_output.getvalue()
                        captured_error = redirected_error.getvalue()

                        st.error(f"Error applying fix '{fix['title']}': {str(e)}")
                        st.error("The AI-generated code snippet might be incorrect or not applicable to your current dataset.")
                        if captured_output:
                                st.markdown("##### Output from fix code:")
                                st.code(captured_output)
                        if captured_error:
                                st.error("##### Error during fix code execution:")
                                st.code(captured_error)

                        # Remove the last item from undo history as the fix failed
                        if st.session_state.undo_history:
                            st.session_state.undo_history.pop()
                
                # Rerun to update UI elements, especially if tabs are involved
                # or to correctly reflect the modified data in other parts of the app.
                st.rerun()