# app.py
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import requests
import json
import os
from ai_integration import get_llm_recommendations_structured, display_and_apply_fixes_structured
import streamlit.components.v1 as components

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt, model="mistral"):
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        st.error(f"Error querying Ollama API: {str(e)}")
        return None

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FixMyData", page_icon="📊")

# --- Helper Function for Data Issue Extraction (as provided, can be expanded) ---
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
            if var_name in df_original.columns: # Ensure column exists in DataFrame
                pandas_type = str(df_original[var_name].dtype)
                if profile_type == "Categorical" and not pd.api.types.is_string_dtype(df_original[var_name]) and not pd.api.types.is_categorical_dtype(df_original[var_name]):
                    var_issues["potential_type_mismatch"] = f"Profiled as {profile_type} but pandas dtype is {pandas_type}"
                elif profile_type == "Numeric" and not pd.api.types.is_numeric_dtype(df_original[var_name]):
                    var_issues["potential_type_mismatch"] = f"Profiled as {profile_type} but pandas dtype is {pandas_type}"
                elif profile_type == "Unsupported" and var_details.get("check_mixed_type", {}).get("mixed_type", False):
                     var_issues["mixed_data_types"] = True


            if var_issues:
                issues["variables"][var_name] = var_issues
        if not issues["general"] and not issues["variables"]:
            return {"message": "No major data quality issues automatically flagged by basic checks. Review the full profile for details."}
        return issues
    except KeyError as e:
        st.warning(f"KeyError while parsing profile JSON: {e}. Structure might have changed or key is missing. Some issues may not be extracted.")
        issues["error"] = f"Could not fully parse profiling report due to KeyError: {e}"
        return issues
    except Exception as e:
        st.error(f"Unexpected error extracting data issues: {e}")
        issues["error"] = f"Unexpected error: {e}"
        return issues

# --- Caching Functions ---
@st.cache_data(max_entries=5, show_spinner="Loading data...")
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', low_memory=False)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

@st.cache_resource(max_entries=5, show_spinner="Generating data profile... this can take a moment.")
def generate_profile(_df, title="Data Profiling Report"):
    if _df is None or _df.empty:
        return None
    try:
        profile = ProfileReport(_df, title=title, explorative=True, lazy=False,
                                progress_bar=True) # Ensure ydata-profiling uses its progress bar if possible
        return profile
    except Exception as e:
        st.error(f"Error during profile generation: {e}")
        return None

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'profile_report': None,
        'df_original': None,
        'data_issues_summary': None,
        'llm_recommendations': None,
        'df_modified': None,
        'file_uploader_key': 0,
        'last_uploaded_filename': None,
        'data_loaded': False,
        'undo_history': [], # For storing previous states of df_modified
        'applied_fixes_log': [], # Log of applied fix titles and code
        'ollama_available': None,
        'fix_applied_once': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
.reportview-container {
    background: #1E1E1E;
    color: #D4D4D4;
}
.sidebar .sidebar-content {
    background: #2D2D2D;
    color: #D4D4D4;
}
h1, h2, h3, h4, h5, h6 {
    color: #569CD6;
}
.stButton>button {
    background-color: #007ACC;
    color: white;
    border-radius: 4px;
    border: none;
    padding: 10px 15px;
    font-size: 1em;
    margin-right: 5px;
}
.stButton>button:hover {
    background-color: #005f99;
    color: white;
}
.stExpander>div>div>div>p {
    font-weight: bold;
}
/* Style for info/warning/error boxes */
.stAlert > div > div:first-child {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Main App Logic ---
st.title("FixMyData: Data Quality Assistant")

# Check if Ollama is running
if st.session_state.ollama_available is None:
    try:
        response = requests.get("http://localhost:11434/api/tags")
        st.session_state.ollama_available = True
    except:
        st.session_state.ollama_available = False
        st.warning("⚠️ **Ollama is not running.** Please start Ollama to enable AI features.")

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.header("1. Upload & Profile")
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            st.session_state.last_uploaded_filename = uploaded_file.name
            try:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df_original = df.copy()
                    st.session_state.df_modified = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.profile_report = None
                    st.session_state.data_issues_summary = None
                    st.session_state.llm_recommendations = None
                    st.session_state.undo_history = []
                    st.session_state.applied_fixes_log = []
                    st.session_state.fix_applied_once = False
                    st.success(f"File '{uploaded_file.name}' loaded successfully!")
                    st.rerun()
                else:
                    st.session_state.data_loaded = False
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.df_original = None
                st.session_state.df_modified = None
                st.session_state.data_loaded = False
                st.session_state.file_uploader_key += 1
                st.rerun()

    if st.session_state.data_loaded and st.session_state.df_original is not None:
        if st.button("Generate Data Profile", key="profile_button", help="Analyzes data to identify quality issues. This may take time for large datasets."):
            if st.session_state.df_original is not None and not st.session_state.df_original.empty:
                st.session_state.profile_report = generate_profile(st.session_state.df_original, title=f"Profiling Report for {st.session_state.last_uploaded_filename}")
                if st.session_state.profile_report:
                    st.success("Profiling complete!")
                    try:
                        report_json = st.session_state.profile_report.to_json()
                        report_data = json.loads(report_json)
                        st.session_state.data_issues_summary = extract_data_issues(report_data, st.session_state.df_original)

                        # Automatically get AI recommendations after extracting issues if Ollama is available and issues were found
                        if st.session_state.ollama_available and st.session_state.data_issues_summary and "error" not in st.session_state.data_issues_summary and "message" not in st.session_state.data_issues_summary:
                            with st.spinner("Consulting AI for explanations and fixes... This may take a moment."):
                                st.session_state.llm_recommendations = get_llm_recommendations_structured(st.session_state.data_issues_summary)
                                if st.session_state.llm_recommendations:
                                     st.success("AI recommendations received!")
                                else:
                                    st.error("Failed to get recommendations from AI after profiling. Check error messages in AI Interaction Details.")

                    except Exception as e:
                        st.warning(f"Could not extract detailed issue summary from JSON report: {e}")
                        st.session_state.data_issues_summary = {"error": f"Could not auto-extract issues: {e}"}
                    st.rerun()
                else:
                    st.error("Failed to generate data profile. Check data format or content.")
            else:
                st.warning("Cannot generate profile. DataFrame is empty or not loaded.")
    else:
        st.info("Upload a data file to begin.")

    if st.session_state.undo_history:
        if st.button("Undo Last Applied Fix", key="undo_fix", help="Reverts the last applied data modification."):
            previous_df_state = st.session_state.undo_history.pop()
            st.session_state.df_modified = previous_df_state
            if st.session_state.applied_fixes_log:
                undone_fix_title = st.session_state.applied_fixes_log.pop()["title"]
                st.success(f"Successfully reverted the fix: '{undone_fix_title}'.")
            else:
                st.success("Last fix undone.")
            st.rerun()

# --- Main Area for Displaying Information ---
if not st.session_state.data_loaded:
    st.markdown("### Welcome to FixMyData!")
    st.markdown("I'm your AI-powered assistant to help you identify and fix data quality issues.")
    st.markdown("**Get started:**")
    st.markdown("1.  **Upload your CSV or Excel file** using the sidebar.")
    st.markdown("2.  Click **'Generate Data Profile'** to analyze your data.")
    st.markdown("3.  Review the **AI Diagnosis & Fixes** to improve your dataset.")
    if not st.session_state.ollama_available:
         st.warning("Remember to start your **Ollama** service to enable AI features.")

if st.session_state.data_loaded and st.session_state.df_original is not None:
    tab_titles = ["Original Data", "Profiling Report", "AI Diagnosis & Fixes"]

    # Determine if the Modified Data tab and related tabs should be shown
    show_modified_tab = st.session_state.fix_applied_once or (st.session_state.df_modified is not None and not st.session_state.df_original.equals(st.session_state.df_modified))

    # Build the list of tabs based on whether modified data exists
    all_tab_titles = tab_titles.copy()
    if show_modified_tab:
        all_tab_titles.extend(["Modified Data", "Custom Code", "Applied Fixes Log"])

    # Create the tabs
    tabs = st.tabs(all_tab_titles)

    # Display content for each tab using indices
    with tabs[0]: # Original Data
        st.subheader(f"Preview of Original Data: `{st.session_state.last_uploaded_filename}`")
        st.dataframe(st.session_state.df_original.head(100), height=400, use_container_width=True)
        st.write(f"Shape: {st.session_state.df_original.shape[0]} rows, {st.session_state.df_original.shape[1]} columns")

    with tabs[1]: # Profiling Report
        st.subheader("Data Profiling Insights")
        if st.session_state.profile_report:
            st.info("The profiling report provides a comprehensive overview of your data. Key insights are used by the AI for recommendations.")

            profile_html = st.session_state.profile_report.to_html()
            components.html(profile_html, height=800, scrolling=True)

            if st.download_button(label="Download Full HTML Report",
                                 data=profile_html,
                                 file_name=f"{st.session_state.last_uploaded_filename}_profiling_report.html",
                                 mime="text/html"):
                st.balloons()

            st.markdown("----")
            st.markdown("**Download Report as PDF (Requires wkhtmltopdf):**")
            st.warning("To download the report as PDF, you need to have **wkhtmltopdf** installed on your system and added to your PATH.")
            st.markdown("Download wkhtmltopdf from [https://wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html)")

            try:
                import pdfkit
                html_path = f"{st.session_state.last_uploaded_filename}_profiling_report.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(profile_html)

                # Example path, user might need to change
                path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
                config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

                try:
                    pdf_output = pdfkit.from_file(html_path, False, configuration=config)
                    st.download_button(
                        label="Download Full PDF Report",
                        data=pdf_output,
                        file_name=f"{st.session_state.last_uploaded_filename}_profiling_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as pdf_e:
                    st.error(f"Error generating PDF: {pdf_e}. Make sure wkhtmltopdf is installed and in your system\'s PATH, or update the path in the code.")
                finally:
                    if os.path.exists(html_path):
                        os.remove(html_path)

            except ImportError:
                st.warning("Python library `pdfkit` not found. Install it (`pip install pdfkit`) to enable PDF downloads.")
            except Exception as e:
                 st.error(f"An unexpected error occurred during PDF setup: {e}")

            st.markdown("---")
            st.markdown("#### Extracted Data Issues Summary:")
            if st.session_state.data_issues_summary:
                 if "message" in st.session_state.data_issues_summary:
                     st.success(st.session_state.data_issues_summary["message"])
                 elif "error" in st.session_state.data_issues_summary:
                     st.error(f"Error extracting issues: {st.session_state.data_issues_summary['error']}")
                 else:
                    st.info("No issues automatically summarized yet, or profile not generated.")
            else:
                st.info("No issues automatically summarized yet, or profile not generated.")


        elif 'profile_button' in st.session_state and st.session_state.profile_button:
            st.warning("Profiling report is being generated or an error occurred. Please check notifications.")
        else:
            st.info("Click 'Generate Data Profile' in the sidebar to view the report and enable AI diagnosis.")


    with tabs[2]: # AI Diagnosis & Fixes
        st.subheader("AI-Powered Diagnosis & Recommendations")
        if not st.session_state.ollama_available:
            st.error("AI features are disabled. Ollama is not running.")
        elif st.session_state.data_issues_summary and "error" not in st.session_state.data_issues_summary and "message" not in st.session_state.data_issues_summary :
            st.markdown("##### Summary of Detected Data Issues (used for AI):")
            if isinstance(st.session_state.data_issues_summary, dict):
                for section, content in st.session_state.data_issues_summary.items():
                    if content:
                        st.markdown(f"**{section.capitalize()}:**")
                        if isinstance(content, dict):
                            for key, value in content.items():
                                st.markdown(f"- **{key.replace('_', ' ').title()}**: {value if not isinstance(value, dict) else json.dumps(value)}")
                        else:
                             st.markdown(f"- {content}")


            if st.session_state.llm_recommendations:
                st.markdown("--- \n#### AI Recommended Fixes:")
                # display_and_apply_fixes_structured function now handles the display and application of fixes
                display_and_apply_fixes_structured(st.session_state.llm_recommendations, fix_key_prefix="ai_fix")
            elif st.session_state.ollama_available:
                 st.info("Click 'Get AI Recommendations' after generating a profile to see suggestions here.")

            elif st.session_state.data_issues_summary and "message" in st.session_state.data_issues_summary:
                st.success(st.session_state.data_issues_summary["message"] + " No specific issues were sent to the AI for recommendations.")
            elif st.session_state.profile_report:
                st.warning("Could not automatically extract a summary of issues for the AI, or summary indicates no major issues. Please review the full profiling report. AI recommendations may be limited.")
            else:
                st.info("Please generate a data profile first to enable AI diagnosis.")



    # Conditionally display other tabs if show_modified_tab is True
    if show_modified_tab:
        # The index of the first conditional tab (Modified Data) will be 3
        modified_data_tab_index = 3
        custom_code_tab_index = 4
        applied_fixes_log_tab_index = 5

        with tabs[modified_data_tab_index]: # Modified Data
            st.subheader("Preview of Modified Data")
            if st.session_state.df_modified is not None:
                if not st.session_state.df_original.equals(st.session_state.df_modified):
                    st.success("Data has been modified based on applied fixes.")
                    st.dataframe(st.session_state.df_modified.head(100), height=400, use_container_width=True)
                    st.write(f"Shape: {st.session_state.df_modified.shape[0]} rows, {st.session_state.df_modified.shape[1]} columns")

                    csv_data = st.session_state.df_modified.to_csv(index=False).encode('utf-8')
                    if st.download_button("Download Modified CSV",
                                         data=csv_data,
                                         file_name=f"{st.session_state.last_uploaded_filename}_modified.csv",
                                         mime="text/csv"):
                        st.balloons()
                else:
                    st.info("No fixes have been applied yet, or applied fixes resulted in no changes.")
            else:
                st.info("Modified data will appear here after fixes are applied.")


        with tabs[custom_code_tab_index]: # Custom Code
            st.subheader("Apply Custom Pandas Code")
            st.warning("⚠️ **Use with caution!** Code entered here will be executed directly. Ensure it\'s correct and safe. It operates on the currently modified version of your data.")
            custom_code = st.text_area("Enter your Pandas code here (use \'df\' as the DataFrame variable):", height=150, placeholder="e.g., df.drop(columns=[\'unwanted_column\'], inplace=True)\\nprint(f\'Shape after custom drop: {df.shape}\')")

            if st.button("Apply Custom Code", key="apply_custom_code"):
                if custom_code and st.session_state.df_modified is not None:
                    with st.spinner("Applying custom code...\")"):
                        original_df_head = st.session_state.df_modified.head().copy()
                        original_df_shape = st.session_state.df_modified.shape
                        original_missing_sum = st.session_state.df_modified.isna().sum().sum()

                        st.session_state.undo_history.append(st.session_state.df_modified.copy())
                        if len(st.session_state.undo_history) > 5:
                            st.session_state.undo_history.pop(0)

                        temp_df = st.session_state.df_modified.copy()
                        # Removed 'np' from namespace as it's not used in the example placeholder and might cause issues if not imported
                        namespace = {'pd': pd, 'df': temp_df, 'st': st}

                        try:
                            from io import StringIO
                            import sys
                            old_stdout = sys.stdout
                            redirected_output = sys.stdout = StringIO()

                            exec(custom_code, namespace)

                            sys.stdout = old_stdout
                            captured_output = redirected_output.getvalue()

                            modified_df_intermediate = namespace['df']

                            if not modified_df_intermediate.equals(st.session_state.df_modified) or captured_output:
                                st.session_state.df_modified = modified_df_intermediate
                                st.session_state.applied_fixes_log.append({
                                    "title": "Custom Code Execution",
                                    "code": custom_code
                                })
                                st.success("Custom code applied successfully!")
                                if captured_output:
                                    st.markdown("##### Output from custom code:")
                                    st.code(captured_output)

                                st.markdown("##### Data Preview (First 5 Rows):")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Before Custom Code:**")
                                    st.data_editor(original_df_head, height=200, key="custom_before_editor", disabled=True, use_container_width=True)
                                with col2:
                                    st.write("**After Custom Code:**")
                                    st.data_editor(st.session_state.df_modified.head(), height=200, key="custom_after_editor", disabled=True, use_container_width=True)

                                st.markdown("##### Impact Statistics:")
                                impact_data = {
                                    "Metric": ["Rows", "Columns", "Missing Values"],
                                    "Before": [original_df_shape[0], original_df_shape[1], original_missing_sum],
                                    "After": [st.session_state.df_modified.shape[0], st.session_state.df_modified.shape[1], st.session_state.df_modified.isna().sum().sum()]
                                }
                                st.table(pd.DataFrame(impact_data))
                                st.session_state.fix_applied_once = True
                            else:
                                st.warning("Custom code resulted in no changes to the dataset and produced no output.")
                                if st.session_state.undo_history:
                                    st.session_state.undo_history.pop()

                        except Exception as e:
                            st.error(f"Error executing custom code: {str(e)}")
                            if st.session_state.undo_history:
                                st.session_state.undo_history.pop()
                    st.rerun()
                elif not custom_code:
                    st.warning("Please enter some code to apply.")
                else:
                    st.error("Data not loaded. Upload and process data first.")


        with tabs[applied_fixes_log_tab_index]: # Applied Fixes Log
            st.subheader("Log of Applied Fixes")
            if st.session_state.applied_fixes_log:
                for i, log_entry in enumerate(reversed(st.session_state.applied_fixes_log)):
                    with st.expander(f"{len(st.session_state.applied_fixes_log)-i}. {log_entry['title']}"):
                        st.code(log_entry['code'], language='python')
            else:
                st.info("No fixes have been applied yet in this session.")


# Final check if DataFrame couldn't be loaded (e.g., unsupported format not caught by uploader type)
if uploaded_file is not None and not st.session_state.data_loaded :
    st.error("Could not load the DataFrame. Please check the file format or content, or try a different file.")