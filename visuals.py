import ast
import streamlit as st
import pandas as pd
import json 
import plotly.express as px
from io import StringIO
from ydata_profiling import ProfileReport


from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore
from langchain_core.prompts import HumanMessagePromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage



# Make sure these imports are at the top of your file
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#from langchain.agents.agent_types import AgentType
import plotly.express as px
from io import StringIO

# --- HELPER FUNCTIONS FOR VISUALIZATION (PHASE 5) ---

# Cập nhật hàm này

def generate_and_execute_chart_code(user_prompt, df, analytical_summary):
    """
    
    """
    if st.session_state.chat_llm is None:
        return None, "LLM not initialized."

    # Tạo một bản tóm tắt DataFrame cho prompt
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    df_head_markdown = df.head().to_markdown(index=False)
    
    # --- PROMPT MỚI, TỐI ƯU HƠN ---
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a highly advanced Python data visualization assistant, specializing in **real estate data analysis**. Your task is to analyze a user's request and, if it is relevant to the provided data, write Python code to generate a single Plotly figure. The DataFrame is available in the execution environment under the variable name 'df'. All text generated, especially chart titles, must be in German."
        )),
        HumanMessagePromptTemplate.from_template(
            """
            **Follow this thought process step-by-step:**

            **Step 1: Analyze Relevance.**
            - Read and understand the "User's Visualization Request".
            - Read and understand the columns and sample data:
              {df_head_markdown}

            - Read and understand the "Analytical Summary of the Data" to understand more key patterns, relationships.
            
            - Synthesize and understand data consistently, From this ask yourself: "Is the user's request a plausible analysis or visualization of THIS specific data?"
            
            - If the request is irrelevant (e.g., asks about the weather, stock prices,... or columns, values,... that don't exist), you MUST respond with a polite refusal in German. Start your response with "REJECTED:" followed by the explanation. Example: "REJECTED: Ihre Anfrage scheint sich nicht auf die bereitgestellten Immobiliendaten zu beziehen."
            - If the request is relevant, proceed to the next steps.

            **Step 2: Understand Column Semantics (Context).**
            
            **Categorical Columns:** Treat a column as CATEGORICAL (for grouping, not for math) if:
            - Its name contains keywords like: `id`, `code`, `number`, `zip`, `plz`, `postleitzahl`, `year`,...
            - **OR** if it contains numbers that represent labels, not quantities (e.g., it makes no sense to calculate the 'average' of a postal code).
            **Action:** In the generated code, always cast these columns to `str` (e.g., `df['PLZ'] = df['PLZ'].astype(str)`).

            **Numerical Columns:** Treat a column as NUMERIC if it represents a measurable quantity:
            - Its name contains keywords like: `price`, `preis`, `amount`, `revenue`, `count`, `age`, `zimmerzahl`, `area`, `fläche`.
            **Action:** Ensure these are numeric types for calculations.
            
            
            **Step 3: Plan the Visualization.**
            - Based on the user's request and your understanding of context, decide on the meaningful chart type (`bar`, `pie`, `line`, `histogram`, `scatter`, etc.) in real estate data analysis.
            - Determine the correct columns for `x`, `y`, `color`,... and any necessary `aggregation`.
           
            **Step 4: Write the Python Code.**
            - The code must use the `plotly.express` library (imported as `px`), pandas (imported as `pd`), and the DataFrame `df`.
            - The final line must be the creation of a Plotly figure object, assigned to a variable named `fig`.
            -  **VERY IMPORTANT for bar charts with categorical axes (example: 'Postleitzahl',...):** After creating the figure, you MUST force the categorical axis to be treated as a category to prevent incorrect numeric scaling.
            - Do NOT include `st.plotly_chart(fig)`. The app will display it.
            - Return ONLY the Python code block, with no explanations.

            ---
            **CONTEXT FOR YOUR TASK:**

            **User's Visualization Request:**
            {user_prompt}

            **Analytical Summary of the Data:**
            {analytical_summary}

            **DataFrame Columns (`df.columns`):**
            {column_names}

            **DataFrame Head (`df.head()`):**
            {df_head_markdown}
            ---

            Now, execute the thought process and provide your response in German.
            """
        )
    ])

    formatted_prompt = prompt_template.format_messages(
        user_prompt=user_prompt,
        analytical_summary=analytical_summary,
        column_names=df.columns.tolist(), # Cung cấp list tên cột
        df_head_markdown=df_head_markdown
    )

    try:
        print("Calling LLM to generate visualization code...")
        response = st.session_state.chat_llm.invoke(formatted_prompt)
        llm_output = response.content.strip()

        print(f"LLM Code Generation Output:\n{llm_output}")

        # --- KIỂM TRA XEM LLM CÓ TỪ CHỐI KHÔNG ---
        if llm_output.startswith("REJECTED:"):
            rejection_message = llm_output.replace("REJECTED:", "").strip()
            return None, rejection_message

        # Trích xuất code từ khối markdown
        code_to_execute = llm_output
        code_block_start = code_to_execute.find("```python")
        code_block_end = code_to_execute.rfind("```")
        if code_block_start != -1:
            code_to_execute = code_to_execute[code_block_start + 9 : code_block_end].strip()
        
        print(f"Code to execute:\n{code_to_execute}")

        # Chuẩn bị môi trường để thực thi code
        local_vars = {"df": df, "px": px, "pd": pd, "json": json, "ast": ast }
        
        # Thực thi code
        exec(code_to_execute, {"px": px, "pd": pd, "json": json, "ast": ast}, local_vars)
        
        # Lấy đối tượng figure đã được tạo
        fig = local_vars.get("fig")

        if fig:
            return fig, None
        else:
            return None, "The generated code did not create a 'fig' object."

    except Exception as e:
        print(f"Error generating or executing chart code: {e}")
        error_message = f"Fehler beim Erstellen des Diagramms:\n```\n{e}\n```\n**Generierter Code (fehlerhaft):**\n```python\n{code_to_execute}\n```"
        return None, error_message
   


def get_analytical_summary(df: pd.DataFrame) -> str:
    """
    Performs statistical analysis on a DataFrame using ydata-profiling,
    then uses an LLM to interpret the results into a human-readable summary.

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A string containing the analytical summary, or an empty string if an error occurs.
    """
    if st.session_state.chat_llm is None:
        print("LLM not available for generating analytical summary.")
        return ""

    try:
        print("Generating statistical profile...")
        # Generate a minimal profile to keep it fast
        profile = ProfileReport(df,
                                minimal=True,
                                title="Statistical Analysis",
                                explorative=True,                           
                                progress_bar=False) # Disable progress bar for cleaner logs

        # Convert the profile description dictionary to a string format for the prompt
        description = profile.get_description()
        
        # Format the statistical summary for the LLM prompt
        statistical_summary = f"""
        - **Dataset Overview:**
          - Number of rows: {description.table['n']}
          - Number of columns: {description.table['n_var']}
          - Missing cells: {description.table['n_cells_missing']} ({description.table['p_cells_missing']:.1%})


         - **Column Analysis:**
        """
        for var, info in description.variables.items(): # Dùng description.variables
            statistical_summary += f"  - **Column '{var}'** ({info['type']}):\n"
            if 'n_distinct' in info:
                statistical_summary += f"    - Distinct values: {info['n_distinct']} ({info['p_distinct']:.1%})\n"
            if 'n_missing' in info and info['n_missing'] > 0:
                statistical_summary += f"    - Missing values: {info['n_missing']} ({info['p_missing']:.1%})\n"
            if info['type'] == 'Numeric':
                # Sử dụng .get() để an toàn hơn nếu key không tồn tại
                mean_val = info.get('mean', 'N/A')
                std_val = info.get('std', 'N/A')
                min_val = info.get('min', 'N/A')
                max_val = info.get('max', 'N/A')
                
                # Định dạng chỉ khi giá trị là số
                mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else "N/A"
                std_str = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else "N/A"

                statistical_summary += f"    - Mean: {mean_str}, Std Dev: {std_str}\n"
                statistical_summary += f"    - Min: {min_val}, Max: {max_val}\n"
            if 'top' in info: # For categorical data
                 statistical_summary += f"    - Most frequent value: '{info['top']}'\n"


        # Prompt for the LLM to interpret the stats
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert data analyst, specializing in **real estate data analysis**. Your task is to interpret a statistical summary of a dataset and provide a concise, human-readable analysis in German. Focus on insights that are useful for visualization."),
            HumanMessagePromptTemplate.from_template(
                """
        **Follow this structured thinking process to analyze the provided data:**

        **Step 1: Structural Recognition**
        - Confirm that the data is tabular with rows and columns.
        - Note the number of rows and columns from the statistical summary.

        **Step 2: Semantic Understanding of Columns**
        - Analyze each column name and its statistical properties.
        - **Apply your domain knowledge about German real estate.** For example:
        - **'Postleitzahl' (PLZ):** Recognize this as a postal code, a categorical identifier for a geographical area. It should be used for grouping and frequency counts, NOT for numerical calculations (like mean).
        - **'Preis', 'Kaufpreis', 'Wohnfläche', 'Anzahl_Zimmer',...:** Identify these as key numeric metrics. They are suitable for aggregations (mean, sum, min, max) and distribution analysis.
        - **'Stadt', 'Ortsteil':** Recognize these as categorical location data.
        - For other columns, infer their meaning (e.g., 'realEstateId' is a unique identifier).

        **Step 3: Descriptive Statistics Analysis**
        - Review the provided statistical summary.
        - **Identify key findings:**
            - **Central Tendency & Spread:** What are the typical values (mean, median) and the range (min, max) for important numeric columns like 'Preis'?
            - **Frequency:** Which categories are most common in columns like 'Postleitzahl' or 'Ortsteil'?
            - **Missing Values:** Note any significant number of missing values and consider their impact.
            - **Potential Outliers:** Look for extreme min/max values. count or calculate for them, notice it for users

        **Step 4: Synthesis and Interpretation (Your Final Output)**
        - **Synthesize your findings** from the previous steps into a coherent analysis in German.
        - **Highlight key insights** that would be useful for visualization
        - **Count or calculate Formulate hypotheses for anomalies.** For example, if 'Preis' is minus, suggest possible reasons
        - **Highlight potential relationships** that would be interesting to visualize (e.g., "Es wäre interessant, die Beziehung zwischen Preis und Wohnfläche zu visualisieren.").
        - **Structure your output** with clear bullet points or short paragraphs. Start with a general overview, then delve into specifics.

        ---
        **DATA FOR YOUR ANALYSIS:**

        **Statistical Summary:**
        {statistical_summary}
        ---

        **Begin your analysis now. Provide the output in German.**
        """
            )
        ])
        
        formatted_prompt = prompt_template.format_messages(statistical_summary=statistical_summary)
        
        print("Calling LLM to generate analytical summary...")
        response = st.session_state.chat_llm.invoke(formatted_prompt)
        summary_text = response.content.strip()
        
        print(f"Generated Analytical Summary:\n{summary_text}")
        return summary_text

    except Exception as e:
        print(f"Error during analytical summary generation: {e}")
        return f"Fehler bei der automatischen Datenanalyse: {e}"