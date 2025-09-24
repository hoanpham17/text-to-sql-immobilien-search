import pandas as pd
import streamlit as st # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import json 
from visuals import *


import plotly.express as px
#from geopy.geocoders import Nominatim
#from geopy.extra.rate_limiter import RateLimiter
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#from langchain.agents.agent_types import AgentType


from sqlalchemy import create_engine, inspect, text # type: ignore
from sqlalchemy.exc import SQLAlchemyError, OperationalError # type: ignore
from sqlalchemy.dialects.postgresql import JSON, JSONB # type: ignore 


from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore
from langchain_core.prompts import HumanMessagePromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
#from langchain.agents import create_openai_tools_agent, AgentExecutor
#from langchain_core.tools import tool
#from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# --- Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Cấu hình lựa chọn LLM mặc định ---
USE_LLM = os.getenv("USE_LLM", "Google")

# --- Folder Creation ---
folders_to_create = ['csvs']
for folder_name in folders_to_create:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


# --- Manage state in Streamlit Session State ---

if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = None
if 'chat_llm' not in st.session_state:
    st.session_state.chat_llm = None
if 'generated_sql' not in st.session_state:
     st.session_state.generated_sql = None
if 'current_llm_info' not in st.session_state:
     st.session_state.current_llm_info = None 
if 'query_result_df' not in st.session_state: 
     st.session_state.query_result_df = None
if 'query_error' not in st.session_state: 
     st.session_state.query_error = None
if 'query_result_df' not in st.session_state:
     st.session_state.query_result_df = None
if 'query_error' not in st.session_state:
     st.session_state.query_error = None
if 'last_user_query' not in st.session_state:
    st.session_state.last_user_query = ""
if 'analytical_summary' not in st.session_state:
    st.session_state.analytical_summary = None
if 'data_previews' not in st.session_state:
     st.session_state.data_previews = None

# --- CÁC KEY MỚI CHO VISUALIZATION & DASHBOARD ---
# Tab 1: Đề xuất
if 'chart_suggestions' not in st.session_state:
    st.session_state.chart_suggestions = [] # Danh sách các dict đề xuất từ LLM
if 'chart_to_display' not in st.session_state:
    st.session_state.chart_to_display = None # Dict của biểu đồ đang được hiển thị trong Tab 1

# Tab 3: Dashboard
if 'dashboard_items' not in st.session_state:
    st.session_state.dashboard_items = [] # Danh sách các widget đã được ghim
if 'original_dashboard_df' not in st.session_state:
    st.session_state.original_dashboard_df = None # DataFrame gốc cho dashboard

# --- Phase 1: Connect to Database and Read Schema ---

def get_data_previews(engine, schema_info):
    """
    Lấy 10 dòng đầu tiên của mỗi bảng usable và lưu vào một dictionary.
    Chuyển đổi các cột có kiểu dữ liệu phức tạp (object) sang chuỗi để tránh lỗi hiển thị của Arrow.

    Args:
        engine: SQLAlchemy engine instance.
        schema_info (dict): Thông tin schema đã được đọc.

    Returns:
        dict: Một dictionary với key là "schema.table" và value là DataFrame Pandas.
    """
    previews = {}
    if not schema_info or not engine:
        return previews

    print("Fetching data previews for usable tables...")
    for schema_name, tables in schema_info.items():
        if isinstance(tables, dict):
            for table_name, columns in tables.items():
                
                if isinstance(columns, list) and columns:
                    full_table_name = f"{schema_name}.{table_name}"
                    try:
                        
                        query = text(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT 10;')
                        df = pd.read_sql_query(query, engine)
                    
                        for col in df.columns:
                            
                            if df[col].dtype == 'object':
                                df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
                        
                        previews[full_table_name] = df
                        print(f"  - Successfully fetched and processed preview for {full_table_name}")
                    except Exception as e:
                        print(f"  - Could not fetch preview for {full_table_name}: {e}")
                        previews[full_table_name] = pd.DataFrame(
                            {"error": [f"Vorschau konnte nicht geladen werden: {e}"]}
                        )
    return previews

def connect_and_load_schema(db_uri):
    """
    Connect to the database, read the schema, and analyze sample structure for JSON/JSONB columns.
    """
    
    if st.session_state.db_engine:
        try:
            st.session_state.db_engine.dispose()
            print("Existing DB engine disposed.")
        except Exception as e:
             print(f"Error disposing existing engine: {e}")

    st.session_state.db_engine = None
    st.session_state.schema_info = None
    st.session_state.generated_sql = None

    st.session_state.chat_llm = None
    st.session_state.current_llm_info = None


    engine = None
    try:
        print(f"Attempting to create database engine for URI: {db_uri}")
        
        # engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600, connect_args={"options": "-c timezone=UTC"})
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)


        print("Testing database connection...")
        with engine.connect() as connection:
    
             connection.execute(text("SELECT 1"))
        print("Database connection successful!")

        print("Reading database schema...")
        inspector = inspect(engine)

        all_schemas = inspector.get_schema_names()
        print(f"Found schemas: {all_schemas}")

        database_schema = {}
        SAMPLE_JSON_ROWS_LIMIT = 5 

        for schema in all_schemas:
            # Skip common system schemas
            system_schemas = ['information_schema', 'pg_catalog', 'mysql', 'sys', 'performance_schema', 'temp_tables', 'information_schema', 'auth', 'realtime', 'storage', 'vault']
            if schema.lower() in [s.lower() for s in system_schemas]: 
                print(f"  Skipping system schema: {schema}")
                continue
            

            print(f"  Processing schema: {schema}")
            database_schema[schema] = {}

            try:
                tables_in_schema = inspector.get_table_names(schema=schema)
                print(f"    Tables in '{schema}': {tables_in_schema}")

                for table in tables_in_schema:
                    print(f"      Processing table: {schema}.{table}")
                    try:
                        columns = inspector.get_columns(table, schema=schema)
                        database_schema[schema][table] = columns
                        print(f"        Found {len(columns)} columns for {schema}.{table}")

                        # --- Analyze sample JSON structure ---
                        for col in columns:
                            # Check if the column is of type JSON or JSONB
                            # Use isinstance to check SQLAlchemy data type
                            # Or check the type name string (in case dialect import fails or for other DBs)
                            is_json_type = isinstance(col['type'], (JSON, JSONB)) or 'JSON' in str(col['type']).upper()

                            if is_json_type:
                                print(f"          Column '{col['name']}' is JSON type. Attempting to sample data...")
                                json_summary = ""
                                try:
                                    # Safe query to sample JSON data
                                    # Using text() and double quotes for DB object names is good practice for PostgreSQL
                                    # Ensure schema/table/column names do not contain '";' to avoid basic SQL Injection here
                                
                                    # sql = text(f'SELECT "{col["name"]}" FROM "{schema}"."{table}" WHERE "{col["name"]}" IS NOT NULL LIMIT {SAMPLE_JSON_ROWS_LIMIT};')
                                    # An even safer way is to use Parameter Binding if column/table names are variables, but here we get them from schema info, so f-string + quoting is temporarily acceptable
                                    sql_query_str = f'SELECT "{col["name"]}" FROM "{schema}"."{table}" WHERE "{col["name"]}" IS NOT NULL LIMIT {SAMPLE_JSON_ROWS_LIMIT};'
                                    sample_sql = text(sql_query_str)


                                    with engine.connect() as connection:
                                        sample_data = connection.execute(sample_sql).fetchall()

                                    if sample_data:
                                        # Simple sample structure analysis
                                        sample_summary_lines = []
                                        common_keys = {}
                                        is_potential_array_of_objects = True
                                        found_array = False
                                        found_object = False
                                        examples_to_show = 1  
                                        examples_found = 0

                                        for i, row in enumerate(sample_data):
                                            json_value = row[0] 
                                            if json_value is None: continue
                                            
                                            if isinstance(json_value, str) and len(json_value) < 15:continue

                                            if examples_found >= examples_to_show:
                                                break
                                            # Summarize 1-2 examples
                                            
                                            try:
                                                # Try dumps with indent for readability, truncate if too long
                                                
                                                sample_str = json.dumps(json_value, ensure_ascii=False, indent=2)
                                                sample_summary_lines.append(f"  - Example:\n{sample_str}")
                                                #sample_summary_lines.append(f"  - Example {i+1}: {sample_str[:400]}{'...' if len(sample_str) > 300 else ''}")
                                                examples_found += 1
                                                
                                            except Exception as e:
                                                # Fallback if dumps fails (e.g., non-JSON-serializable type)
                                                sample_summary_lines.append(f"  - Example: {str(json_value)} (Serialization Error: {e})")
                                                #sample_summary_lines.append(f"  - Example {i+1}: {str(json_value)[:400]}{'...' if len(str(json_value)) > 300 else ''} (Serialization Error: {e})")
                                                examples_found += 1
                                            

                                            # Analyze common keys if value is list/dict
                                            if isinstance(json_value, list):
                                                found_array = True
                                                
                                                if not all(isinstance(item, dict) or item is None for item in json_value): 
                                                    is_potential_array_of_objects = False

                                                for item in json_value:
                                                    if isinstance(item, dict):
                                                        for key in item:
                                                                common_keys[key] = common_keys.get(key, 0) + 1
                                            elif isinstance(json_value, dict):
                                                found_object = True
                                                is_potential_array_of_objects = False 
                                                for key in json_value:
                                                    common_keys[key] = common_keys.get(key, 0) + 1
                                            else:
                                                is_potential_array_of_objects = False 


                                        # Build summary string for JSON structure
                                        json_summary_parts = []
                                        if found_array and is_potential_array_of_objects:
                                            json_summary_parts.append("  - This seems to be a JSON array of objects.")
                                            if common_keys:
                                                json_summary_parts.append("    Common keys found in array elements: " + ", ".join([f"'{k}'" for k in common_keys.keys()]) + ".")
                                        elif found_object:
                                            json_summary_parts.append("  - This seems to be a JSON object.")
                                            if common_keys:
                                                json_summary_parts.append("    Common top-level keys: " + ", ".join([f"'{k}'" for k in common_keys.keys()]) + ".")
                                        elif found_array: # Chỉ là mảng, không phải mảng object
                                            json_summary_parts.append("  - This seems to be a JSON array (not of objects).")
                                        else: # Kiểu dữ liệu khác (chuỗi, số, boolean trong JSON)
                                            json_summary_parts.append("  - Structure: Mixed or simple values (not array/object) in samples.")

                                        if sample_summary_lines:
                                            json_summary_parts.append("  Sample data examples:")
                                            json_summary_parts.extend(sample_summary_lines)
                                    
                                        json_summary = "\n".join(json_summary_parts)


                                    else:
                                        json_summary = "  - No non-null sample data found for JSON column.\n"

                                    # Lưu tóm tắt vào thông tin cột
                                    col['json_structure_summary'] = json_summary
                                    print(f"            JSON summary added for '{col['name']}'.")
                                    
                                    
                                except Exception as json_sample_exc:
                                    print(f"          Could not sample JSON data for {schema}.{table}.{col['name']}: {json_sample_exc}")
                                    
                                    err_detail = str(json_sample_exc)
                                    if hasattr(json_sample_exc, 'orig') and json_sample_exc.orig is not None:
                                        err_detail += f" (Original: {json_sample_exc.orig})"
                                    col['json_structure_summary'] = f"  - Could not sample JSON data: {err_detail}\n" 
                           
                    except Exception as col_exc:
                        print(f"        Could not get columns for {schema}.{table}: {col_exc}")
                        database_schema[schema][table] = []

            except Exception as table_list_exc:
                print(f"    Could not get tables for schema {schema}: {table_list_exc}")
                


        # Removing empty or column-less schema
        schemas_to_remove = [s for s, tables in database_schema.items() if not tables or all(not cols for cols in tables.values())]
        for s in schemas_to_remove:
             print(f"  Removing empty or column-less schema '{s}' from schema info.")
             del database_schema[s]


        st.session_state.db_engine = engine
        st.session_state.schema_info = database_schema

        if not st.session_state.schema_info:
            warning_msg = "Connection successful, but no usable schemas/tables found (excluding system schemas and empty ones)."
            print(warning_msg)
            st.warning(warning_msg)
            return {"message": warning_msg, "warning": True}


        
        st.session_state.data_previews = get_data_previews(engine, st.session_state.schema_info)
        

        print("Database schema loaded successfully!")
        success_msg = "Connection successful and schema loaded!"
        st.success(success_msg) 
        return {"message": success_msg} 


    except OperationalError as e:
        print(f"Database connection failed: {e}")
        if engine:
             engine.dispose()
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        st.error(f"Database connection error: {err_detail}")
        return {"error": f"Database connection failed: {err_detail}"}
    except SQLAlchemyError as e:
        print(f"A SQLAlchemy error occurred: {e}")
        if engine:
             engine.dispose()
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        st.error(f"SQLAlchemy error: {err_detail}")
        return {"error": f"A database error occurred: {err_detail}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if engine:
             engine.dispose()
        st.error(f"unexpected error: {e}")
        return {"error": f"An unexpected error occurred: {e}"}




# --- LLM models definition ---
AVAILABLE_LLM_MODELS = {
    "OpenAI": [
        "gpt-3.5-turbo",
        "o3",
        "o3-mini",
        "o4-mini", 
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "o1-mini",
    ],
    "Google": [
        "gemini-2.5-flash", 
        "gemini-2.5-pro", 
        "gemini-2.5-flash-lite-preview-06-17", 
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro" 
      
    ]
    
}

if USE_LLM not in AVAILABLE_LLM_MODELS or not AVAILABLE_LLM_MODELS[USE_LLM]:
     found_provider = False
     for provider, models in AVAILABLE_LLM_MODELS.items():
          if models:
               USE_LLM = provider
               found_provider = True
               break
     print(f"Using default LLM provider based on availability: {USE_LLM}")


def initialize_llm(llm_choice, model_name, temperature):
    """
    Initialisiert eine Instanz des LLM basierend auf der Auswahl des Anbieters, des Modellnamens
    und des Temperature-Werts. Fehlerbehandlung, wenn Parameter nicht unterstützt werden.
    Speichert die LLM-Instanz und die verwendete Konfiguration im Session State.

    Argumente:
        llm_choice (str): "OpenAI" oder "Google".
        model_name (str): Spezifischer Modellname.
        temperature (float): Gewünschter Temperature-Wert.

    Rückgabe:
        dict: Ergebnis mit Erfolgs- oder Fehlermeldung sowie LLM-/Konfigurationsinformationen.
        Gibt ein dict mit dem Schlüssel 'success' oder 'error' zurück.
    """
    # Dispose previous LLM if exists before creating a new one
    if st.session_state.chat_llm:
         st.session_state.chat_llm = None
    st.session_state.current_llm_info = None # Reset saved LLM info

    print(f"Initializing LLM: {llm_choice} - {model_name} with temperature={temperature}")

    
    llm_params = {"temperature": temperature}
    # llm_params["top_p"] = top_p_value
    # llm_params["max_tokens"] = max_tokens_value


    initialized_llm = None
    actual_config_used = {} 

    # Check if API key exists before attempting initialization
    if llm_choice == "OpenAI" and not openai_api_key:
         st.error("OpenAI API Key not found. Please set the environment variable OPENAI_API_KEY.")
         return {"error": "Missing OpenAI API Key"}
    if llm_choice == "Google" and not google_api_key:
         st.error("Google API Key not found. Please set the environment variable GOOGLE_API_KEY.")
         return {"error": "Missing Google API Key"}

    # Ensure the selected model_name is actually available for this provider
    available_models_for_choice = AVAILABLE_LLM_MODELS.get(llm_choice, [])
    if model_name not in available_models_for_choice:
         # Fallback model: Try the most common model for the provider if the selected model is not available
         # Use the first model in the available list as fallback
         fallback_model = available_models_for_choice[0] if available_models_for_choice else None

         if fallback_model:
            print(f"Warning: Model '{model_name}' for {llm_choice} not in available list. Using fallback '{fallback_model}'.")
            model_name = fallback_model
         else:
            st.error(f"Model '{model_name}' for {llm_choice} is not supported and no fallback model found.")
            return {"error": f"Unsupported or missing model for {llm_choice}: {model_name}"}


    try:
        if llm_choice == "OpenAI":
            # --- Try initializing OpenAI LLM with parameters ---
            try:
                initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, **llm_params)
                actual_config_used = llm_params # If successful, record the config used
                print(f"Initialized OpenAI LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 error_message = str(e_params).lower()
                 # Check if the error message contains keywords related to unsupported parameters
                 # OpenAI errors often have clearer error codes or messages
                 print(f"Warning: Parameter error or other issue during OpenAI init with config {llm_params}: {e_params}. Trying without specific parameters.")
                 try:
                     initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name)
                     actual_config_used = {} # Record that no custom config was used
                     print(f"Initialized OpenAI LLM: {model_name} without custom temperature")
                 except Exception as e_retry:
                      # If retry still fails, show real error
                      st.error(f"Error initializing OpenAI LLM '{model_name}' (retry without parameters): {e_retry}")
                      return {"error": f"Failed to initialize OpenAI LLM ({model_name}): {e_retry}"}


        elif llm_choice == "Google":

            try:
                 
                 initialized_llm = ChatGoogleGenerativeAI(
                     google_api_key=google_api_key,
                     model=model_name,
                     **llm_params # Pass temperature and other potential params
                     # client_options=client_options 
                 )
                 actual_config_used = llm_params
                 print(f"Initialized Google LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 # Catch more specific errors from Google GenAI if possible, or rely on the error message.
                 error_message = str(e_params).lower()
                 print(f"Warning: Parameter error or other issue during Google init with config {llm_params}: {e_params}. Trying without specific parameters.")
                 try:
                      # Retry without custom parameters
                      initialized_llm = ChatGoogleGenerativeAI(
                         google_api_key=google_api_key,
                         model=model_name,
                         # client_options=client_options 
                     )
                      actual_config_used = {}
                      print(f"Initialized Google LLM: {model_name} without custom temperature")
                 except Exception as e_retry:
                      st.error(f"Error initializing Google LLM '{model_name}' (retry without parameters): {e_retry}")
                      return {"error": f"Failed to initialize Google LLM ({model_name}): {e_retry}"}

        else:
            st.warning(f"LLM choice '{llm_choice}' is invalid...")
            return {"error": f"Invalid LLM choice: {llm_choice}"}

    except Exception as e:
       
        print(f"An unexpected error occurred during LLM initialization: {e}")
        st.error(f"An unexpected error occurred during LLM initialization: {e}")
        return {"error": f"An unexpected error occurred during LLM initialization: {e}"}

    # If initialization is successful (in any try/except branch)
    if initialized_llm:
        st.session_state.chat_llm = initialized_llm
        st.session_state.current_llm_info = {
            "llm_type": llm_choice,
            "model_name": model_name, 
            "config": actual_config_used 
        }
        st.sidebar.success(f"Initialized {llm_choice} LLM ({model_name})") # Show success message directly
        return {"success": True} 


    else:
         
         err_msg = f"Initialization failed for {llm_choice} LLM ({model_name}) without specific error."
         print(err_msg)
         st.error(err_msg) 
         return {"error": err_msg}


# --- Phase 2: Prepare Schema Info and Build Prompt ---
def generate_sql_prompt(schema_info_dict, user_query):
    """
    Prepare schema information as text and build a prompt for the LLM.
    Includes JSON structure summaries if available.

    Args:
        schema_info_dict (dict): Database schema structure from st.session_state.schema_info.
        user_query (str): User's question.

    Returns:
        LangChain Prompt Template or None, and.  an error message (str) or None.
    """
    if not schema_info_dict:
        return None, "No database schema information available to generate SQL. Please connect to the database first."

    # Bước 2.1: Prepare Schema Information as a Text String
    # This format helps the LLM easily understand the structure
    schema_text = "Database Schema:\n\n"

    for schema_name, tables in schema_info_dict.items():
        # Ignore schema if no tables are read or all tables have empty columns.

        if not tables or not isinstance(tables, dict) or all(not cols for cols in tables.values() if isinstance(cols, list)):
             print(f"  Skipping schema '{schema_name}' in prompt as it has no usable tables/columns.")
             continue

        schema_text += f"Schema: `{schema_name}`\n"

        for table_name, columns in tables.items():
            # Ignore table if no columns are read.
            if not columns or not isinstance(columns, list):
                print(f"    Skipping table '{schema_name}.{table_name}' in prompt as it has no columns info.")
                continue

            schema_text += f"  Table: `{schema_name}.{table_name}`\n" 
            schema_text += "    Column:\n"
            for col in columns:
                # column information as "name (data_type) [Properties]"
                col_info = f"    - `{col.get('name', 'N/A')}` ({col.get('type', 'N/A')})"
                details = []
                if col.get('primary_key'):
                    details.append("PK")
                
                if col.get('nullable') is False:
                    details.append("NOT NULL")
                
                
                if col.get('autoincrement'):
                     details.append("AUTO_INCREMENT") 

                if details:
                    col_info += f" [{', '.join(details)}]"

                # --- Thêm tóm tắt cấu trúc JSON nếu có ---
                if col.get('json_structure_summary'):
                     
                     json_summary_formatted = "\n".join(["      " + line for line in col['json_structure_summary'].splitlines()])
                     col_info += f"\n{json_summary_formatted}" 

                schema_text += col_info + "\n"
            schema_text += "\n" 

    if schema_text.strip() == "Database Schema:":
         return None, "No schema or usable tables/columns found to describe for the LLM."


    
    # Step 2.3: Build the Prompt using LangChain PromptTemplate
    # Using PromptTemplate makes it easier to manage prompt structure
    # SystemMessage provides instructions and database context
    # HumanMessagePromptTemplate contains the user's question


   
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a PostgreSQL AI expert, specializing in JSON data and deep technical support. Your task is to generate SQL queries based on user questions and the provided database schema. When selecting columns, use common German naming conventions. "
            "Pay special attention to columns with JSON/JSONB data types and leverage the additional schema information provided to accurately query data within JSON.\n\n"
            f"{schema_text}\n\n"

            "Please adhere to the following rules:\n"
            "1. Return only the SQL query. Do not add any explanations, extra text, or notes.\n"
            "2. Place the single SQL query within a markdown code block, starting with ```sql and ending with ```.\n"
            '''3. Always use the full schema name when referencing tables (ví dụ: `raw.haus_mieten_test`). When generating SQL statements, always enclose column names and table names in double quotes ("...") in all parts of the SQL statement, including SELECT, WHERE, FROM, or any other clause. This applies to all column names, even if they do not contain special characters, to ensure consistency and compliance with SQL standards (e.g., ANSI SQL or PostgreSQL), for example: t."column name","@id","@publishDate".\n'''
            "4. Only use tables and columns described in the provided database schema.\n"
            "5. Avoid dangerous SQL statements (such as DROP, DELETE, UPDATE, INSERT, ALTER, CREATE). Generate SELECT statements only.\n"
            "6. If the question is not related to querying data from the given tables, respond with the text: 'I cannot generate an SQL query for this request.'\n"
            "7. **Instruction Querying JSON/JSONB Data (PostgreSQL):**\n"
            "   - If you must to select an entire JSONB column for display in the result, ALWAYS cast it to TEXT to avoid type inference issues with Pandas/PyArrow.\n"
            "   - Use the `->` operator to access a key and return the result as a JSONB object/array.\n"
            "   - Use the `->>` operator to access a key and return the result as **TEXT**.\n"
            "   - To access nested keys, chain -> operators and end with ->> if you want the final value as TEXT.\n"
            "     Example: `(json_column -> 'key1' -> 'nested_key' ->> 'final_key')`\n"
            "   - Always Use `jsonb_array_elements(json_array_column)` to expand  JSON arrays into rows.\n"
            "   - Casting: When you must to use a TEXT value (from ->> or #>>) as a numeric (integer, float) or boolean type for comparisons/calculations, cast it.\n"
            "   - The casting syntax is `(json_text_expression)::target_type`. Always wrap the JSON access expression in parentheses before casting.\n"
            "   - Example for casting to numeric: `(json_column -> 'price' ->> 'value')::numeric`\n" 
            ''' - When concatenating data from two columns, pay attention to necessary quotes and parentheses. Example: (t."resultlist.realEstate" -> 'address' ->> 'street') || ', ' || (t."resultlist.realEstate" -> 'address' ->> 'city') AS "Adresse" '''
            '''8. Here is an example for querying objects within JSON data. You can refer to it to develop queries for more deeply nested or complex objects:
                SELECT "attributes"
                FROM raw.haus_kaufen
                WHERE (
                SELECT REPLACE(REPLACE(attr->>'value', '.', ''), ' €', '')::numeric
                FROM jsonb_array_elements(attributes) AS item,
                    jsonb_array_elements(item->'attribute') AS attr
                WHERE attr->>'label' = 'Kaufpreis'
                    AND attr->>'value' ~ '^[0-9\.]+ €$'  -- only get numeric values+ €
                ) > 1000000;
            '''
            '''
            - Pay attention to deeply nested combinations of both arrays and objects within JSON data types. For example, a URL (@href) might be many levels deep and pass through both arrays and objects. This requires careful consideration of array or object conditions. \n"
            - ví dụ về cấu trúc mảng/array lồng nhau: - galleryAttachments: array of objects
                                                        - attachment: array of objects
                                                            - urls: array of objects
                                                                - url: object
                                                                    - @href: string
            -- FOR EXAMPLE: Retrieving Image URLs from deeply nested JSON structures:
            SELECT
                t."realEstateId",
                url_item -> 'url' ->> '@href' AS href
            FROM
                raw.haus_kaufen AS t,
                LATERAL (
                    SELECT attachment_item
                    FROM jsonb_array_elements(t."resultlist.realEstate" -> 'galleryAttachments' -> 'attachment') AS attachment_item
                ) AS attachment_sub,
                LATERAL (
                    SELECT url_item
                    FROM jsonb_array_elements(attachment_sub.attachment_item -> 'urls') AS url_item
                    WHERE jsonb_typeof(attachment_sub.attachment_item -> 'urls') = 'array'
                ) AS url_sub
            WHERE
                jsonb_typeof(t."resultlist.realEstate" -> 'galleryAttachments' -> 'attachment') = 'array'
                AND url_item -> 'url' ->> '@href' IS NOT NULL;

            ''' 
            + "\n" 
        )),
        HumanMessagePromptTemplate.from_template("{user_query}") 
    ])

    
    return template, None 


# --- Phase 2: Call LLM, Receive Results, and Extract SQL ---

def get_sql_from_llm(user_query):
    """
    Call the initialized LLM to generate an SQL query
    based on the user's query and the loaded schema (including JSON info).

    Args:
        user_query (str): The user's question.

    Returns:
        tuple: (extracted_sql: str | None, error_msg: str | None)
    Returns the SQL query and None if successful, or None and an error message if failed.

    """
    
    if st.session_state.chat_llm is None:
         None, "LLM wurde noch nicht initialisiert. Bitte wählen und initialisieren Sie das LLM zuerst."
    if st.session_state.schema_info is None or not st.session_state.schema_info:
         None, "Datenbankschema-Informationen wurden noch nicht geladen oder sind leer. Bitte verbinden Sie zuerst die Datenbank."

    print(f"Generating prompt for query: {user_query}")
   
    # Prepare schema info and prompt
    
    prompt_template, error_msg = generate_sql_prompt(st.session_state.schema_info, user_query)

    if error_msg:
        return None, error_msg 
    if prompt_template is None:
         None, "Prompt konnte nicht erstellt werden, da keine verwendbaren Schema-Informationen vorhanden sind."


    # Format prompt with user question 
    formatted_prompt = prompt_template.format_messages(user_query=user_query)

    # print("--- Formatted Prompt sent to LLM ---")
    # for msg in formatted_prompt:
    #      # In vài ký tự đầu của nội dung để tránh tràn console
    #      # print(f"Type: {msg.type}, Content: {msg.content[:500]}...")
    #      print(f"Type: {msg.type}, Content: {msg.content[:1000]}{'...' if len(msg.content) > 1000 else ''}")
    # print("-----------------------------------")

    try:
        print(f"Calling LLM ({st.session_state.chat_llm.__class__.__name__})...") 
        
        response = st.session_state.chat_llm.invoke(formatted_prompt)
        llm_output = response.content 

        print(f"LLM Raw Output:\n{llm_output}")

        # SQL-Anweisung aus der Antwort extrahieren
        # den block code markdown ```sql ... ``` finden
        sql_block_start_tag = "```sql"
        sql_block_end_tag = "```"

        sql_start_index = llm_output.find(sql_block_start_tag)
        # Endzeichen nach einem Startzeichen finden
        sql_end_index = llm_output.find(sql_block_end_tag, sql_start_index + len(sql_block_start_tag))

        if sql_start_index != -1 and sql_end_index != -1:
            
            extracted_sql = llm_output[sql_start_index + len(sql_block_start_tag) : sql_end_index].strip()
            print(f"Extracted SQL:\n{extracted_sql}")

            # Prüfen, ob SQL ein SELECT-Befehl ist  
            # Akzeptiert SELECT, gefolgt von einer Klammer oder einem Leerzeichen.
            if not extracted_sql.strip().upper().startswith("SELECT"):
                   return None, f"Der LLM hat keinen SELECT-Befehl generiert oder konnte keinen gültigen SELECT-Befehl extrahieren. Bitte versuchen Sie es erneut oder passen Sie die Frage an.\nGenerated Query:\n```sql\n{extracted_sql}\n```"


            return extracted_sql, None 
        else:
            
            print("Could not find SQL code block in LLM output.")
           
           
            if "Ich kann für diese Anfrage keine SQL-Anweisung erstellen" in llm_output:
                 return None, llm_output 
            else:
                
                return None, f"SQL konnte nicht extrahiert werden):\n```\n{llm_output}\n```\n\n**Fehler:** Es wurde kein SQL-Codeblock (` ```sql ... ``` `) in der KI-Antwort gefunden. Bitte versuchen Sie es erneut oder passen Sie die Frage an."

    except Exception as e:
        
        print(f"Error during LLM call or processing: {e}")
        st.error(f"Fehler beim Aufruf des LLM oder bei der Verarbeitung der Antwort: {e}")
        return None, f"Fehler beim Aufruf des LLM oder bei der Verarbeitung der Antwort: {e}"


# --- Phase 4: SQL Execution ---
def execute_sql_query(engine, sql_query):
    """
    Execute a given SQL SELECT statement and return the result as a Pandas DataFrame.

    Args:
        engine: SQLAlchemy engine instance.
        sql_query (str): The SQL statement to execute.

    Returns:
        tuple: (dataframe_result: pd.DataFrame | None, error_msg: str | None)
               Returns the DataFrame and None if successful, or None and an error message if failed.
    """
    if engine is None:
        return None, "Datenbankverbindung nicht hergestellt." 

    if not sql_query or not sql_query.strip().upper().startswith("SELECT"):
        return None, "Ungültige oder keine SELECT-SQL-Abfrage zum Ausführen." 

    print(f"Executing SQL Query:\n{sql_query}")

    try:
        
        df = pd.read_sql_query(
            text(sql_query), engine
            )
        print("SQL Query executed successfully.")
        return df, None
    except OperationalError as e:
        print(f"Database operational error during query execution: {e}")
        
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        return None, f"Datenbank-Betriebsfehler beim Ausführen der Abfrage: {err_detail}" 
    except SQLAlchemyError as e:
        print(f"A SQLAlchemy error occurred during query execution: {e}")
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        return None, f"Ein Datenbankfehler ist beim Ausführen der Abfrage aufgetreten: {err_detail}" 
    except Exception as e:
        print(f"An unexpected error occurred during query execution: {e}")
        return None, f"Ein unerwarteter Fehler ist beim Ausführen der Abfrage aufgetreten: {e}"


#--------------------------------------------------------------------------------------------------------




# --- Streamlit UI ---

st.set_page_config(page_title="Text-to-SQL Chatbot - JSON-Verständnis verbessern", layout="wide") 
#st.title("Chat with Database") 
st.markdown("<h1 style='text-align: center; color: black;'>Chat with Database</h1>", unsafe_allow_html=True)

st.sidebar.header("Einstellungen") 


st.sidebar.subheader("Datenbankverbindung")


#default_uri = os.getenv("DATABASE_URI_LOCAL") 
default_uri = os.getenv("DATABASE_URI")

db_uri_input = st.sidebar.text_input(
    "Datenbank-URI eingeben:", 
    value=default_uri,
    type="password", 
    key="db_uri_input"
)

#  start the connection and schema reading process

if st.sidebar.button("Verbinden & Schema lesen", key="connect_db_button"): 
    if not db_uri_input:
        st.sidebar.warning("Bitte geben Sie die Datenbank-URI ein.") 
    else:
        st.sidebar.info("Verbinde und lese Datenbankschema...") 
        
        connect_and_load_schema(db_uri_input)
        
        # --- LLM-Auswahl und Initialisierung ---
        
        if st.session_state.chat_llm is None:

             default_temp = st.session_state.get('llm_temperature_input', 0.4)
             current_provider = st.session_state.get('llm_provider_selection', USE_LLM) 

             default_model_name = None
             if current_provider in AVAILABLE_LLM_MODELS and AVAILABLE_LLM_MODELS[current_provider]:
                  default_model_name = AVAILABLE_LLM_MODELS[current_provider][0]

             if default_model_name:
                  
                  initialize_llm(current_provider, default_model_name, default_temp)
             else:
                  st.sidebar.warning(f"Keine verfügbaren Modelle für Anbieter '{current_provider}' gefunden. Bitte wählen und initialisieren Sie das LLM manuell.") # Dịch



st.sidebar.subheader("LLM auswählen") 

llm_provider_options = list(AVAILABLE_LLM_MODELS.keys())

initial_llm_provider_index = llm_provider_options.index(USE_LLM) if USE_LLM in llm_provider_options else 0

llm_provider = st.sidebar.selectbox(
    "LLM-Anbieter auswählen:", 
    llm_provider_options,
    index=initial_llm_provider_index,
    key="llm_provider_selection", 
    
)

model_options = AVAILABLE_LLM_MODELS.get(llm_provider, []) 


current_llm_info_state = st.session_state.get('current_llm_info')

# identify current_model and current_temp 
if isinstance(current_llm_info_state, dict):
    current_model = current_llm_info_state.get('model_name')
    current_temp = current_llm_info_state.get('config', {}).get('temperature', 0.4)
else:
    
    current_model = None 
    
    current_temp = st.session_state.get('llm_temperature_input', 0.4)


# Logic to choose initial_model_index for selectbox model:
initial_model_index = 0

if current_model and current_model in model_options:
     initial_model_index = model_options.index(current_model)

elif st.session_state.get('llm_model_selection') in model_options:
    initial_model_index = model_options.index(st.session_state.llm_model_selection)

elif len(model_options) > 0:
    default_first_model_in_list = model_options[0]
    initial_model_index = model_options.index(default_first_model_in_list) if default_first_model_in_list in model_options else 0 
else:
    
     initial_model_index = 0 


selected_model = st.sidebar.selectbox(
    "LLM-Modell auswählen:", 
    model_options,
    index=initial_model_index,
    key="llm_model_selection" 
)

# --- Add custom UI for Temperature (slider) ---


st.markdown("""
<style>
/* --- Style for the FILLED part of the slider track --- */
div[data-testid="stSlider"] div.st-ds {
    background-color: #1E90FF !important; /* Deep Sky Blue */
}

/* --- Style for the SLIDER THUMB --- */
div[data-testid="stSlider"] div.st-dg { /* class may be st-dj or st-dg depending on version */
    background-color: #1E90FF !important; /* Deep Sky Blue */
    border-color: #1E90FF !important;
}
</style>
""", unsafe_allow_html=True)


selected_temperature = st.sidebar.slider(
    "Temperatur:", 
    min_value=0.0, 
    max_value=2.0, 
    value=current_temp,     
    step=0.01,     
    format="%.2f", 
    key="llm_temperature_input" 
)



if st.sidebar.button("LLM initialisieren/wechseln", key="init_llm_button"): 
     if selected_model:
         initialize_llm(llm_provider, selected_model, selected_temperature)
     else:
         st.sidebar.warning("Bitte wählen Sie ein LLM-Modell aus.") 



st.sidebar.subheader("Status") 
if st.session_state.db_engine:
    st.sidebar.success("Datenbank verbunden") 
else:
    st.sidebar.warning("Datenbank nicht verbunden") 


current_llm_info_display = st.session_state.get('current_llm_info')
if isinstance(current_llm_info_display, dict):
    llm_info = current_llm_info_display
    config_str = ", ".join([f"{k}={v}" for k, v in llm_info.get('config', {}).items()])
    status_text = f"LLM bereit ({llm_info.get('llm_type', 'N/A')}: {llm_info.get('model_name', 'N/A')})" 
    if config_str:
        status_text += f" mit Konfiguration: {config_str}" 
    st.sidebar.success(status_text)
else:
    st.sidebar.warning("LLM nicht initialisiert") 


# --- PHẦN UI MỚI: HIỂN THỊ TỔNG QUAN VÀ XEM TRƯỚC DỮ LIỆU ---
st.subheader("Datenbank-Übersichte") 

if st.session_state.get('schema_info') and isinstance(st.session_state.schema_info, dict):

    DATABASE_OVERVIEW_TEXT = """

   Entdecken Sie mit mir den deutschen Immobilienmarkt. Diese Datenbank wird mit Daten von **immobilienscout24.de** gespeist und bietet Ihnen tiefe Einblicke in Kauf- und Mietobjekte.

**Was dieser Chatbot für Sie tun kann:**

*   🗣️ **Einfach chatten:** Stellen Sie Ihre Fragen in natürlicher Sprache. Ich finde die passenden Immobilien für Sie, ohne dass Sie technisches Wissen benötigen.
    
*   🤖 **SQL-Frage (Text-to-SQL):** Haben Sie komplexe Anforderungen? Beschreiben Sie, was Sie suchen, und ich erstelle für Sie die passende SQL-Abfrage. Filtern und kombinieren Sie Bedingungen wie Preis, Zimmeranzahl, Postleitzahl (PLZ), Fläche, Maklergebühren und mehr.

*   📊 **Chat-to-Visualization (Demnächst verfügbar):** Verwandeln Sie trockene Daten in anschauliche Diagramme und Grafiken. Diese Funktion hilft Ihnen, Markttrends und Vergleiche auf einen Blick zu erfassen.

*   📋 **Datenvorschau (unten):** Die folgenden Registerkarten zeigen eine Vorschau der ersten 10 Zeilen jeder Tabelle. So bekommen Sie ein besseres Gefühl für die Daten und die Struktur, mit der wir arbeiten.

**Legen Sie los! Stellen Sie Ihre Frage unten.**
    """
    st.info(DATABASE_OVERVIEW_TEXT)

   
    if st.session_state.get('data_previews') and isinstance(st.session_state.data_previews, dict):
        previews_to_display = st.session_state.data_previews
        if previews_to_display:
            
            tab_names = list(previews_to_display.keys())
            
            
            tabs = st.tabs(tab_names)

            
            for tab, table_name in zip(tabs, tab_names):
                with tab:
                    st.markdown(f"**Vorschau der ersten 10 Zeilen von `{table_name}`**")
                    df_preview = previews_to_display[table_name]
                    
                    
                    st.dataframe(df_preview, use_container_width=True, height=350) 
        else:
            st.warning("Keine Datenvorschau verfügbar. Möglicherweise enthalten die Tabellen keine Daten.") # Thông báo nếu không có preview
    else:
 
        st.warning("Datenvorschau wird geladen oder konnte nicht abgerufen werden.")

else:
   
    st.info("Bitte geben Sie die Datenbank-URI ein und klicken Sie auf 'Verbinden & Schema lesen', um zu beginnen.")

# --- Create SQL (UI for User Query)  ---
st.subheader("SQL - Frage") 

# Check Conditions
if st.session_state.get('schema_info') is None or not st.session_state.schema_info:
    st.warning("Bitte verbinden Sie die Datenbank und lesen Sie das Schema, bevor Sie fragen.") 
elif st.session_state.get('chat_llm') is None:
    st.warning("Bitte initialisieren Sie das LLM, bevor Sie fragen.") 
else:
    user_query = st.text_area(
        "Geben Sie Ihre Frage zu den Daten ein (z.B.: 'haus kaufen, würzburg, display: id, title, plz, zimmeranzahl, preis'):", # Dịch
        key="user_query_input", 
        height=100
    )

    if st.button("SQL-Abfrage generieren", key="generate_sql_button"): 
        if not user_query:
            st.warning("Bitte geben Sie eine Frage ein.") 
        else:
            #st.info("Generiere SQL-Abfrage mit LLM...") 
            st.session_state.generated_sql = None

            
            generated_sql, error_msg = get_sql_from_llm(user_query) 

            if error_msg:
                st.error(error_msg)
                st.session_state.generated_sql = None 
            else:
                st.session_state.generated_sql = generated_sql 

    # --- immer display existed SQL ---
    if st.session_state.generated_sql:
        st.success("SQL-Abfrage generiert:") 
        st.code(st.session_state.generated_sql, language="sql")






# --- Phase 4: Executing SQL ---
# --- A. SQL Execution and Raw Data Display ---
st.subheader("SQL ausführen und Ergebnisse anzeigen")

if st.session_state.get('generated_sql'):
    if st.session_state.db_engine:
        if st.button("SQL ausführen", key="execute_sql_button"):
            with st.spinner("Führe SQL-Abfrage aus..."):
                # Reset all analysis-related states for a new query
                st.session_state.query_result_df = None
                st.session_state.query_error = None
                st.session_state.viz_chat_history = []
                st.session_state.dashboard_items = []
                st.session_state.original_dashboard_df = None
                st.session_state.last_fig = None # Ensure this is also cleared
                st.session_state.analytical_summary = None

                result_df, error_msg = execute_sql_query(
                    st.session_state.db_engine,
                    st.session_state.generated_sql
                )

                if error_msg:
                    st.session_state.query_error = error_msg
                else:
                    st.session_state.query_result_df = result_df
                    st.session_state.original_dashboard_df = result_df.copy() # Save original df for dashboard filters
            
            # Show status and then generate summary
            if st.session_state.query_result_df is not None:
                st.success("SQL-Abfrage erfolgreich ausgeführt!")
                if not st.session_state.query_result_df.empty:
                    with st.spinner("Analysiere Daten..."):
                        st.session_state.analytical_summary = get_analytical_summary(st.session_state.query_result_df)
            else:
                st.error(st.session_state.query_error)


    else:
        st.warning("Datenbank nicht verbunden. Bitte verbinden Sie die Datenbank, um SQL auszuführen.")
else:
     st.info("Generieren Sie oben eine SQL-Abfrage, um sie auszuführen.")

# Always display the data table if it exists
if st.session_state.query_result_df is not None:
    df = st.session_state.query_result_df
    num_rows = df.shape[0]
    st.markdown(f"##### **Abfrageergebnisse:** Total {num_rows} rows")
    
    if not df.empty:

        st.dataframe(df, use_container_width=True)
    else:
        st.info("Die Abfrage hat keine Ergebnisse zurückgegeben.")

    # Display the analytical summary
    if st.session_state.analytical_summary:
        with st.expander("Automatische Datenanalyse & Einblicke"):
            st.markdown(st.session_state.analytical_summary)

# Display error from the last execution attempt
elif st.session_state.query_error:
    st.error(st.session_state.query_error)


# --- B. Display the Analysis Tabs if there is data ---
if st.session_state.query_result_df is not None and not st.session_state.query_result_df.empty:
    df = st.session_state.query_result_df
    
    st.markdown("---")
    st.header("Analyse & Visualisierung")

    # Initialize session states for the tabs if they don't exist
    if "viz_chat_history" not in st.session_state:
        st.session_state.viz_chat_history = []
    if "dashboard_items" not in st.session_state:
        st.session_state.dashboard_items = []

    tab1, tab2 = st.tabs(["📊 Diagramm erstellen", "📋 Dashboard"])

    # --- TAB 1: Create Chart via Chat ---
    with tab1:
        st.subheader("Erstellen Sie ein Diagramm per Chat")
        
        # Button to clear the analysis chat
        if st.button("Neue Analyse starten 🔄", key="clear_viz_chat_button"):
            st.session_state.viz_chat_history = []
            st.rerun()

        # Display the entire chat history, including charts
        for i, message in enumerate(st.session_state.viz_chat_history):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
                # If the assistant message has a figure, display it
                if role == "assistant" and "figure" in message.additional_kwargs:
                    fig_to_show = message.additional_kwargs["figure"]
                    st.plotly_chart(fig_to_show, use_container_width=True)
                    
                    # --- NEW LOGIC: "Add to Dashboard" button for the LAST message only ---
                    # Check if this is the last message in the history
                    is_last_message = (i == len(st.session_state.viz_chat_history) - 1)
                    
                    if is_last_message and fig_to_show:
                        # Use columns to align the button to the right
                        _, col_btn = st.columns([4, 1]) # Adjust ratio if needed
                        with col_btn:
                            title = fig_to_show.layout.title.text if fig_to_show.layout.title else "Diagramm"
                            if st.button("Zum Dashboard hinzufügen", key=f"add_to_dashboard_btn_{i}"):
                                st.session_state.dashboard_items.append({"title": title, "figure": fig_to_show})
                                st.toast(f"Diagramm '{title}' zum Dashboard hinzugefügt!", icon="✅")


        # The chat input is the LAST element
        if prompt := st.chat_input("Was möchten Sie visualisieren?"):
            # Append user message to history
            st.session_state.viz_chat_history.append(HumanMessage(content=prompt))
            
            # Rerun to display the user's message immediately
            st.rerun()

        # Check if the last message is from the user, then process it
        if st.session_state.viz_chat_history and isinstance(st.session_state.viz_chat_history[-1], HumanMessage):
            last_user_prompt = st.session_state.viz_chat_history[-1].content
            
            with st.chat_message("assistant"):
                with st.spinner("Assistent erstellt Diagramm..."):
                    fig, error_msg = generate_and_execute_chart_code(last_user_prompt, df, st.session_state.analytical_summary)
                    
                    if error_msg:
                        response_content = error_msg
                        st.session_state.viz_chat_history.append(AIMessage(content=response_content))
                    else:
                        title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Unbenanntes Diagramm"
                        response_content = f"Hier ist das Diagramm für:"
                        # Append assistant's response WITH the figure
                        st.session_state.viz_chat_history.append(AIMessage(
                            content=response_content,
                            additional_kwargs={"figure": fig}
                        ))
            
            # Rerun to display the assistant's new message and chart
            st.rerun()

    # --- TAB 2: Dashboard ---
    with tab2:
        st.subheader("Dein Dashboard")
        
        if not st.session_state.dashboard_items:
            st.info("Ihr Dashboard ist leer.")
        else:
            if st.button("Dashboard leeren", key="clear_dashboard_btn"):
                st.session_state.dashboard_items = []
                st.rerun()
            st.markdown("---")
            
            dashboard_cols = st.columns(2)
            for i, item in enumerate(st.session_state.dashboard_items):
                with dashboard_cols[i % 2]:
                    with st.container(border=True):
                        #st.markdown(f"**{item.get('title', '')}**")
                        st.plotly_chart(item['figure'], use_container_width=True, key=f"dash_chart_{i}")