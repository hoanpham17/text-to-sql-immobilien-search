import pandas as pd
import streamlit as st # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import json # Cần import json để xử lý dữ liệu JSON mẫu

# Import các lớp cần thiết từ SQLAlchemy cho kết nối và đọc schema
from sqlalchemy import create_engine, inspect, text # type: ignore
from sqlalchemy.exc import SQLAlchemyError, OperationalError # type: ignore
# SỬA LỖI Ở ĐÂY: Import JSON/JSONB từ dialect postgresql
from sqlalchemy.dialects.postgresql import JSON, JSONB # Import kiểu dữ liệu JSON/JSONB từ dialect PostgreSQL


# Import các lớp cần thiết từ LangChain Core và các tích hợp LLM
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore
from langchain_core.prompts import HumanMessagePromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore

# --- Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Cấu hình lựa chọn LLM mặc định ---
USE_LLM = os.getenv("USE_LLM", "Google")

# --- Folder Creation (Giữ nguyên) ---
folders_to_create = ['csvs']
for folder_name in folders_to_create:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


# --- Quản lý trạng thái trong Streamlit Session State ---
# Khởi tạo các key trong session state
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = None
if 'chat_llm' not in st.session_state:
    st.session_state.chat_llm = None
if 'generated_sql' not in st.session_state:
     st.session_state.generated_sql = None
if 'current_llm_info' not in st.session_state:
     st.session_state.current_llm_info = None # Lưu thông tin LLM đã khởi tạo và config
# --- THÊM KEY MỚI CHO GIAI ĐOẠN 4 ---
if 'query_result_df' not in st.session_state: # Dòng này đã được thêm
     st.session_state.query_result_df = None
if 'query_error' not in st.session_state: # Dòng này đã được thêm
     st.session_state.query_error = None



# --- Giai đoạn 1: Kết nối Database và đọc Schema (Đã thêm logic phân tích JSON mẫu) ---



def connect_and_load_schema(db_uri):
    """
    Kết nối đến database, đọc schema, và phân tích cấu trúc mẫu
    cho các cột JSON/JSONB.
    """
    # Dispose LLM cũ nếu có trước khi tạo mới
    if st.session_state.db_engine:
        try:
            st.session_state.db_engine.dispose()
            print("Existing DB engine disposed.")
        except Exception as e:
             print(f"Error disposing existing engine: {e}")

    st.session_state.db_engine = None
    st.session_state.schema_info = None
    st.session_state.generated_sql = None
    # Khi kết nối lại DB, reset cả thông tin LLM vì có thể schema ảnh hưởng đến prompt/model
    st.session_state.chat_llm = None
    st.session_state.current_llm_info = None


    engine = None
    try:
        print(f"Attempting to create database engine for URI: {db_uri}")
        # Thêm connect_args nếu cần cho một số loại DB hoặc cấu hình đặc biệt
        # engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600, connect_args={"options": "-c timezone=UTC"})
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)


        print("Testing database connection...")
        with engine.connect() as connection:
             # Sử dụng text() cho các câu lệnh SQL đơn giản cũng là cách tốt hơn
             connection.execute(text("SELECT 1"))
        print("Database connection successful!")

        print("Reading database schema...")
        inspector = inspect(engine)

        all_schemas = inspector.get_schema_names()
        print(f"Found schemas: {all_schemas}")

        database_schema = {}
        SAMPLE_JSON_ROWS_LIMIT = 5 # Số lượng bản ghi JSON mẫu để phân tích

        for schema in all_schemas:
            # Bỏ qua các schema hệ thống phổ biến
            system_schemas = ['information_schema', 'pg_catalog', 'mysql', 'sys', 'performance_schema', 'temp_tables', 'information_schema']
            if schema.lower() in [s.lower() for s in system_schemas]: # So sánh không phân biệt hoa thường
                print(f"  Skipping system schema: {schema}")
                continue
            # Có thể thêm logic để bỏ qua các schema rỗng hoặc không có bảng nào

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

                         # --- PHẦN MỚI: Phân tích cấu trúc JSON mẫu ---
                         for col in columns:
                             # Kiểm tra xem cột có kiểu JSON hoặc JSONB không
                             # Sử dụng isinstance để kiểm tra kiểu dữ liệu SQLAlchemy
                             # HOẶc kiểm tra chuỗi tên kiểu (đề phòng import dialect thất bại hoặc DB khác)
                             # Sử dụng isinstance là tốt nhất nếu import được kiểu cụ thể
                             is_json_type = isinstance(col['type'], (JSON, JSONB)) or 'JSON' in str(col['type']).upper()

                             if is_json_type:
                                 print(f"          Column '{col['name']}' is JSON type. Attempting to sample data...")
                                 json_summary = ""
                                 try:
                                     # Query an toàn để lấy mẫu dữ liệu JSON
                                     # Sử dụng text() và dấu ngoặc kép cho tên đối tượng DB là cách tốt cho PostgreSQL
                                     # Đảm bảo tên schema/bảng/cột không chứa '";' để tránh SQL Injection cơ bản ở đây
                                     # (Mặc dù `text()` thường an toàn hơn)
                                     # sql = text(f'SELECT "{col["name"]}" FROM "{schema}"."{table}" WHERE "{col["name"]}" IS NOT NULL LIMIT {SAMPLE_JSON_ROWS_LIMIT};')
                                     # Cách an toàn hơn nữa là dùng Parameter Binding nếu tên cột/bảng là biến, nhưng ở đây chúng ta lấy từ schema info, tạm chấp nhận f-string + quoting
                                     sql_query_str = f'SELECT "{col["name"]}" FROM "{schema}"."{table}" WHERE "{col["name"]}" IS NOT NULL LIMIT {SAMPLE_JSON_ROWS_LIMIT};'
                                     sample_sql = text(sql_query_str)


                                     with engine.connect() as connection:
                                         sample_data = connection.execute(sample_sql).fetchall()

                                     if sample_data:
                                         # Phân tích cấu trúc mẫu đơn giản
                                         sample_summary_lines = []
                                         common_keys = {}
                                         is_potential_array_of_objects = True
                                         found_array = False
                                         found_object = False

                                         for i, row in enumerate(sample_data):
                                             json_value = row[0] # Giá trị đã được SQLAlchemy convert sang Python object (dict/list)
                                             if json_value is None: continue

                                             # Tóm tắt 1-2 ví dụ
                                             if i < 2:
                                                  try:
                                                       # Thử dumps với indent cho dễ đọc, cắt bớt
                                                       # ensure_ascii=False để hiển thị tiếng Việt
                                                       sample_str = json.dumps(json_value, ensure_ascii=False, indent=2)
                                                       sample_summary_lines.append(f"  - Example {i+1}: {sample_str[:300]}{'...' if len(sample_str) > 300 else ''}")
                                                  except Exception as e:
                                                       # Fallback nếu dumps thất bại (ví dụ: kiểu dữ liệu không JSON-serializable)
                                                       sample_summary_lines.append(f"  - Example {i+1}: {str(json_value)[:300]}{'...' if len(str(json_value)) > 300 else ''} (Serialization Error: {e})")


                                             # Phân tích key phổ biến nếu là list/dict
                                             if isinstance(json_value, list):
                                                  found_array = True
                                                  # Giả định đây là mảng các object, nếu không phải, đánh dấu lại
                                                  if not all(isinstance(item, dict) or item is None for item in json_value): # Kiểm tra item is not None cũng
                                                       is_potential_array_of_objects = False

                                                  for item in json_value:
                                                       if isinstance(item, dict):
                                                            for key in item:
                                                                 common_keys[key] = common_keys.get(key, 0) + 1
                                             elif isinstance(json_value, dict):
                                                  found_object = True
                                                  is_potential_array_of_objects = False # Nếu tìm thấy object trực tiếp, không phải mảng object
                                                  for key in json_value:
                                                       common_keys[key] = common_keys.get(key, 0) + 1
                                             else:
                                                 is_potential_array_of_objects = False # Không phải list/dict


                                         # Xây dựng chuỗi tóm tắt cấu trúc
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
                                     # Hiển thị lỗi chi tiết hơn nếu có orig
                                     err_detail = str(json_sample_exc)
                                     if hasattr(json_sample_exc, 'orig') and json_sample_exc.orig is not None:
                                          err_detail += f" (Original: {json_sample_exc.orig})"
                                     col['json_structure_summary'] = f"  - Could not sample JSON data: {err_detail}\n" # Lưu thông báo lỗi vào summary


                         # --- END PHẦN MỚI ---

                     except Exception as col_exc:
                         print(f"        Could not get columns for {schema}.{table}: {col_exc}")
                         database_schema[schema][table] = []

            except Exception as table_list_exc:
                print(f"    Could not get tables for schema {schema}: {table_list_exc}")
                # Nếu không đọc được bảng, loại bỏ schema khỏi database_schema
                # del database_schema[schema] # Hoặc giữ lại tên schema nhưng với dict rỗng {}


        # Sau khi đọc xong tất cả, kiểm tra lại database_schema để loại bỏ các schema rỗng (không có bảng hoặc không có cột)
        schemas_to_remove = [s for s, tables in database_schema.items() if not tables or all(not cols for cols in tables.values())]
        for s in schemas_to_remove:
             print(f"  Removing empty or column-less schema '{s}' from schema info.")
             del database_schema[s]


        st.session_state.db_engine = engine
        st.session_state.schema_info = database_schema

        if not st.session_state.schema_info:
            # Nếu không còn schema nào sau khi lọc
            warning_msg = "Connection successful, but no usable schemas/tables found (excluding system schemas and empty ones)."
            print(warning_msg)
            st.warning(warning_msg)
            return {"message": warning_msg, "warning": True}


        print("Database schema loaded successfully!")
        success_msg = "Connection successful and schema loaded!"
        st.success(success_msg) # Hiển thị success message trực tiếp ở đây sau khi tất cả logic xong
        return {"message": success_msg} # Không cần trả về success/error key nữa, chỉ cần return dict


    except OperationalError as e:
        print(f"Database connection failed: {e}")
        if engine:
             engine.dispose()
        # Log thêm chi tiết nếu có trong OperationalError (ví dụ: lý do từ DB driver)
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        st.error(f"Lỗi kết nối database: {err_detail}")
        return {"error": f"Database connection failed: {err_detail}"}
    except SQLAlchemyError as e:
        print(f"A SQLAlchemy error occurred: {e}")
        if engine:
             engine.dispose()
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        st.error(f"Lỗi SQLAlchemy: {err_detail}")
        return {"error": f"A database error occurred: {err_detail}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if engine:
             engine.dispose()
        st.error(f"Lỗi không xác định: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


# --- Giai đoạn 3.1: Định nghĩa danh sách các mô hình khả dụng (Giữ nguyên) ---
AVAILABLE_LLM_MODELS = {
    "OpenAI": [
        "gpt-3.5-turbo",
        "o3",
        "o3-mini",
        "o4-mini", # Có context dài hơn
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o", # Mô hình mới nhất, thường tốt nhất
        "o1-mini",
        # Có thể thêm các model khác nếu cần
    ],
    "Google": [
        "gemini-2.5-flash", # Có context rất dài
        "gemini-2.5-pro", # Mô hình text chính
        "gemini-2.5-flash-lite-preview-06-17", # Nhanh hơn, rẻ hơn
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash", # Nếu cần xử lý ảnh (không cần cho Text-to-SQL)
        "gemini-1.5-pro" 
        # Có thể thêm các model khác nếu cần
    ]
    # Có thể thêm các nhà cung cấp khác như "Anthropic", "Mistral" vào đây
}
# Cập nhật USE_LLM nếu giá trị cũ không còn trong AVAILABLE_LLM_MODELS
if USE_LLM not in AVAILABLE_LLM_MODELS or not AVAILABLE_LLM_MODELS[USE_LLM]:
     # Tìm nhà cung cấp có model đầu tiên khả dụng, hoặc mặc định là Google
     found_provider = False
     for provider, models in AVAILABLE_LLM_MODELS.items():
          if models:
               USE_LLM = provider
               found_provider = True
               break
     if not found_provider: # Nếu không có model nào cả trong bất kỳ provider nào
         USE_LLM = "Google" # Fallback cuối cùng (có thể sẽ báo lỗi sau nếu Google cũng không có model)
     print(f"Using default LLM provider based on availability: {USE_LLM}")

# --- Giai đoạn 2 & 3.3 & 3.5: Lựa chọn và Khởi tạo LLM (Đã thêm xử lý lỗi tham số và nhận temperature) ---
def initialize_llm(llm_choice, model_name, temperature):
    """
    Khởi tạo instance của LLM dựa trên lựa chọn nhà cung cấp, tên mô hình,
    và giá trị temperature. Xử lý lỗi khi tham số không được hỗ trợ.
    Lưu LLM instance và cấu hình đã dùng vào session state.

    Args:
        llm_choice (str): "OpenAI" hoặc "Google".
        model_name (str): Tên mô hình cụ thể.
        temperature (float): Giá trị temperature mong muốn.

    Returns:
        dict: Kết quả chứa thông báo thành công hoặc lỗi, và thông tin LLM/config.
        Trả về dict với key 'success' hoặc 'error'.
    """
    # Dispose LLM cũ nếu có trước khi tạo mới
    if st.session_state.chat_llm:
         st.session_state.chat_llm = None
    st.session_state.current_llm_info = None # Reset thông tin LLM đã lưu

    print(f"Initializing LLM: {llm_choice} - {model_name} with temperature={temperature}")

    # Cấu hình các tham số cho LLM
    llm_params = {"temperature": temperature}
    # Có thể thêm các tham số khác mà bạn muốn cho phép tùy chỉnh sau này (top_p, max_tokens...)
    # llm_params["top_p"] = top_p_value
    # llm_params["max_tokens"] = max_tokens_value


    initialized_llm = None
    actual_config_used = {} # Dictionary để lưu cấu hình thực tế đã dùng

    # Kiểm tra xem API key có tồn tại không trước khi thử khởi tạo
    if llm_choice == "OpenAI" and not openai_api_key:
         st.error("OpenAI API Key không tìm thấy. Vui lòng đặt biến môi trường OPENAI_API_KEY.")
         return {"error": "Missing OpenAI API Key"}
    if llm_choice == "Google" and not google_api_key:
         st.error("Google API Key không tìm thấy. Vui lòng đặt biến môi trường GOOGLE_API_KEY.")
         return {"error": "Missing Google API Key"}

    # Đảm bảo model_name được chọn thực sự có trong list khả dụng cho provider này
    available_models_for_choice = AVAILABLE_LLM_MODELS.get(llm_choice, [])
    if model_name not in available_models_for_choice:
         # Fallback model: Thử model phổ biến nhất của provider nếu model được chọn không có
         # Tìm model đầu tiên trong list khả dụng làm fallback
         fallback_model = available_models_for_choice[0] if available_models_for_choice else None

         if fallback_model:
            print(f"Warning: Model '{model_name}' for {llm_choice} not in available list. Using fallback '{fallback_model}'.")
            model_name = fallback_model
         else:
            st.error(f"Mô hình '{model_name}' cho {llm_choice} không hỗ trợ và không tìm thấy mô hình thay thế.")
            return {"error": f"Unsupported or missing model for {llm_choice}: {model_name}"}


    try:
        if llm_choice == "OpenAI":
            # --- Thử khởi tạo OpenAI LLM với các tham số ---
            try:
                initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, **llm_params)
                actual_config_used = llm_params # Nếu thành công, ghi lại config đã dùng
                print(f"Initialized OpenAI LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 error_message = str(e_params).lower()
                 # Kiểm tra thông báo lỗi có chứa từ khóa liên quan đến tham số không hỗ trợ không
                 # Các lỗi OpenAI thường có mã lỗi hoặc thông báo rõ ràng hơn
                 print(f"Warning: Parameter error or other issue during OpenAI init with config {llm_params}: {e_params}. Trying without specific parameters.")
                 try:
                     initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name)
                     actual_config_used = {} # Ghi lại rằng không dùng config tùy chỉnh
                     print(f"Initialized OpenAI LLM: {model_name} without custom temperature")
                 except Exception as e_retry:
                      # Nếu thử lại vẫn lỗi, báo lỗi thật
                      st.error(f"Lỗi khi khởi tạo OpenAI LLM '{model_name}' (thử lại không tham số): {e_retry}")
                      return {"error": f"Failed to initialize OpenAI LLM ({model_name}): {e_retry}"}


        elif llm_choice == "Google":

            try:
                 # Google API might need specific regional endpoints for some models/features.
                 # from google.api_core.client_options import ClientOptions
                 # client_options = ClientOptions(api_endpoint=f"us-central1-generativelanguage.googleapis.com") # Example endpoint

                 initialized_llm = ChatGoogleGenerativeAI(
                     google_api_key=google_api_key,
                     model=model_name,
                     **llm_params # Pass temperature and other potential params
                     # client_options=client_options # Uncomment if needed
                 )
                 actual_config_used = llm_params
                 print(f"Initialized Google LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 # Bắt các lỗi cụ thể hơn từ Google GenAI nếu có thể, hoặc dựa vào thông báo lỗi.
                 error_message = str(e_params).lower()
                 print(f"Warning: Parameter error or other issue during Google init with config {llm_params}: {e_params}. Trying without specific parameters.")
                 try:
                      # Thử lại không có các tham số tùy chỉnh
                      initialized_llm = ChatGoogleGenerativeAI(
                         google_api_key=google_api_key,
                         model=model_name,
                         # client_options=client_options # Uncomment if needed
                     )
                      actual_config_used = {}
                      print(f"Initialized Google LLM: {model_name} without custom temperature")
                 except Exception as e_retry:
                      st.error(f"Lỗi khi khởi tạo Google LLM '{model_name}' (thử lại không tham số): {e_retry}")
                      return {"error": f"Failed to initialize Google LLM ({model_name}): {e_retry}"}

        else:
            st.warning(f"Lựa chọn LLM '{llm_choice}' không hợp lệ...")
            return {"error": f"Invalid LLM choice: {llm_choice}"}

    except Exception as e:
        # Bắt các lỗi khác không liên quan đến cấu hình ban đầu (ví dụ: lỗi API key sai, lỗi mạng)
        # Langchain thường wrap các lỗi từ provider API, nên e có thể là lỗi từ langchain hoặc lỗi gốc
        print(f"An unexpected error occurred during LLM initialization: {e}")
        st.error(f"Lỗi không xác định khi khởi tạo LLM: {e}")
        return {"error": f"An unexpected error occurred during LLM initialization: {e}"}

    # Nếu khởi tạo thành công (trong bất kỳ nhánh try/except nào)
    if initialized_llm:
        st.session_state.chat_llm = initialized_llm
        st.session_state.current_llm_info = {
            "llm_type": llm_choice,
            "model_name": model_name, # Lưu tên model đã chọn ban đầu (có thể là fallback)
            "config": actual_config_used # Lưu cấu hình thực tế đã sử dụng
        }
        st.sidebar.success(f"Initialized {llm_choice} LLM ({model_name})") # Hiển thị success message trực tiếp
        return {"success": True} # Chỉ trả về success


    else:
         # Trường hợp này không nên xảy ra nếu logic try/except đúng, nhưng thêm vào để an toàn
         err_msg = f"Initialization failed for {llm_choice} LLM ({model_name}) without specific error."
         print(err_msg)
         st.error(err_msg) # Hiển thị lỗi
         return {"error": err_msg}


# --- Giai đoạn 2: Chuẩn bị Schema Info và Xây dựng Prompt (Đã cập nhật để thêm JSON summary) ---
def generate_sql_prompt(schema_info_dict, user_query):
    """
    Chuẩn bị thông tin schema thành văn bản và xây dựng prompt cho LLM.
    Bao gồm cả tóm tắt cấu trúc JSON nếu có.

    Args:
        schema_info_dict (dict): Cấu trúc schema database từ st.session_state.schema_info.
        user_query (str): Câu hỏi của người dùng.

    Returns:
        LangChain Prompt Template hoặc None, và thông báo lỗi (str) hoặc None.
    """
    if not schema_info_dict:
        return None, "Không có thông tin schema database để tạo SQL. Vui lòng kết nối DB trước."

    # Bước 2.1: Chuẩn bị thông tin Schema thành chuỗi văn bản
    # Format này giúp LLM dễ hiểu cấu trúc
    schema_text = "Database Schema:\n\n"

    for schema_name, tables in schema_info_dict.items():
        # Bỏ qua schema nếu không có bảng nào được đọc hoặc tất cả các bảng đều trống cột
        # Kiểm tra lại điều kiện này để tránh lỗi nếu tables là None (trường hợp không mong muốn)
        if not tables or not isinstance(tables, dict) or all(not cols for cols in tables.values() if isinstance(cols, list)):
             print(f"  Skipping schema '{schema_name}' in prompt as it has no usable tables/columns.")
             continue

        schema_text += f"Schema: `{schema_name}`\n"

        for table_name, columns in tables.items():
            # Bỏ qua bảng nếu không có cột nào được đọc
            if not columns or not isinstance(columns, list):
                print(f"    Skipping table '{schema_name}.{table_name}' in prompt as it has no columns info.")
                continue

            schema_text += f"  Bảng: `{schema_name}.{table_name}`\n" # Luôn dùng cú pháp schema.table
            schema_text += "    Cột:\n"
            for col in columns:
                # Định dạng thông tin cột: tên (kiểu dữ liệu) [Các thuộc tính]
                col_info = f"    - `{col.get('name', 'N/A')}` ({col.get('type', 'N/A')})"
                details = []
                if col.get('primary_key'):
                    details.append("PK")
                # Kiểm tra explicit False vì giá trị có thể là None (không biết)
                if col.get('nullable') is False:
                    details.append("NOT NULL")
                # Kiểm tra nếu có giá trị default và hiển thị
                if col.get('default') is not None:
                    default_val = str(col.get('default'))
                    # Cắt bớt nếu giá trị default quá dài để tránh làm prompt quá tải
                    if len(default_val) > 50:
                         default_val = default_val[:50] + "..."
                    details.append(f"DEFAULT '{default_val}'")
                # Thuộc tính auto increment
                if col.get('autoincrement'):
                     details.append("AUTO_INCREMENT") # Hoặc SERIAL cho Postgres

                if details:
                    col_info += f" [{', '.join(details)}]"

                # --- PHẦN MỚI: Thêm tóm tắt cấu trúc JSON nếu có ---
                if col.get('json_structure_summary'):
                     # Thêm tóm tắt cấu trúc JSON xuống dòng mới và thụt lề vào mô tả cột
                     # Dùng khoảng trắng để thụt lề giống mức của gạch đầu dòng cột
                     json_summary_formatted = "\n".join(["      " + line for line in col['json_structure_summary'].splitlines()])
                     col_info += f"\n{json_summary_formatted}" # Nối thêm tóm tắt JSON đã format

                schema_text += col_info + "\n"
            schema_text += "\n" # Khoảng trống giữa các bảng

    # Kiểm tra lại schema_text sau khi lặp, đề phòng trường hợp schema_info_dict không rỗng nhưng không có schema/bảng/cột khả dụng nào được thêm vào text
    if schema_text.strip() == "Database Schema:":
         return None, "Không tìm thấy schema hoặc bảng/cột khả dụng để mô tả cho LLM."


    # print("--- Schema Text for Prompt ---")
    # print(schema_text)
    # print("----------------------------")

    # Bước 2.3: Xây dựng Prompt bằng LangChain PromptTemplate
    # Sử dụng PromptTemplate giúp dễ dàng quản lý cấu trúc prompt
    # SystemMessage hướng dẫn vai trò và context DB
    # HumanMessagePromptTemplate chứa câu hỏi của người dùng

    # Prompt Template: Hướng dẫn chi tiết cho LLM
    # Cập nhật hướng dẫn để LLM chú ý đến thông tin JSON bổ sung
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là một siêu AI về PostgreSQL, dữ liệu JSON, và hỗ trợ kỹ thuật chuyên sâu. Nhiệm vụ của bạn là tạo ra câu lệnh SQL, lưu ý khi select nếu đặt tên thì dùng theo tiếng đức thông dụng "
            "dựa trên câu hỏi của người dùng và cấu trúc database được cung cấp.\n"
            "Hãy chú ý đặc biệt đến các cột có kiểu dữ liệu JSON/JSONB và sử dụng thông tin cấu trúc bổ sung được cung cấp để truy vấn dữ liệu bên trong JSON một cách chính xác.\n\n"
            f"{schema_text}\n\n" # Chèn thông tin schema đã format vào đây
            "Hãy tuân thủ các quy tắc sau:\n"
            "1. Chỉ trả về câu lệnh SQL. Không thêm bất kỳ giải thích, văn bản bổ sung, hay ghi chú nào.\n"
            "2. Đặt câu lệnh SQL duy nhất trong khối mã markdown, bắt đầu bằng ```sql và kết thúc bằng ```.\n"
            '''3. Luôn sử dụng tên schema đầy đủ khi tham chiếu đến bảng (ví dụ: `raw.haus_mieten_test`), chú ý khi select các cột luôn sử dụng dấu trích dẫn ví dụ  t."abCd","@id","@publishDate".\n'''
            "4. Chỉ sử dụng các bảng và cột được mô tả trong cấu trúc database.\n"
            "5. Tránh các câu lệnh SQL nguy hiểm (như DROP, DELETE, UPDATE, INSERT, ALTER, CREATE). Chỉ tạo câu lệnh SELECT.\n"
            "6. Nếu câu hỏi không liên quan đến việc truy vấn dữ liệu từ các bảng đã cho, hãy trả lời bằng văn bản 'Tôi không thể tạo câu lệnh SQL cho yêu cầu này.'\n"
            "7. **Truy vấn dữ liệu JSON/JSONB (PostgreSQL):**\n"
            ".  - Nếu bạn cần chọn toàn bộ một cột JSONB để hiển thị trong kết quả, hãy LUÔN luôn ép kiểu nó thành TEXT để tránh các vấn đề về suy luận kiểu của Pandas/PyArrow)\n"
            "   - Sử dụng toán tử `->` để truy cập một key và trả về kết quả dưới dạng **JSONB object/array**.\n"
            "   - Sử dụng toán tử `->>` để truy cập một key và trả về kết quả dưới dạng **TEXT**.\n"
            "   - Để truy cập các key lồng nhau, hãy chuỗi các toán tử `->` và kết thúc bằng `->>` nếu bạn muốn lấy giá trị cuối cùng dưới dạng TEXT.\n"
            "     Ví dụ: `(json_column -> 'key1' -> 'nested_key' ->> 'final_key')`\n"
            "   - Sử dụng `jsonb_array_elements(json_array_column)` để mở rộng mảng JSON thành các dòng.\n"
            "   - **Ép kiểu (Casting):** Khi bạn cần sử dụng giá trị TEXT (từ `->>` hoặc `#>>`) dưới dạng số (numeric, integer, float) hoặc boolean để so sánh/tính toán, hãy ép kiểu nó.\n"
            "   - Cú pháp ép kiểu là `(json_text_expression)::target_type`. Luôn bọc biểu thức truy cập JSON trong ngoặc đơn trước khi ép kiểu.\n"
            "   - Ví dụ ép kiểu sang numeric: `(json_column -> 'price' ->> 'value')::numeric`\n" # Ví dụ đúng cú pháp giá
            '''   - khi gộp dữ liệu của 2 cột cần chú ý các dấu và ngoặc cần thiết ví dụ: (t."resultlist.realEstate" -> 'address' ->> 'street') || ', ' || (t."resultlist.realEstate" -> 'address' ->> 'city') AS "Adresse" '''
            "   - **Tránh:** KHÔNG ép kiểu kết quả của `->>` sang `jsonb` một cách không cần thiết. Ví dụ `(json_column ->> 'key')::jsonb` là SAI nếu bạn chỉ muốn lấy giá trị TEXT hoặc ép kiểu sang kiểu khác (numeric, boolean...).\n"
            '''8. Đây là 1 ví dụ để truy vấn các đối tượng trong kiểu dữ liệu json, bạn có thể tham khảo để phát triển tìm các đối tượng khác nằm sâu hơn, hoặc phức tạp hơn:
                SELECT "attributes"
                FROM raw.haus_kaufen
                WHERE (
                SELECT REPLACE(REPLACE(attr->>'value', '.', ''), ' €', '')::numeric
                FROM jsonb_array_elements(attributes) AS item,
                    jsonb_array_elements(item->'attribute') AS attr
                WHERE attr->>'label' = 'Kaufpreis'
                    AND attr->>'value' ~ '^[0-9\.]+ €$'  -- chỉ lấy giá trị dạng số + €
                ) > 1000000;
            '''
            '''
            - chú ý sự Lồng sâu kết hợp của cả mảng và object trong kiểu dữ liệu json: ví dụ URL (@href) nằm sâu 4-5 cấp và đi qua cả mảng cả object. Điều này đòi hỏi phải chú ý điều kiện về mảng hoặc object \n"
            -- ein BEISPIEL: Abrufen von Bild-URLs aus tief verschachtelten JSON-Strukturen: 
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

            ''' # Dịch và thêm ví dụ mới
            
             # Thêm ví dụ truy vấn JSON khác nếu cần thiết cho cấu trúc dữ liệu của bạn.
             # Ví dụ cho trường hợp object trực tiếp: SELECT attributes->>'some_key' FROM ...
             # Ví dụ cho trường hợp mảng: SELECT item->>'some_key' FROM jsonb_array_elements(attributes) as item WHERE ...
            + "\n" # Thêm dòng trống cuối SystemMessage
        )),
        HumanMessagePromptTemplate.from_template("{user_query}") # Template cho câu hỏi của người dùng
    ])

    # Trả về template và thông báo thành công/lỗi (nếu có ở phần chuẩn bị schema)
    return template, None # Trả về template sẵn sàng để format, và không có lỗi


# --- Giai đoạn 2: Gọi LLM, Nhận kết quả và Trích xuất SQL (Giữ nguyên logic gọi hàm get_sql_from_llm) ---
# Hàm này KHÔNG thay đổi, nó chỉ gọi generate_sql_prompt đã được cập nhật
def get_sql_from_llm(user_query):
    """
    Gọi LLM đã được khởi tạo để tạo câu lệnh SQL
    dựa trên query của người dùng và schema đã tải (bao gồm cả JSON info).

    Args:
        user_query (str): Câu hỏi của người dùng.

    Returns:
        tuple: (extracted_sql: str | None, error_msg: str | None)
               Trả về SQL và None nếu thành công, hoặc None và thông báo lỗi nếu thất bại.
    """
    # Kiểm tra các điều kiện tiên quyết
    if st.session_state.chat_llm is None:
        return None, "LLM chưa được khởi tạo. Vui lòng chọn và khởi tạo LLM trước."
    if st.session_state.schema_info is None or not st.session_state.schema_info:
         return None, "Thông tin schema database chưa được tải hoặc trống. Vui lòng kết nối DB trước."

    print(f"Generating prompt for query: {user_query}")
    # Bước 2.1 & 2.3: Chuẩn bị schema info và xây dựng prompt
    # Hàm generate_sql_prompt đã được cập nhật để bao gồm thông tin JSON
    prompt_template, error_msg = generate_sql_prompt(st.session_state.schema_info, user_query)

    if error_msg:
        return None, error_msg # Trả về lỗi nếu không tạo được prompt (ví dụ: không có schema)
    if prompt_template is None:
         return None, "Không thể tạo prompt vì không có thông tin schema khả dụng."


    # Format prompt với câu hỏi của người dùng
    formatted_prompt = prompt_template.format_messages(user_query=user_query)

    # print("--- Formatted Prompt sent to LLM ---")
    # for msg in formatted_prompt:
    #      # In vài ký tự đầu của nội dung để tránh tràn console
    #      # print(f"Type: {msg.type}, Content: {msg.content[:500]}...")
    #      print(f"Type: {msg.type}, Content: {msg.content[:1000]}{'...' if len(msg.content) > 1000 else ''}")
    # print("-----------------------------------")

    try:
        print(f"Calling LLM ({st.session_state.chat_llm.__class__.__name__})...") # In tên lớp LLM
        # Bước 2.4: Gọi LLM và nhận phản hồi
        response = st.session_state.chat_llm.invoke(formatted_prompt)
        llm_output = response.content # Lấy nội dung phản hồi dạng chuỗi

        print(f"LLM Raw Output:\n{llm_output}")

        # Bước 2.5: Trích xuất Câu lệnh SQL từ phản hồi
        # Tìm block code markdown ```sql ... ```
        sql_block_start_tag = "```sql"
        sql_block_end_tag = "```"

        sql_start_index = llm_output.find(sql_block_start_tag)
        # Tìm ký hiệu kết thúc sau ký hiệu bắt đầu
        sql_end_index = llm_output.find(sql_block_end_tag, sql_start_index + len(sql_block_start_tag))

        if sql_start_index != -1 and sql_end_index != -1:
            # Trích xuất chuỗi nằm giữa hai ký hiệu
            extracted_sql = llm_output[sql_start_index + len(sql_block_start_tag) : sql_end_index].strip()
            print(f"Extracted SQL:\n{extracted_sql}")

            # Tùy chọn: Kiểm tra xem SQL có phải là SELECT không (để tăng cường an toàn)
            # Chấp nhận SELECT theo sau bởi dấu ngoặc đơn hoặc khoảng trắng
            if not extracted_sql.strip().upper().startswith("SELECT"):
                 # Nếu LLM tạo ra lệnh không phải SELECT mặc dù đã nhắc trong prompt
                 # Hoặc nếu đó là thông báo lỗi từ LLM nhưng được format trong code block
                 return None, f"LLM đã tạo ra lệnh không phải SELECT hoặc không trích xuất được lệnh SELECT hợp lệ. Vui lòng thử lại hoặc điều chỉnh câu hỏi.\nGenerated Query:\n```sql\n{extracted_sql}\n```"


            return extracted_sql, None # Trả về SQL đã trích xuất và không có lỗi
        else:
            # Nếu không tìm thấy format ```sql ... ```
            print("Could not find SQL code block in LLM output.")
            # Trả về toàn bộ output của LLM và thông báo lỗi trích xuất
            # Kiểm tra xem LLM có trả lời rằng không thể tạo SQL không
            if "Tôi không thể tạo câu lệnh SQL cho yêu cầu này" in llm_output:
                 return None, llm_output # Trả về thông báo lỗi từ chính LLM
            else:
                # Trường hợp LLM trả lời nhưng không đúng format hoặc không liên quan
                return None, f"LLM Response (Không trích xuất được SQL):\n```\n{llm_output}\n```\n\n**Lỗi:** Không tìm thấy block mã SQL (` ```sql ... ``` `) trong phản hồi của AI. Vui lòng thử lại hoặc điều chỉnh câu hỏi."

    except Exception as e:
        # Bắt các lỗi khác trong quá trình gọi LLM hoặc xử lý phản hồi
        print(f"Error during LLM call or processing: {e}")
        st.error(f"Lỗi khi gọi LLM hoặc xử lý phản hồi: {e}")
        return None, f"Lỗi khi gọi LLM hoặc xử lý phản hồi: {e}"


# --- GIAI ĐOẠN 4: HÀM THỰC THI SQL ---
def execute_sql_query(engine, sql_query):
    """
    Thực thi một câu lệnh SQL SELECT đã cho và trả về kết quả dưới dạng Pandas DataFrame.

    Args:
        engine: SQLAlchemy engine instance.
        sql_query (str): Câu lệnh SQL cần thực thi.

    Returns:
        tuple: (dataframe_result: pd.DataFrame | None, error_msg: str | None)
               Trả về DataFrame và None nếu thành công, hoặc None và thông báo lỗi nếu thất bại.
    """
    if engine is None:
        return None, "Datenbankverbindung nicht hergestellt." # Dịch

    if not sql_query or not sql_query.strip().upper().startswith("SELECT"):
        return None, "Ungültige oder keine SELECT-SQL-Abfrage zum Ausführen." # Dịch

    print(f"Executing SQL Query:\n{sql_query}")

    try:
        # Sử dụng pandas.read_sql_query để đọc kết quả trực tiếp vào DataFrame
        df = pd.read_sql_query(
            text(sql_query), engine
            )
        print("SQL Query executed successfully.")
        return df, None
    except OperationalError as e:
        print(f"Database operational error during query execution: {e}")
        # Log chi tiết hơn nếu có original error
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        return None, f"Datenbank-Betriebsfehler beim Ausführen der Abfrage: {err_detail}" # Dịch
    except SQLAlchemyError as e:
        print(f"A SQLAlchemy error occurred during query execution: {e}")
        err_detail = str(e)
        if hasattr(e, 'orig') and e.orig is not None:
             err_detail += f" (Original: {e.orig})"
        return None, f"Ein Datenbankfehler ist beim Ausführen der Abfrage aufgetreten: {err_detail}" # Dịch
    except Exception as e:
        print(f"An unexpected error occurred during query execution: {e}")
        return None, f"Ein unerwarteter Fehler ist beim Ausführen der Abfrage aufgetreten: {e}" # Dịch


#--------------------------------------------------------------------------------------------------------




# --- Streamlit UI ---

st.set_page_config(page_title="Text-to-SQL Chatbot - JSON-Verständnis verbessern", layout="wide") # Dịch tiêu đề
st.title("Text-to-SQL Chatbot") # Dịch tiêu đề

st.sidebar.header("Einstellungen") # Dịch

# --- Phần Kết nối Database (UI Sidebar) ---
st.sidebar.subheader("Datenbankverbindung") # Dịch

# Input cho URI database với giá trị mặc định
default_uri = os.getenv("DATABASE_URI", "postgresql://postgres:123456@localhost:5432/immobilien") # Lấy từ env nếu có, hoặc dùng default
db_uri_input = st.sidebar.text_input(
    "Datenbank-URI eingeben:", # Dịch
    value=default_uri,
    type="password", # Dùng type="password" để ẩn URI
    key="db_uri_input" # Thêm key
)

# Nút để bắt đầu quá trình kết nối và đọc schema
if st.sidebar.button("Verbinden & Schema lesen", key="connect_db_button"): # Dịch và thêm key
    if not db_uri_input:
        st.sidebar.warning("Bitte geben Sie die Datenbank-URI ein.") # Dịch
    else:
        st.sidebar.info("Verbinde und lese Datenbankschema...") # Dịch
        # connect_and_load_schema giờ hiển thị thông báo trực tiếp
        connect_and_load_schema(db_uri_input)

        # Sau khi kết nối xong (thành công hoặc thất bại), tự động khởi tạo LLM mặc định nếu chưa có
        # và có thông tin LLM provider/model khả dụng
        if st.session_state.chat_llm is None:
             # Lấy giá trị temperature mặc định từ UI nếu đã có, hoặc 0.4 nếu chưa
             default_temp = st.session_state.get('llm_temperature_input', 0.4)
             current_provider = st.session_state.get('llm_provider_selection', USE_LLM) # Lấy provider hiện tại hoặc mặc định

             # Tìm model mặc định đầu tiên trong danh sách của provider đã chọn
             default_model_name = None
             if current_provider in AVAILABLE_LLM_MODELS and AVAILABLE_LLM_MODELS[current_provider]:
                  default_model_name = AVAILABLE_LLM_MODELS[current_provider][0]

             if default_model_name:
                  # st.sidebar.info(f"Versuche, standardmäßiges LLM zu initialisieren: {current_provider} ({default_model_name}) mit Temp={default_temp}") # Dịch
                  # initialize_llm giờ hiển thị thông báo và lỗi trực tiếp
                  initialize_llm(current_provider, default_model_name, default_temp)
             else:
                  st.sidebar.warning(f"Keine verfügbaren Modelle für Anbieter '{current_provider}' gefunden. Bitte wählen und initialisieren Sie das LLM manuell.") # Dịch


# --- Giai đoạn 3.2 & 3.4: UI cho Lựa chọn LLM, Mô hình và Temperature (UI Sidebar) ---
st.sidebar.subheader("LLM auswählen") # Dịch

llm_provider_options = list(AVAILABLE_LLM_MODELS.keys())
# Đảm bảo USE_LLM mặc định có trong options, nếu không chọn cái đầu tiên
initial_llm_provider_index = llm_provider_options.index(USE_LLM) if USE_LLM in llm_provider_options else 0

llm_provider = st.sidebar.selectbox(
    "LLM-Anbieter auswählen:", # Dịch
    llm_provider_options,
    index=initial_llm_provider_index,
    key="llm_provider_selection", # Thêm key
    # on_change có thể reset model selection nếu cần
)

# Dựa vào nhà cung cấp đã chọn, hiển thị selectbox cho các mô hình khả dụng
model_options = AVAILABLE_LLM_MODELS.get(llm_provider, []) # Lấy list model, trả về rỗng nếu provider không tồn tại

# Sửa lỗi: Lấy thông tin LLM hiện tại một cách an toàn
current_llm_info_state = st.session_state.get('current_llm_info')

# Sửa lỗi: Xác định current_model và current_temp một cách an toàn
if isinstance(current_llm_info_state, dict):
    current_model = current_llm_info_state.get('model_name')
    current_temp = current_llm_info_state.get('config', {}).get('temperature', 0.4)
else:
    # Nếu không có thông tin LLM hoặc không đúng định dạng, dùng giá trị mặc định ban đầu (hoặc từ UI input trước đó)
    current_model = None # Sẽ dùng logic chọn model mặc định bên dưới
    # Lấy default temperature từ input UI nếu nó đã tồn tại trong session state, nếu không thì 0.4
    current_temp = st.session_state.get('llm_temperature_input', 0.4)


# Logic chọn initial_model_index cho selectbox model:
initial_model_index = 0
# Ưu tiên 1: Model đang được load trong session state
if current_model and current_model in model_options:
     initial_model_index = model_options.index(current_model)
# Ưu tiên 2: Model được chọn trong selectbox trước đó (trước khi script re-run) và vẫn còn trong list options mới
elif st.session_state.get('llm_model_selection') in model_options:
    initial_model_index = model_options.index(st.session_state.llm_model_selection)
# Ưu tiên 3: Model mặc định đầu tiên của provider được chọn
elif len(model_options) > 0:
    default_first_model_in_list = model_options[0]
    initial_model_index = model_options.index(default_first_model_in_list) if default_first_model_in_list in model_options else 0 # Chỉ số 0 nếu list không rỗng
else:
     # Không có model nào khả dụng
     initial_model_index = 0 # Sẽ chọn chỉ số 0, nhưng selectbox sẽ rỗng nếu model_options rỗng


selected_model = st.sidebar.selectbox(
    "LLM-Modell auswählen:", # Dịch
    model_options,
    index=initial_model_index,
    key="llm_model_selection" # Thêm key
)

# --- Bước 3.4: Thêm UI tùy chỉnh Temperature (dạng thanh kéo) ---

# Sử dụng CSS để style thanh trượt (giữ nguyên)
st.markdown("""
<style>
/* --- Style cho THÂN thanh trượt đã đầy (filled track) --- */
div[data-testid="stSlider"] div.st-ds {
    background-color: #1E90FF !important; /* Deep Sky Blue */
}

/* --- Style cho NÚM kéo (thumb) của thanh trượt --- */
div[data-testid="stSlider"] div.st-dg { /* class có thể là st-dj hoặc st-dg tùy version */
    background-color: #1E90FF !important; /* Deep Sky Blue */
    border-color: #1E90FF !important;
}
</style>
""", unsafe_allow_html=True)

# Sử dụng current_temp đã lấy một cách an toàn ở trên
selected_temperature = st.sidebar.slider(
    "Temperatur:", # Dịch
    min_value=0.0, # Giá trị nhỏ nhất
    max_value=2.0, # Giá trị lớn nhất
    value=current_temp,     # Giá trị mặc định hoặc giá trị hiện tại
    step=0.01,     # Bước nhảy khi kéo
    format="%.2f", # Định dạng hiển thị số
    key="llm_temperature_input" # Lưu giá trị vào session state
)


# Nút để khởi tạo LLM với nhà cung cấp, mô hình và temperature đã chọn
if st.sidebar.button("LLM initialisieren/wechseln", key="init_llm_button"): # Dịch và thêm key
     if selected_model:
         # st.sidebar.info(f"Initialisiere LLM: {llm_provider} ({selected_model}) Temp={selected_temperature}...") # Dịch
         # initialize_llm giờ hiển thị thông báo và lỗi trực tiếp
         initialize_llm(llm_provider, selected_model, selected_temperature)
     else:
         st.sidebar.warning("Bitte wählen Sie ein LLM-Modell aus.") # Dịch


# Hiển thị trạng thái kết nối và LLM (Đã cập nhật để hiển thị config)
st.sidebar.subheader("Status") # Dịch
if st.session_state.db_engine:
    st.sidebar.success("Datenbank verbunden") # Dịch
else:
    st.sidebar.warning("Datenbank nicht verbunden") # Dịch

# Hiển thị thông tin LLM từ session state một cách an toàn
current_llm_info_display = st.session_state.get('current_llm_info')
if isinstance(current_llm_info_display, dict):
    llm_info = current_llm_info_display
    config_str = ", ".join([f"{k}={v}" for k, v in llm_info.get('config', {}).items()])
    status_text = f"LLM bereit ({llm_info.get('llm_type', 'N/A')}: {llm_info.get('model_name', 'N/A')})" # Dịch
    if config_str:
        status_text += f" mit Konfiguration: {config_str}" # Dịch
    st.sidebar.success(status_text)
else:
    st.sidebar.warning("LLM nicht initialisiert") # Dịch


# --- Hiển thị thông tin Schema đã đọc được (Main Panel) ---
st.subheader("Schema-Informationen") # Dịch

# Kiểm tra schema_info một cách an toàn
if st.session_state.get('schema_info') and isinstance(st.session_state.schema_info, dict):
    schema_info_to_display = st.session_state.schema_info
    if schema_info_to_display:
        # st.write("Gefundene Datenbankschema-Struktur (ohne System- und leere Schemata):") # Đã bỏ dòng này vì thông tin sẽ hiển thị trong selectbox

        # *** PHẦN ĐÃ SỬA ĐỔI: Thêm thông tin số lượng vào label của Selectbox ***

        # Chuẩn bị danh sách schema với thông tin số lượng bảng usable
        # schema_options sẽ là danh sách các tên schema (raw, public, ...)
        # schema_format_map sẽ ánh xạ tên schema tới chuỗi hiển thị
        schema_options_for_selectbox = []
        schema_format_map = {}
        for s_name, s_tables in schema_info_to_display.items():
            if isinstance(s_tables, dict):
                # Đếm số lượng bảng CÓ CỘT trong schema này
                num_usable_tables_in_schema = len([
                    t_name for t_name, cols in s_tables.items()
                    if isinstance(cols, list) and cols # Đảm bảo là list và không rỗng
                ])
                # Chỉ thêm vào danh sách lựa chọn nếu schema có ít nhất 1 bảng có cột
                if num_usable_tables_in_schema > 0:
                    schema_options_for_selectbox.append(s_name)
                    schema_format_map[s_name] = f"{s_name} ({num_usable_tables_in_schema} Tabellen)" # Dịch: "raw (4 Tabellen)"

        # Selectbox cho Schema
        initial_schema_index = 0 if len(schema_options_for_selectbox) > 0 else None
        selected_schema_name = st.selectbox(
            "Schema auswählen:", # Dịch
            schema_options_for_selectbox, # Danh sách tên schema thực tế
            index=initial_schema_index,
            format_func=lambda s: schema_format_map.get(s, s), # Sử dụng map để hiển thị chuỗi đã định dạng
            key="schema_selection",
            help="Wählen Sie ein Datenbankschema." # Dịch
        )

        # Khởi tạo selected_table_name là None ban đầu
        selected_table_name = None
        usable_tables_in_schema = {} # Khởi tạo usable_tables_in_schema là dict rỗng

        if selected_schema_name and selected_schema_name in schema_info_to_display:
            current_schema_tables = schema_info_to_display[selected_schema_name]

            # Lọc chỉ các bảng usable (có cột) trong schema đã chọn
            usable_tables_in_schema = {
                 t_name: cols for t_name, cols in current_schema_tables.items()
                 if isinstance(cols, list) and cols # Kiểm tra cols là list và không rỗng
            }

            if usable_tables_in_schema:
                # st.write(f"Tabellen im Schema **`{selected_schema_name}`**:") # Bỏ dòng này hoặc giữ lại nếu muốn có heading phụ

                # Chuẩn bị danh sách bảng với thông tin số lượng cột
                table_options_for_selectbox = []
                table_format_map = {}
                for t_name, t_cols in usable_tables_in_schema.items():
                    num_columns = len(t_cols)
                    table_options_for_selectbox.append(t_name)
                    table_format_map[t_name] = f"{t_name} ({num_columns} Spalten)" # Dịch: "haus_kaufen (10 Spalten)"

                # Sử dụng một selectbox khác để chọn Bảng trong schema đã chọn
                initial_table_index = 0 if len(table_options_for_selectbox) > 0 else None
                selected_table_name = st.selectbox(
                    "Tabelle auswählen:", # Dịch
                    table_options_for_selectbox, # Danh sách tên bảng thực tế
                    index=initial_table_index,
                    format_func=lambda t: table_format_map.get(t, t), # Sử dụng map để hiển thị chuỗi đã định dạng
                    key="table_selection",
                     help="Wählen Sie eine Tabelle im ausgewählten Schema." # Dịch
                )

                if selected_table_name and selected_table_name in usable_tables_in_schema:
                    # Lấy thông tin cột của bảng đã chọn
                    columns_of_selected_table = usable_tables_in_schema[selected_table_name]

                    # --- Hiển thị chi tiết của Bảng đã chọn trong expander (KHÔNG DÙNG KEY) ---
                    # Bỏ key để tránh lỗi "unexpected keyword argument 'key'"
                    with st.expander(f"Details zur Tabelle **`{selected_schema_name}.{selected_table_name}`**", expanded=False): # Dịch. Mở mặc định.

                        # Hạ cấp Überschrift và tùy chọn làm text nhỏ hơn (CSS)
                        st.markdown("#### Spalteninformationen:") # Dịch. Sử dụng Markdown level 4 heading (nhỏ hơn subheader)

                        # Tạo DataFrame chỉ với các cột thông tin cơ bản
                        columns_df = pd.DataFrame(columns_of_selected_table)
                        cols_to_display_order = ['name', 'type', 'primary_key', 'nullable', 'default', 'autoincrement']
                        cols_to_display_filtered = [col for col in columns_df.columns if col in cols_to_display_order]

                        # Hiển thị DataFrame cơ bản
                        if not columns_df.empty and cols_to_display_filtered:
                            st.dataframe(columns_df[cols_to_display_filtered], use_container_width=True)
                        elif not columns_df.empty and not cols_to_display_filtered:
                            st.write("Keine der Standardspalteninformationen (PK, NOT NULL, DEFAULT, AUTO) für diese Tabelle.") # Dịch
                        else: # columns_df is empty
                            st.write("Spalteninformationen für diese Tabelle konnten nicht gelesen werden.") # Dịch

                        # --- Hiển thị JSON summary nếu có ---
                        json_cols_summaries = {
                            col.get('name'): col.get('json_structure_summary')
                            for col in columns_of_selected_table if isinstance(col, dict) and col.get('json_structure_summary')
                        }

                        if json_cols_summaries:
                            st.markdown("**Details zu JSON-Spalten:**") # Dịch. Sử dụng bold markdown.
                            for col_name, summary in json_cols_summaries.items():
                                if col_name and summary:
                                    st.write(f"**Spalte `{col_name}`:**") # Dịch. Sử dụng bold markdown cho tên cột.
                                    st.text(summary) # st.text cho summary

                else: # Đã chọn Schema nhưng không có Table usable để chọn
                     st.info(f"Keine nutzbaren Tabellen im Schema **`{selected_schema_name}`** gefunden.") # Dịch
                     selected_table_name = None # Đảm bảo selected_table_name là None
            else: # Không có Schema usable nào để chọn
                 st.info("Keine nutzbaren Schemata in der Datenbank gefunden.") # Dịch
                 selected_schema_name = None # Đảm bảo selected_schema_name là None

    else: # Trường hợp schema_info rỗng (DB chưa connect hoặc lỗi connect)
         st.warning("Verbindung erfolgreich, aber keine nutzbaren Schemata oder Tabellen außerhalb der Systemschemata gefunden.") # Dịch

else: # Trường hợp st.session_state.schema_info là None
    st.info("Bitte geben Sie die Datenbank-URI ein und klicken Sie auf 'Verbinden & Schema lesen', um zu beginnen.") # Dịch

# Phần còn lại của code (Generate SQL, Execute SQL Placeholder) giữ nguyên


# --- Giai đoạn 2: Tạo SQL (UI cho User Query) (Giữ nguyên logic gọi hàm get_sql_from_llm) ---
st.subheader("SQL - Frage") # Dịch

# Kiểm tra các điều kiện tiên quyết một cách an toàn
if st.session_state.get('schema_info') is None or not st.session_state.schema_info:
    st.warning("Bitte verbinden Sie die Datenbank und lesen Sie das Schema, bevor Sie fragen.") # Dịch
elif st.session_state.get('chat_llm') is None:
    st.warning("Bitte initialisieren Sie das LLM, bevor Sie fragen.") # Dịch
else:
    user_query = st.text_area(
        "Geben Sie Ihre Frage zu den Daten ein (z.B.: 'Listen Sie die 5 Adressen und höchsten Mietpreise aus der Tabelle haus_mieten_test im Schema raw auf'):", # Dịch
        key="user_query_input", # Thêm key
        height=100
    )

    if st.button("SQL-Abfrage generieren", key="generate_sql_button"): # Dịch và thêm key
        if not user_query:
            st.warning("Bitte geben Sie eine Frage ein.") # Dịch
        else:
            #st.info("Generiere SQL-Abfrage mit LLM...") # Dịch
            st.session_state.generated_sql = None

            # Hàm get_sql_from_llm sẽ gọi generate_sql_prompt đã được cập nhật
            generated_sql, error_msg = get_sql_from_llm(user_query) # Hàm này sử dụng LLM từ session state

            if error_msg:
                st.error(error_msg)
                st.session_state.generated_sql = None # Đảm bảo reset nếu có lỗi gen
            else:
                st.session_state.generated_sql = generated_sql # LƯU VÀO SESSION STATE DÙ LUÔN LUÔN HIỂN THỊ

    # --- PHẦN MỚI: Luôn hiển thị SQL đã tạo nếu có ---
    if st.session_state.generated_sql:
        st.success("SQL-Abfrage generiert:") # Dịch
        st.code(st.session_state.generated_sql, language="sql")






# --- Giai đoạn 4: Thực thi SQL và Hiển thị Kết quả ---
st.subheader("SQL ausführen und Ergebnisse anzeigen") # Dịch. Đã bỏ "(Phase 4)"

# Kiểm tra nếu có SQL đã tạo và DB đã kết nối
if st.session_state.get('generated_sql'): # Điều kiện này đảm bảo nút thực thi chỉ hiện khi có SQL
    if st.session_state.db_engine:
        if st.button("SQL ausführen", key="execute_sql_button"): # Dịch và thêm key
            st.info("Führe SQL-Abfrage aus...") # Dịch
            st.session_state.query_result_df = None # Reset kết quả cũ
            st.session_state.query_error = None # Reset lỗi cũ

            result_df, error_msg = execute_sql_query(
                st.session_state.db_engine,
                st.session_state.generated_sql
            )

            if error_msg:
                st.session_state.query_error = error_msg
                st.error(error_msg)
            else:
                st.session_state.query_result_df = result_df
                st.success("SQL-Abfrage erfolgreich ausgeführt!") # Dịch

        # Hiển thị kết quả sau khi thực thi
        if st.session_state.query_result_df is not None:
            # --- PHẦN MỚI: HIỂN THỊ TỔNG SỐ DÒNG ---
            num_rows = st.session_state.query_result_df.shape[0]
            st.markdown(f"##### **Abfrageergebnisse:** Total {num_rows} rows") # Dịch, thêm số dòng
            
            if not st.session_state.query_result_df.empty:
                st.dataframe(st.session_state.query_result_df, use_container_width=True)
            else:
                st.info("Die Abfrage hat keine Ergebnisse zurückgegeben.") # Dịch
        elif st.session_state.query_error: # Nếu có lỗi từ lần thực thi trước
            st.error(st.session_state.query_error)

    else:
        st.warning("Datenbank nicht verbunden. Bitte verbinden Sie die Datenbank, um SQL auszuführen.") # Dịch
else:
     st.info("Generieren Sie oben eine SQL-Abfrage, um sie auszuführen.") # Dịch


# --- Chú ý cuối file (Giữ nguyên) ---
print("Streamlit app is running. Waiting for user interaction.")