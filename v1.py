import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv

# Import các lớp cần thiết từ SQLAlchemy cho kết nối và đọc schema
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# Import các lớp cần thiết từ LangChain Core và các tích hợp LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# Import lớp lỗi cụ thể của OpenAI nếu có thể để bắt chính xác lỗi tham số
# try:
#      from openai import InvalidRequestError as OpenAIInvalidRequestError
# except ImportError:
#      OpenAIInvalidRequestError = Exception # Fallback nếu không import được

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


# --- Giai đoạn 1: Kết nối Database và đọc Schema (Giữ nguyên) ---
# Đặt code của hàm connect_and_load_schema ở đây
def connect_and_load_schema(db_uri):
    # ... (code hàm connect_and_load_schema) ...
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

        for schema in all_schemas:
            system_schemas = ['information_schema', 'pg_catalog', 'mysql', 'sys', 'performance_schema', 'temp_tables']
            if schema in system_schemas:
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
                     except Exception as col_exc:
                         print(f"        Could not get columns for {schema}.{table}: {col_exc}")
                         database_schema[schema][table] = []

            except Exception as table_list_exc:
                print(f"    Could not get tables for schema {schema}: {table_list_exc}")
                # Không cần pop schema ở đây, giữ lại tên schema để thông báo là không đọc được bảng

        st.session_state.db_engine = engine
        st.session_state.schema_info = database_schema

        print("Database schema loaded successfully!")
        return {"message": "Connection successful and schema loaded!"}

    except OperationalError as e:
        print(f"Database connection failed: {e}")
        if engine:
             engine.dispose()
        return {"error": f"Database connection failed: {e}"}
    except SQLAlchemyError as e:
        print(f"A SQLAlchemy error occurred: {e}")
        if engine:
             engine.dispose()
        return {"error": f"A database error occurred: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if engine:
             engine.dispose()
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
        "gemini-2.5-pro", # Mô hình text chính
        "gemini-2.5-flash", # Có context rất dài
        "gemini-2.5-flash-lite-preview-06-17", # Nhanh hơn, rẻ hơn
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash", # Nếu cần xử lý ảnh (không cần cho Text-to-SQL)
        "gemini-1.5-pro" 
        # Có thể thêm các model khác nếu cần
    ]
    # Có thể thêm các nhà cung cấp khác như "Anthropic", "Mistral" vào đây
}

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

    try:
        if llm_choice == "OpenAI":
            if not openai_api_key:
                st.error("OpenAI API Key không tìm thấy...")
                return {"error": "Missing OpenAI API Key"}
            if model_name not in AVAILABLE_LLM_MODELS["OpenAI"]:
                 st.warning(f"Mô hình OpenAI '{model_name}' không nằm trong danh sách hỗ trợ. Sử dụng mặc định 'gpt-3.5-turbo'.")
                 model_name = "gpt-3.5-turbo" # Fallback

            # --- Thử khởi tạo OpenAI LLM với các tham số ---
            try:
                # Thử với tham số temperature
                initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, **llm_params)
                actual_config_used = llm_params # Nếu thành công, ghi lại config đã dùng
                print(f"Initialized OpenAI LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 # Bắt lỗi cụ thể hơn nếu có thể, ví dụ OpenAIInvalidRequestError
                 # Kiểm tra thông báo lỗi có chứa từ khóa liên quan đến tham số không hỗ trợ không
                 error_message = str(e_params).lower()
                 if 'unsupported parameter' in error_message or 'unsupported value' in error_message or 'invalid_request_error' in error_message:
                     print(f"Warning: Parameter error during OpenAI init with config {llm_params}: {e_params}. Trying without parameter 'temperature'.")
                     # Thử lại chỉ với model_name
                     try:
                         initialized_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name)
                         actual_config_used = {} # Ghi lại rằng không dùng config tùy chỉnh
                         print(f"Initialized OpenAI LLM: {model_name} without custom temperature")
                     except Exception as e_retry:
                          # Nếu thử lại vẫn lỗi, báo lỗi thật
                          st.error(f"Lỗi khi khởi tạo OpenAI LLM '{model_name}' (thử lại không temperature): {e_retry}")
                          return {"error": f"Failed to initialize OpenAI LLM ({model_name}): {e_retry}"}
                 else:
                     # Nếu là lỗi khác không phải do tham số
                     st.error(f"Lỗi không xác định khi khởi tạo OpenAI LLM '{model_name}': {e_params}")
                     return {"error": f"Failed to initialize OpenAI LLM ({model_name}): {e_params}"}


        elif llm_choice == "Google":
            if not google_api_key:
                st.error("Google API Key không tìm thấy...")
                return {"error": "Missing Google API Key"}
            if model_name not in AVAILABLE_LLM_MODELS["Google"]:
                 st.warning(f"Mô hình Google '{model_name}' không nằm trong danh sách hỗ trợ. Sử dụng mặc định 'gemini-pro'.")
                 model_name = "gemini-pro" # Fallback

            # --- Thử khởi tạo Google LLM với các tham số ---
            try:
                 # Có thể cần client_options cho một số model/region của Google
                 # from google.api_core.client_options import ClientOptions
                 # google_region = "us-central1"
                 # client_options = ClientOptions(api_endpoint=f"{google_region}-generativelanguage.googleapis.com")

                 # Thử với tham số temperature
                 # **llm_params sẽ truyền temperature=...
                 initialized_llm = ChatGoogleGenerativeAI(
                     google_api_key=google_api_key,
                     model=model_name,
                     **llm_params # Truyền các tham số từ dictionary
                     # client_options=client_options # Bỏ comment nếu cần
                 )
                 actual_config_used = llm_params # Nếu thành công, ghi lại config đã dùng
                 print(f"Initialized Google LLM: {model_name} with config {actual_config_used}")

            except Exception as e_params:
                 # Kiểm tra thông báo lỗi có chứa từ khóa liên quan đến tham số không hỗ trợ không
                 error_message = str(e_params).lower()
                 # Lỗi Google có thể khác OpenAI, cần điều chỉnh từ khóa
                 if 'unsupported value' in error_message or 'invalid argument' in error_message: # Ví dụ các từ khóa lỗi của Google
                     print(f"Warning: Parameter error during Google init with config {llm_params}: {e_params}. Trying without parameter 'temperature'.")
                     # Thử lại chỉ với model_name (và client_options nếu có)
                     try:
                         initialized_llm = ChatGoogleGenerativeAI(
                             google_api_key=google_api_key,
                             model=model_name,
                             # client_options=client_options # Bỏ comment nếu cần
                         )
                         actual_config_used = {} # Ghi lại rằng không dùng config tùy chỉnh
                         print(f"Initialized Google LLM: {model_name} without custom temperature")
                     except Exception as e_retry:
                          # Nếu thử lại vẫn lỗi, báo lỗi thật
                          st.error(f"Lỗi khi khởi tạo Google LLM '{model_name}' (thử lại không temperature): {e_retry}")
                          return {"error": f"Failed to initialize Google LLM ({model_name}): {e_retry}"}
                 else:
                     # Nếu là lỗi khác không phải do tham số
                     st.error(f"Lỗi không xác định khi khởi tạo Google LLM '{model_name}': {e_params}")
                     return {"error": f"Failed to initialize Google LLM ({model_name}): {e_params}"}

        else:
            st.warning(f"Lựa chọn LLM '{llm_choice}' không hợp lệ...")
            return {"error": f"Invalid LLM choice: {llm_choice}"}

    except Exception as e:
        # Bắt các lỗi khác không liên quan đến cấu hình ban đầu (ví dụ: lỗi API key sai, lỗi mạng)
        print(f"An unexpected error occurred during LLM initialization: {e}")
        return {"error": f"An unexpected error occurred during LLM initialization: {e}"}

    # Nếu khởi tạo thành công (trong bất kỳ nhánh try/except nào)
    if initialized_llm:
        st.session_state.chat_llm = initialized_llm
        st.session_state.current_llm_info = {
            "llm_type": llm_choice,
            "model_name": model_name, # Lưu tên model đã chọn ban đầu
            "config": actual_config_used # Lưu cấu hình thực tế đã sử dụng
        }
        return {"message": f"Initialized {llm_choice} LLM ({model_name})",
                "model_name": model_name,
                "llm_type": llm_choice,
                "config": actual_config_used}
    else:
         # Trường hợp này không nên xảy ra nếu logic try/except đúng, nhưng thêm vào để an toàn
         return {"error": f"Initialization failed for {llm_choice} LLM ({model_name}) without specific error."}


# --- Giai đoạn 2: Chuẩn bị Schema Info và Xây dựng Prompt (Giữ nguyên) ---
def generate_sql_prompt(schema_info_dict, user_query):
    """
    Chuẩn bị thông tin schema và xây dựng prompt cho LLM.

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
        # Bỏ qua schema nếu không có bảng nào được đọc
        if not tables:
             # print(f"  Skipping schema '{schema_name}' in prompt as it has no usable tables.")
             continue

        schema_text += f"Schema: `{schema_name}`\n"

        for table_name, columns in tables.items():
            # Bỏ qua bảng nếu không có cột nào được đọc
            if not columns:
                # print(f"    Skipping table '{schema_name}.{table_name}' in prompt as it has no columns info.")
                continue

            schema_text += f"  Bảng: `{schema_name}.{table_name}`\n" # Luôn dùng cú pháp schema.table
            schema_text += "    Cột:\n"
            for col in columns:
                # Định dạng thông tin cột: tên (kiểu dữ liệu) [Các thuộc tính]
                col_info = f"    - `{col['name']}` ({col['type']})"
                details = []
                if col.get('primary_key'):
                    details.append("PK")
                # Kiểm tra explicit False vì giá trị có thể là None (không biết)
                if col.get('nullable') is False:
                    details.append("NOT NULL")
                # Kiểm tra nếu có giá trị default và hiển thị
                if col.get('default') is not None:
                    default_val = str(col['default'])
                    # Cắt bớt nếu giá trị default quá dài để tránh làm prompt quá tải
                    if len(default_val) > 50:
                         default_val = default_val[:50] + "..."
                    details.append(f"DEFAULT '{default_val}'")
                # Thuộc tính auto increment
                if col.get('autoincrement'):
                     details.append("AUTO_INCREMENT") # Hoặc SERIAL cho Postgres

                if details:
                    col_info += f" [{', '.join(details)}]"
                schema_text += col_info + "\n"
            schema_text += "\n" # Khoảng trống giữa các bảng

    if schema_text == "Database Schema:\n\n": # Kiểm tra nếu không có schema/bảng nào được thêm vào text
         return None, "Không tìm thấy schema hoặc bảng khả dụng để mô tả cho LLM."

    # print("--- Schema Text for Prompt ---")
    # print(schema_text)
    # print("----------------------------")

    # Bước 2.3: Xây dựng Prompt bằng LangChain PromptTemplate
    # Sử dụng PromptTemplate giúp dễ dàng quản lý cấu trúc prompt
    # SystemMessage hướng dẫn vai trò và context DB
    # HumanMessagePromptTemplate chứa câu hỏi của người dùng

    # Prompt Template: Hướng dẫn chi tiết cho LLM
    template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là một trợ lý chuyên gia về PostgreSQL, dữ liệu JSON, và hỗ trợ kỹ thuật chuyên sâu. Nhiệm vụ của bạn là tạo ra câu lệnh SQL "
            "dựa trên câu hỏi của người dùng và cấu trúc database được cung cấp.\n\n"
            f"{schema_text}\n\n" # Chèn thông tin schema đã format vào đây
            "Hãy tuân thủ các quy tắc sau:\n"
            "1. Chỉ trả về câu lệnh SQL. Không thêm bất kỳ giải thích, văn bản bổ sung, hay ghi chú nào.\n"
            "2. Đặt câu lệnh SQL duy nhất trong khối mã markdown, bắt đầu bằng ```sql và kết thúc bằng ```.\n"
            "3. Luôn sử dụng tên schema đầy đủ khi tham chiếu đến bảng (ví dụ: `raw.haus_mieten_test`).\n"
            "4. Chỉ sử dụng các bảng và cột được mô tả trong cấu trúc database.\n"
            "5. Tránh các câu lệnh SQL nguy hiểm (như DROP, DELETE, UPDATE, INSERT, ALTER, CREATE). Chỉ tạo câu lệnh SELECT.\n"
            "6. Nếu câu hỏi không liên quan đến việc truy vấn dữ liệu từ các bảng đã cho, hãy trả lời bằng văn bản 'Tôi không thể tạo câu lệnh SQL cho yêu cầu này.'\n"
            "7. Nếu có thể, hãy giới hạn số lượng kết quả trả về (ví dụ: thêm LIMIT 100) để tránh tải quá nhiều dữ liệu.\n"
            # Có thể thêm các hướng dẫn cụ thể khác tùy vào loại DB và nhu cầu
            # Ví dụ: "Sử dụng cú pháp SQL chuẩn cho PostgreSQL."
        )),
        HumanMessagePromptTemplate.from_template("{user_query}") # Template cho câu hỏi của người dùng
    ])

    # Trả về template và thông báo thành công/lỗi (nếu có ở phần chuẩn bị schema)
    return template, None # Trả về template sẵn sàng để format, và không có lỗi





# --- Giai đoạn 2: Gọi LLM, Nhận kết quả và Trích xuất SQL (Giữ nguyên) ---
def get_sql_from_llm(user_query):
    """
    Gọi LLM đã được khởi tạo để tạo câu lệnh SQL
    dựa trên query của người dùng và schema đã tải.

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
         return None, "Thông tin schema database chưa được tải. Vui lòng kết nối DB trước."

    print(f"Generating prompt for query: {user_query}")
    # Bước 2.1 & 2.3: Chuẩn bị schema info và xây dựng prompt
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
    #      print(f"Type: {msg.type}, Content: {msg.content[:500]}...")
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
            if not extracted_sql.strip().upper().startswith("SELECT"):
                 # Nếu LLM tạo ra query không phải SELECT mặc dù đã nhắc trong prompt
                 return None, f"LLM đã tạo ra lệnh không phải SELECT. Vui lòng thử lại hoặc điều chỉnh câu hỏi.\nGenerated Query:\n{extracted_sql}"


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
        return None, f"Lỗi khi gọi LLM hoặc xử lý phản hồi: {e}"






# --- Streamlit UI ---

st.set_page_config(page_title="Text-to-SQL Chatbot - Giai đoạn 3.5", layout="wide") # Cập nhật tiêu đề
st.title("Text-to-SQL Chatbot - Giai đoạn 3.5: Tùy chỉnh & Sửa lỗi LLM Config")

st.sidebar.header("Cài đặt")

# --- Phần Kết nối Database (UI Sidebar) (Giữ nguyên) ---
st.sidebar.subheader("Kết nối Database")

# Input cho URI database với giá trị mặc định
default_uri = "postgresql://postgres:password@localhost:5432/immobilien"
db_uri_input = st.sidebar.text_input(
    "Nhập Database URI:",
    value=default_uri,
    type="password",
    key="db_uri_input"
)

# Nút để bắt đầu quá trình kết nối và đọc schema
if st.sidebar.button("Kết nối & Đọc Schema"):
    if not db_uri_input:
        st.sidebar.warning("Vui lòng nhập Database URI.")
    else:
        st.sidebar.info("Đang kết nối và đọc schema database...")
        result = connect_and_load_schema(db_uri_input)

        if "error" in result:
            st.sidebar.error(result["error"])
        else:
            st.sidebar.success(result["message"])
            # Sau khi kết nối thành công, tự động khởi tạo LLM mặc định nếu chưa có
            # Lấy giá trị temperature mặc định từ UI nếu đã có, hoặc 0.4 nếu chưa
            default_temp = st.session_state.get('llm_temperature_input', 0.4)
            if st.session_state.chat_llm is None and USE_LLM in AVAILABLE_LLM_MODELS and AVAILABLE_LLM_MODELS[USE_LLM]:
                 default_model_name = AVAILABLE_LLM_MODELS[USE_LLM][0]
                 st.sidebar.info(f"Kết nối DB thành công. Tự động khởi tạo LLM mặc định: {USE_LLM} ({default_model_name}) với temp={default_temp}")
                 # Pass default_temp vào initialize_llm
                 init_llm_result = initialize_llm(USE_LLM, default_model_name, default_temp) # <-- Pass temperature

                 if "error" in init_llm_result:
                      st.sidebar.error(init_llm_result["error"])
                 # Kết quả thành công sẽ tự động lưu vào session state trong initialize_llm

            elif st.session_state.chat_llm:
                st.sidebar.info("Database kết nối thành công.")
            else:
                 st.sidebar.warning(f"Kết nối DB thành công nhưng không thể tự động khởi tạo LLM mặc định ({USE_LLM}). Vui lòng chọn và khởi tạo thủ công.")


# --- Giai đoạn 3.2 & 3.4: UI cho Lựa chọn LLM, Mô hình và Temperature (UI Sidebar) ---
st.sidebar.subheader("Chọn LLM")

llm_provider = st.sidebar.selectbox(
    "Chọn nhà cung cấp LLM:",
    list(AVAILABLE_LLM_MODELS.keys()),
    index=list(AVAILABLE_LLM_MODELS.keys()).index(USE_LLM) if USE_LLM in AVAILABLE_LLM_MODELS else 0,
    key="llm_provider_selection"
)

# Dựa vào nhà cung cấp đã chọn, hiển thị selectbox cho các mô hình khả dụng
model_options = AVAILABLE_LLM_MODELS.get(llm_provider, []) # Lấy list model, trả về rỗng nếu provider không tồn tại
selected_model = st.sidebar.selectbox(
    "Chọn mô hình LLM:",
    model_options,
    key="llm_model_selection"
)

# --- Bước 3.4: Thêm UI tùy chỉnh Temperature (dạng thanh kéo) ---
selected_temperature = st.sidebar.slider(
    "Temperature:",
    min_value=0.0, # Giá trị nhỏ nhất
    max_value=2.0, # Giá trị lớn nhất
    value=0.4,     # Giá trị mặc định
    step=0.01,     # Bước nhảy khi kéo
    format="%.2f", # Định dạng hiển thị số
    key="llm_temperature_input" # Lưu giá trị vào session state
)
# st.slider tự hiển thị khoảng giá trị và tên. Không cần hiển thị "Temperature (0.0 - 2.0):" trong label


# Nút để khởi tạo LLM với nhà cung cấp, mô hình và temperature đã chọn
if st.sidebar.button("Khởi tạo/Đổi LLM"):
     if selected_model:
         # Pass selected_temperature vào initialize_llm
         init_result = initialize_llm(llm_provider, selected_model, selected_temperature) # <-- Pass temperature
         # Kết quả thành công sẽ tự động lưu vào session state trong initialize_llm
         if "error" in init_result:
              st.sidebar.error(init_result["error"])
     else:
         st.sidebar.warning("Vui lòng chọn một mô hình LLM.")


# Hiển thị trạng thái kết nối và LLM (Đã cập nhật để hiển thị config)
st.sidebar.subheader("Trạng thái")
if st.session_state.db_engine:
    st.sidebar.success("Database Connected")
else:
    st.sidebar.warning("Database Not Connected")

# Hiển thị thông tin LLM từ session state
if st.session_state.get('current_llm_info'):
    llm_info = st.session_state.current_llm_info
    config_str = ", ".join([f"{k}={v}" for k, v in llm_info.get('config', {}).items()])
    status_text = f"LLM Ready ({llm_info['llm_type']}: {llm_info['model_name']})"
    if config_str:
        status_text += f" with config: {config_str}"
    st.sidebar.success(status_text)
else:
    st.sidebar.warning("LLM Not Initialized")


# --- Hiển thị thông tin Schema đã đọc được (Main Panel) (Giữ nguyên) ---
st.subheader("Thông tin Schema đã Tải")

if st.session_state.schema_info:
    if st.session_state.schema_info:
        st.write("Cấu trúc schema database tìm thấy (không bao gồm schema hệ thống):")

        for schema, tables in st.session_state.schema_info.items():
            with st.expander(f"Schema: **`{schema}`** ({len(tables)} bảng)"):
                if not tables:
                    st.write(f"Schema '{schema}' không có bảng nào khả dụng (hoặc không đọc được).")
                else:
                    for table, columns in tables.items():
                         st.markdown(f"**Bảng: `{schema}.{table}`** ({len(columns)} cột)")

                         if not columns:
                             st.write("Bảng này không có cột nào (hoặc không đọc được thông tin cột).")
                         else:
                             columns_df = pd.DataFrame(columns)
                             cols_to_display_order = ['name', 'type', 'primary_key', 'nullable', 'default', 'autoincrement']
                             cols_to_display_filtered = [col for col in cols_to_display_order if col in columns_df.columns]
                             st.dataframe(columns_df[cols_to_display_filtered], use_container_width=True)

                         st.markdown("---")

    else:
         st.warning("Kết nối thành công nhưng không tìm thấy schema hoặc bảng nào khả dụng ngoài các schema hệ thống.")

else:
    st.info("Vui lòng nhập Database URI và nhấn 'Kết nối & Đọc Schema' để bắt đầu.")


# --- Giai đoạn 2: Tạo SQL (UI cho User Query) (Giữ nguyên logic gọi hàm get_sql_from_llm) ---
st.subheader("Tạo SQL Query từ Câu hỏi")

if st.session_state.schema_info is None or not st.session_state.schema_info:
    st.warning("Vui lòng kết nối database và đọc schema trước khi hỏi.")
elif st.session_state.chat_llm is None:
    st.warning("Vui lòng khởi tạo LLM trước khi hỏi.")
else:
    user_query = st.text_area(
        "Nhập câu hỏi của bạn về dữ liệu (ví dụ: 'Liệt kê 5 địa chỉ và giá cao nhất từ bảng haus_mieten_test trong schema raw'):",
        key="user_query_input",
        height=100
    )

    if st.button("Tạo SQL Query từ Câu hỏi"):
        if not user_query:
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            st.info("Đang tạo SQL query với LLM...")
            st.session_state.generated_sql = None

            generated_sql, error_msg = get_sql_from_llm(user_query) # Hàm này sử dụng LLM từ session state

            if error_msg:
                st.error(error_msg)
            else:
                st.success("Đã tạo SQL query:")
                st.code(generated_sql, language="sql")
                st.session_state.generated_sql = generated_sql


# --- Placeholder cho Giai đoạn 4 (Thực thi SQL) (Giữ nguyên) ---
st.subheader("Thực thi SQL và Hiển thị Kết quả (Giai đoạn 4)")
if st.session_state.get('generated_sql'):
    st.info("Câu lệnh SQL đã được tạo ở trên. Giai đoạn tiếp theo sẽ là thêm nút 'Thực thi SQL' và hiển thị kết quả dưới dạng bảng/biểu đồ.")
    # Nút "Thực thi SQL" và logic thực thi sẽ thêm vào ở Giai đoạn 4
    # if st.button("Thực thi SQL"):
    #     # ... code thực thi SQL ở đây ...
    pass
else:
     st.info("Tạo câu hỏi ở trên để generate SQL.")


# --- Chú ý cuối file (Giữ nguyên) ---
print("Streamlit app is running. Waiting for user interaction.")