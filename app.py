import streamlit as st
import os
import sys
from pathlib import Path
import re
import json
import requests
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
import time  # 添加time模組用於打字效果

# 載入環境變數
load_dotenv()

# 創建一個函數來獲取環境變數，本地開發時優先使用.env，生產環境優先使用st.secrets
def get_env_variable(key, default_value=""):
    """獲取環境變數，本地開發時優先使用.env，生產環境優先使用st.secrets"""
    # 先從環境變數獲取
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
        
    # 如果環境變數中沒有，再嘗試從st.secrets獲取
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        return default_value

# 設置頁面配置和標題
st.set_page_config(page_title="數位分身系統", layout="wide", initial_sidebar_state="collapsed")

# 載入數位分身提示詞
def load_system_prompt(prompt_filename="wang.txt"):
    """從 system_prompts 資料夾載入系統提示詞檔案"""
    prompt_folder = Path("system_prompts")
    prompt_file = prompt_folder / prompt_filename
    
    if not prompt_folder.exists():
        try:
            prompt_folder.mkdir(exist_ok=True)
            st.warning(f"已建立 {prompt_folder} 資料夾，但尚未包含提示詞檔案")
            return ""
        except Exception as e:
            st.error(f"無法建立 system_prompts 資料夾: {str(e)}")
            return ""
            
    if not prompt_file.exists():
        st.warning(f"找不到提示詞檔案 {prompt_file}，請確保該檔案存在")
        return ""
        
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"讀取提示詞檔案時出錯: {str(e)}")
        return ""

# 驗證功能
def check_password():
    """返回`True`如果用戶輸入正確的用戶名和密碼"""
    
    # 從環境變數讀取認證信息
    correct_username = get_env_variable("AUTH_USERNAME", "admin")
    correct_password = get_env_variable("AUTH_PASSWORD", "password")
    
    # 檢查是否已經登入
    if "authentication_status" in st.session_state and st.session_state["authentication_status"]:
        return True
    
    # 顯示登入表單
    st.title("數位分身系統 (Beta)")
    st.header("登入")
    
    # 使用表單，可以讓用戶按下 Enter 鍵提交
    with st.form("login_form"):
        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        submit_button = st.form_submit_button("登入")
        
        if submit_button:
            if username == correct_username and password == correct_password:
                st.session_state["authentication_status"] = True
                st.rerun()  # 強制頁面重新渲染，確保表單消失
                return True
            else:
                st.error("帳號或密碼錯誤")
                return False
    
    return False

# 檢查登入狀態
if not check_password():
    st.stop()  # 如果未登入，停止應用程式執行

# 以下是應用程式主體部分，只有在通過驗證後才會執行
st.title("數位分身系統 (Beta)")

# 初始化會話狀態
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_persona" not in st.session_state:
    st.session_state.use_persona = False

if "prompt_filename" not in st.session_state:
    st.session_state.prompt_filename = "wang.txt"

if "knowledge_table" not in st.session_state:
    st.session_state.knowledge_table = get_env_variable("KNOWLEDGE_TABLE", "knowledge_base")  # 從環境變數讀取資料表名稱

# 載入數位分身提示詞
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = load_system_prompt(st.session_state.prompt_filename)

# 取得OpenAI API金鑰
openai_api_key = get_env_variable("OPENAI_API_KEY", "")

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=openai_api_key)

# 讀取環境變數中的模型名稱
llm_model = get_env_variable("LLM_MODEL", "gpt-4o")

# 側邊欄設置
with st.sidebar:
    st.header("設置")
    
    # 讀取環境變數中的API金鑰
    supabase_url = get_env_variable("SUPABASE_URL", "")
    supabase_key = get_env_variable("SUPABASE_KEY", "")
    voyage_api_key = get_env_variable("VOYAGE_API_KEY", "")
    voyage_model = get_env_variable("VOYAGE_MODEL", "voyage-2")
    
    # 進階設置
    with st.expander("進階設置"):
        st.session_state.knowledge_table = st.text_input("知識庫資料表名稱", value=st.session_state.knowledge_table)
        llm_temperature = st.slider("AI 回答溫度", min_value=0.0, max_value=1.0, value=float(get_env_variable("LLM_TEMPERATURE", "0.3")), step=0.1)
        
        # 輸入系統提示詞檔案名稱
        prompt_filename = st.text_input("系統提示詞檔案名稱", value=st.session_state.prompt_filename)
        if prompt_filename != st.session_state.prompt_filename:
            st.session_state.prompt_filename = prompt_filename
            new_prompt = load_system_prompt(prompt_filename)
            if new_prompt:
                st.session_state.custom_prompt = new_prompt
                st.success(f"已載入提示詞檔案: {prompt_filename}")
    # 添加登出按鈕
    if st.button("登出"):
        st.session_state["authentication_status"] = False
        st.rerun()

# 初始化 Supabase 客戶端（如果提供了必要憑證）
if supabase_url and supabase_key:
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
    except Exception as e:
        supabase = None
else:
    supabase = None

# === RAG 功能實現 ===

def generate_embeddings(text):
    """使用 Voyage API 生成文本的向量嵌入"""
    try:
        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {voyage_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": voyage_model,
                "input": text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        st.error(f"生成嵌入時出錯: {str(e)}")
        return None

def extract_keywords(query):
    """從查詢中提取可能的關鍵詞"""
    # 常見的問句開頭和修飾詞，可能會干擾精確配斷
    stopwords = [
        '想知道', '請問', '告訴我', '關於', '誰是', '是誰', '什麼是', '的', '是',
        '嗎', '呢', '啊', '吧', '了', '哦', '喔', '耶', '呀', '？', '?',
        '請', '幫我', '可以', '能', '應該', '會', '要', '需要'
    ]
    
    # 清理標點符號和特殊字符
    cleaned_query = re.sub(r'[^\w\s]', ' ', query)
    
    # 移除停用詞
    for word in stopwords:
        cleaned_query = cleaned_query.replace(word, ' ')
    
    # 分割並過濾空字串，保持原始順序
    keywords = []
    seen = set()  # 用於去重
    for k in cleaned_query.split():
        k = k.strip()
        if k and k not in seen:
            keywords.append(k)
            seen.add(k)
    
    return keywords

def extract_core_question_with_llm(query):
    """使用LLM提取查詢的核心問題和關鍵詞"""
    try:
        system_prompt = """你是一個專業的文本分析工具。
        你的任務是從用戶的問題中提取核心問題和關鍵詞。
        請分析用戶的問題，去除禮貌用語、修飾詞和冗餘內容，
        只保留能表達核心意思的最少詞語。
        
        請以JSON格式返回結果，包含兩個字段：
        1. core_question: 精簡後的核心問題
        2. keywords: 關鍵詞列表
        """
        
        user_prompt = f"請分析這個問題並提取核心問題和關鍵詞：{query}"
        
        # 使用單獨的環境變數來設置核心提取器的模型
        core_extractor_model = get_env_variable("CORE_EXTRACTOR_MODEL", "gpt-3.5-turbo")
        
        response = client.chat.completions.create(
            model=core_extractor_model, # 使用較小的模型以節省成本
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        parsed_result = json.loads(result)
        
        return parsed_result
    except Exception as e:
        st.error(f"使用LLM提取核心問題時出錯: {str(e)}")
        # 如果LLM處理失敗，回退到原始的關鍵詞提取方法
        fallback_keywords = extract_keywords(query)
        return {
            "core_question": query,
            "keywords": fallback_keywords
        }

def search_knowledge(query, match_threshold=0.7, match_count=6):
    """從向量知識庫中搜索相關的知識點"""
    # 不再直接顯示狀態訊息，只處理搜索邏輯
    
    if not supabase:
        return []
        
    # 生成查詢的嵌入向量
    query_embedding = generate_embeddings(query)
    
    if not query_embedding:
        return []
    
    try:
        # 1. 首先嘗試直接檢查是否有完全配對的關鍵詞
        exact_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", f"%{query}%").execute()
        if hasattr(exact_result, 'data') and exact_result.data:
            # 添加一個相似度欄位以與向量搜索結果格式一致
            for item in exact_result.data:
                item['similarity'] = 1.0  # 設定為最高相似度
            return exact_result.data
        
        # 2. 嘗試提取關鍵詞進行部分配對
        keywords = extract_keywords(query)
        if keywords:
            for keyword in keywords:
                if len(keyword) >= 2:  # 確保關鍵詞至少2個字
                    keyword_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", f"%{keyword}%").execute()
                    if hasattr(keyword_result, 'data') and keyword_result.data:
                        # 添加相似度欄位
                        for item in keyword_result.data:
                            item['similarity'] = 0.95  # 設定為稍低於完全配對的相似度
                        return keyword_result.data
        
        # 3. 使用向量搜索進行相似性搜索
        result = supabase.rpc(
            "match_knowledge", 
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }
        ).execute()
        
        # 處理結果
        if hasattr(result, 'data') and result.data:
            return result.data
        else:
            # 如果向量搜索未找到結果，進一步降低閾值再試一次
            if match_threshold > 0.5:
                return search_knowledge(query, match_threshold=0.5, match_count=match_count)
            
            # 4. 最後，搜索所有欄位作為後備選項
            backup_result = supabase.table(st.session_state.knowledge_table).select("*").execute()
            if hasattr(backup_result, 'data') and backup_result.data:
                # 簡單的文本配對
                matched_items = []
                for item in backup_result.data:
                    combined_text = f"{item['concept']}: {item['explanation']}"
                    # 檢查是否包含查詢中的任何關鍵詞
                    if any(keyword in combined_text for keyword in keywords):
                        item['similarity'] = 0.6  # 設定為較低的相似度
                        matched_items.append(item)
                
                if matched_items:
                    return matched_items
            
            return []
    except Exception as e:
        return []


def generate_direct_response(query):
    """當知識庫中沒有相關信息時，直接使用 OpenAI 回答"""
    current_status = "我正在努力生成回答..."
    
    try:
        # 獲取系統提示詞
        system_content = st.session_state.custom_prompt or """你是一個謹慎、實事求是的助手。對於用戶的問題，如果你不確定答案或沒有足夠的信息，請坦率地表示「我沒有足夠的信息來回答這個問題」或「我不確定，需要更多資料才能給出準確回答」。避免猜測或提供可能不準確的信息。尤其對於特定人物、組織或專業領域的問題，如果你沒有確切資料，更應明確表示不知道，而不是提供可能的幻覺信息。"""
        
        response = client.chat.completions.create(
            model=llm_model,  # 使用從環境變數讀取的模型
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            temperature=llm_temperature
        )
        
        return response.choices[0].message.content.strip(), current_status
    except Exception as e:
        current_status = f"生成直接回答時出錯: {str(e)}"
        return f"生成回答時發生錯誤: {str(e)}", current_status

# 聊天界面部分
st.subheader("數位分身聊天")

# 移除之前的狀態訊息相關CSS樣式定義，保留其他樣式
st.markdown("""
<style>
.block-container {
    padding-bottom: 1rem;
}
.stChatFloatingInputContainer {
    padding-bottom: 0px;
    bottom: 0px;
}
.stChatInput {
    margin-bottom: 0px;
}
[data-testid="stChatMessageContent"] {
    border-radius: 12px;
    padding: 10px;
}
.assistant-message p {
    margin: 0;
    padding: 0;
}
</style>
""", unsafe_allow_html=True)

# 初始化狀態歷史（如果不存在）
if "status_history" not in st.session_state:
    st.session_state.status_history = []

# 顯示聊天歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 輸入框
prompt = st.chat_input("請輸入您的問題...")

# 處理用戶輸入
if prompt:
    # 清空狀態歷史
    st.session_state.status_history = []
    
    # 添加用戶消息到聊天記錄
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 立即顯示用戶訊息
    with st.chat_message("user"):
        st.write(prompt)
    
    # 創建一個固定位置的狀態訊息 - 使用單一佔位元素
    status_msg = st.empty()
    
    # 檢查必要的API金鑰是否設置
    if not openai_api_key:
        with st.chat_message("assistant"):
            st.error("請設置 OpenAI API Key 才能使用對話功能。")
        st.info("如何設置API金鑰: \n\n"
                "本地開發: 在專案根目錄創建.env檔案，然後添加 `OPENAI_API_KEY=your_key`\n\n"
                "Streamlit Cloud: 在應用程式設置中添加密鑰，設置名稱為 `OPENAI_API_KEY`")
        st.stop()  # 如果沒有設置API金鑰，停止執行
    
    # 更新狀態訊息函數 - 只在一個位置更新
    def update_status(status):
        # 保存狀態歷史
        if status not in st.session_state.status_history:
            st.session_state.status_history.append(status)
        
        # 在固定位置顯示當前狀態
        with status_msg.container():
            with st.chat_message("assistant"):
                st.info(status)
    
    # 初始狀態
    update_status("思考中...")
    
    # 修改 RAG 函數來使用我們的狀態更新函數
    def rag_with_status(query):
        update_status("我正在思考你的問題...")
        core_result = extract_core_question_with_llm(query)
        core_question = core_result.get("core_question", query)
        st.session_state.last_core_question = core_question
        st.session_state.last_keywords = core_result.get("keywords", [])
        
        update_status("正在從知識庫尋找相關資訊...")
        knowledge_points = search_knowledge(core_question)
        st.session_state.last_knowledge_points = knowledge_points
        
        if not knowledge_points:
            update_status("好像沒有未找到相關知識點")
            return direct_with_status(query)
        
        update_status("找到了相關資訊，正在生成回答...")
        context = "\n".join([f"概念: {item['concept']}\n解釋: {item['explanation']}" for item in knowledge_points])
        
        prompt = f"""

        知識點:
        {context}
        
        原始問題: {query}
        核心問題: {core_question}
        
        回答:"""
        
        # 獲取系統提示詞
        system_content = st.session_state.custom_prompt
        
        # 生成回答
        try:
            response = client.chat.completions.create(
                model=llm_model,  # 使用從環境變數讀取的模型
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=llm_temperature
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            update_status(f"生成回答時出錯: {str(e)}")
            return f"生成回答時發生錯誤: {str(e)}"
    
    # 直接回答的函數，帶狀態更新
    def direct_with_status(query):
        update_status("我正在努力生成回答...")
        try:
            # 獲取系統提示詞
            system_content = st.session_state.custom_prompt or """你是一個謹慎、實事求是的助手。對於用戶的問題，如果你不確定答案或沒有足夠的信息，請坦率地表示「我沒有足夠的信息來回答這個問題」或「我不確定，需要更多資料才能給出準確回答」。避免猜測或提供可能不準確的信息。尤其對於特定人物、組織或專業領域的問題，如果你沒有確切資料，更應明確表示不知道，而不是提供可能的幻覺信息。"""
            
            response = client.chat.completions.create(
                model=llm_model,  # 使用從環境變數讀取的模型
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ],
                temperature=llm_temperature
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            update_status(f"生成直接回答時出錯: {str(e)}")
            return f"生成回答時發生錯誤: {str(e)}"
    
    # 使用知識庫或直接回答
    if supabase and voyage_api_key:
        response_text = rag_with_status(prompt)
    else:
        response_text = direct_with_status(prompt)
    
    # 清空狀態訊息
    status_msg.empty()
    
    # 顯示最終回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = response_text
        
        # 逐字顯示回答
        displayed_message = ""
        for i in range(len(full_response) + 1):
            displayed_message = full_response[:i]
            message_placeholder.markdown(displayed_message)
            time.sleep(0.01)  # 控制打字速度
    
    # 添加助手消息到聊天記錄
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 重新載入頁面以重置聊天界面
    st.rerun() 