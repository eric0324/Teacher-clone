import streamlit as st
import json
import time
from pathlib import Path

# 導入自定義模塊
from utils.config import load_config, get_env_variable, load_system_prompt
from utils.ui import setup_ui, setup_sidebar, display_chat_history
from utils.auth import check_password
from utils.knowledge import search_knowledge, extract_core_question_with_llm
from utils.llm_providers import (
    generate_openai_response, 
    generate_claude_response, 
    generate_deepseek_response
)

# 設置頁面配置和標題
st.set_page_config(page_title="數位分身系統", layout="wide", initial_sidebar_state="collapsed")

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
    st.session_state.knowledge_table = get_env_variable("KNOWLEDGE_TABLE", "knowledge_base")  

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "deepseek"  # 預設使用 DeepSeek

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True  # 預設啟用串流回應

if "memory_length" not in st.session_state:
    st.session_state.memory_length = 5
    
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = load_system_prompt(st.session_state.prompt_filename)

# 載入配置
config = load_config()

# 將配置添加到會話狀態中
st.session_state.supabase = config.get("supabase")

# 設置側邊欄
setup_sidebar()

# 聊天界面部分
st.subheader("數位分身聊天")

# 設置UI樣式
setup_ui()

# 初始化狀態歷史（如果不存在）
if "status_history" not in st.session_state:
    st.session_state.status_history = []

# 顯示聊天歷史
display_chat_history()

# RAG 查詢處理函數
def rag_with_status(query, update_status):
    """使用RAG (Retrieval Augmented Generation) 處理查詢"""
    update_status("我正在思考你的問題...")
    try:
        core_result = extract_core_question_with_llm(query)
        core_question = core_result.get("core_question", query)
        st.session_state.last_core_question = core_question
        st.session_state.last_keywords = core_result.get("keywords", [])
        
        update_status("正在從知識庫尋找相關資訊...")
        try:
            knowledge_points = search_knowledge(core_question)
            st.session_state.last_knowledge_points = knowledge_points
        except Exception as e:
            update_status(f"搜索知識庫時出錯: {str(e)}，將直接使用AI生成回答")
            st.error(f"搜索知識時出錯: {str(e)}")
            return direct_with_status(query, update_status)
        
        if not knowledge_points:
            update_status("好像沒有找到相關知識點")
            return direct_with_status(query, update_status)
        
        update_status("找到了相關資訊，正在生成回答...")
        
        # 準備知識點信息
        knowledge_info = []
        for item in knowledge_points:
            match_info = f"({item.get('match_type', '未知匹配類型')}, 相似度: {item.get('similarity', 0):.2f})"
            knowledge_info.append(f"概念: {item['concept']} {match_info}\n解釋: {item['explanation']}")
        
        context = "\n\n".join(knowledge_info)
        
        prompt = f"""
        知識點:
        {context}
        
        原始問題: {query}
        核心問題: {core_question}
        
        請根據上述知識點信息回答用戶的問題。回答應該清晰、準確，並基於提供的知識點。
        請以自然、對話的方式回答，不要直接複製知識點的文本，而是根據內容提供信息豐富的解釋。
        不需要在回答中提及匹配類型和相似度，這些只是用來幫助你理解知識點的重要性。
        如果知識點中包含矛盾的信息，請優先考慮相似度較高的知識點。
        
        回答:
        """
        
        # 獲取系統提示詞
        system_content = st.session_state.custom_prompt
        
        # 構建對話歷史，包含最近的對話
        messages = [{"role": "system", "content": system_content}]
        
        # 添加歷史對話記錄（只保留最近的幾輪）
        memory_limit = st.session_state.memory_length
        recent_messages = st.session_state.messages[-2*memory_limit:] if len(st.session_state.messages) > 2*memory_limit else st.session_state.messages[:]
        
        # 添加歷史對話記錄
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加當前問題和上下文
        messages.append({"role": "user", "content": prompt})
        
        # 生成回答
        return generate_response(messages)
    except Exception as e:
        update_status(f"處理問題時發生錯誤: {str(e)}，轉用直接回答")
        return direct_with_status(query, update_status)

# 直接回答的函數，帶狀態更新
def direct_with_status(query, update_status):
    """直接使用LLM生成回答，不使用知識庫"""
    update_status("我正在努力生成回答...")
    try:
        # 獲取系統提示詞
        system_content = st.session_state.custom_prompt
        
        # 構建對話歷史，包含最近的對話
        messages = [{"role": "system", "content": system_content}]
        
        # 添加歷史對話記錄（只保留最近的幾輪）
        memory_limit = st.session_state.memory_length
        recent_messages = st.session_state.messages[-2*memory_limit:] if len(st.session_state.messages) > 2*memory_limit else st.session_state.messages[:]
        
        # 添加歷史對話記錄
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加當前問題
        messages.append({"role": "user", "content": query})
        
        # 生成回答
        return generate_response(messages)
    except Exception as e:
        update_status(f"生成直接回答時出錯: {str(e)}")
        return f"生成回答時發生錯誤: {str(e)}"

def generate_response(messages):
    """根據選定的LLM供應商生成回答"""
    llm_provider = st.session_state.llm_provider
    use_streaming = st.session_state.use_streaming
    
    if llm_provider == "openai":
        llm_model = get_env_variable("LLM_MODEL", "gpt-4o")
        if use_streaming:
            response, _ = generate_openai_response(
                messages=messages,
                model=llm_model,
                streaming=True
            )
            return response, "串流"
        else:
            response_text, _ = generate_openai_response(
                messages=messages,
                model=llm_model,
                streaming=False
            )
            return response_text
    elif llm_provider == "claude":
        claude_model = get_env_variable("CLAUDE_MODEL", "claude-3-5-sonnet-20240620-v1")
        if use_streaming:
            response, _ = generate_claude_response(
                messages=messages,
                model=claude_model,
                streaming=True
            )
            return response, "串流"
        else:
            response_text, _ = generate_claude_response(
                messages=messages,
                model=claude_model,
                streaming=False
            )
            return response_text
    elif llm_provider == "deepseek":
        deepseek_model = get_env_variable("DEEPSEEK_MODEL", "deepseek-chat")
        if use_streaming:
            response, _ = generate_deepseek_response(
                messages=messages,
                model_id=deepseek_model,
                streaming=True
            )
            return response, "串流"
        else:
            response_text, _ = generate_deepseek_response(
                messages=messages,
                model_id=deepseek_model,
                streaming=False
            )
            return response_text
    else:
        return "未支援的 LLM 供應商"

def display_streaming_response(stream_response, message_placeholder):
    """顯示串流回應"""
    full_response = ""
    llm_provider = st.session_state.llm_provider
    
    if llm_provider == "openai":
        # 處理 OpenAI 串流
        for chunk in stream_response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response)
                elif hasattr(chunk.choices[0], 'text'):
                    # 舊版 API 可能使用 text 而非 content
                    content = chunk.choices[0].text
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response)
    
    elif llm_provider == "claude":
        # 處理 Claude 串流
        for chunk in stream_response:
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                content = chunk.delta.content
                if content:
                    full_response += content
                    message_placeholder.markdown(full_response)
    
    elif llm_provider == "deepseek":
        # 處理 DeepSeek 串流
        for line in stream_response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    json_data = json.loads(line[6:])
                    if 'choices' in json_data and json_data['choices'] and 'delta' in json_data['choices'][0]:
                        content = json_data['choices'][0]['delta'].get('content', '')
                        if content:
                            full_response += content
                            message_placeholder.markdown(full_response)
    
    return full_response

def simulate_typing(message_placeholder, text):
    """模擬打字效果顯示回應"""
    displayed_message = ""
    for i in range(len(text) + 1):
        displayed_message = text[:i]
        message_placeholder.markdown(displayed_message)
        time.sleep(0.01)  # 控制打字速度
    return text

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
    
    # 創建一個固定位置的狀態訊息
    status_msg = st.empty()
    
    # 檢查必要的API金鑰是否設置
    api_check_passed = False
    llm_provider = st.session_state.llm_provider
    
    if llm_provider == "openai" and not get_env_variable("OPENAI_API_KEY", ""):
        with st.chat_message("assistant"):
            st.error("請設置 OpenAI API Key 才能使用對話功能。")
        st.info("如何設置API金鑰: \n\n"
                "本地開發: 在專案根目錄創建.env檔案，然後添加 `OPENAI_API_KEY=your_key`\n\n"
                "Streamlit Cloud: 在應用程式設置中添加密鑰，設置名稱為 `OPENAI_API_KEY`")
    elif llm_provider == "claude" and not get_env_variable("CLAUDE_API_KEY", ""):
        with st.chat_message("assistant"):
            st.error("請設置 Claude API Key 才能使用 Claude 模型。")
        st.info("如何設置 Claude API Key: \n\n"
                "本地開發: 在專案根目錄創建.env檔案，然後添加 `CLAUDE_API_KEY=your_key`\n\n"
                "Streamlit Cloud: 在應用程式設置中添加密鑰，設置名稱為 `CLAUDE_API_KEY`")
    elif llm_provider == "deepseek" and not get_env_variable("DEEPSEEK_API_KEY", ""):
        with st.chat_message("assistant"):
            st.error("請設置 DeepSeek API Key 才能使用 DeepSeek 模型。")
        st.info("如何設置 DeepSeek API Key: \n\n"
                "本地開發: 在專案根目錄創建.env檔案，然後添加 `DEEPSEEK_API_KEY=your_key`\n\n"
                "Streamlit Cloud: 在應用程式設置中添加密鑰，設置名稱為 `DEEPSEEK_API_KEY`")
    else:
        api_check_passed = True
    
    if not api_check_passed:
        st.stop()  # 如果沒有設置API金鑰，停止執行
    
    # 更新狀態訊息函數
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
    
    # 使用知識庫或直接回答
    voyage_api_key = get_env_variable("VOYAGE_API_KEY", "")
    if st.session_state.supabase and voyage_api_key:
        response_result = rag_with_status(prompt, update_status)
    else:
        response_result = direct_with_status(prompt, update_status)
    
    # 判斷回應類型
    if isinstance(response_result, tuple) and response_result[1] == "串流":
        stream_response = response_result[0]
        is_streaming = True
    else:
        response_text = response_result
        is_streaming = False
    
    # 清空狀態訊息
    status_msg.empty()
    
    # 顯示回答（串流或完整）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if is_streaming:
            # 處理串流響應
            full_response = display_streaming_response(stream_response, message_placeholder)
            response_text = full_response  # 保存完整的回答用於添加到聊天歷史
        else:
            # 非串流模式，使用打字效果
            response_text = simulate_typing(message_placeholder, response_text)
    
    # 添加助手消息到聊天記錄
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 重新載入頁面以重置聊天界面
    st.rerun() 