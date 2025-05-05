import streamlit as st
from utils.config import get_env_variable, load_system_prompt
import os

def setup_ui():
    """設置UI樣式和元素"""
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

    /* 表格樣式 */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 0.9em;
    }

    th {
        background-color: #f8f9fa;
        color: #333;
        font-weight: bold;
        text-align: left;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }

    td {
        padding: 8px 10px;
        border: 1px solid #e0e0e0;
    }

    tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    tr:hover {
        background-color: #f1f3f4;
    }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """設置側邊欄內容"""
    with st.sidebar:
        st.header("設置")
        
        # 讀取環境變數中的API金鑰
        supabase_url = get_env_variable("SUPABASE_URL", "")
        supabase_key = get_env_variable("SUPABASE_KEY", "")
        voyage_api_key = get_env_variable("VOYAGE_API_KEY", "")
        voyage_model = get_env_variable("VOYAGE_MODEL", "voyage-2")
        
        # LLM 供應商選擇
        llm_provider_options = ["deepseek", "claude", "openai"]
        selected_provider = st.selectbox(
            "選擇大語言模型供應商", 
            llm_provider_options,
            index=llm_provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in llm_provider_options else 0
        )
        
        if selected_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = selected_provider
            st.success(f"已切換到 {selected_provider} 模型")
        
        # 串流模式設置
        streaming_enabled = st.toggle("啟用串流模式", st.session_state.use_streaming, 
                                    help="啟用串流模式後，回答將逐步顯示，不必等待完整回答生成。")
        if streaming_enabled != st.session_state.use_streaming:
            st.session_state.use_streaming = streaming_enabled
            st.success("串流模式設定已更新")
        
        # 根據供應商顯示對應的模型選擇
        if st.session_state.llm_provider == "openai":
            openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            llm_model = get_env_variable("LLM_MODEL", "gpt-4o")
            selected_openai_model = st.selectbox(
                "選擇 OpenAI 模型", 
                openai_models, 
                index=openai_models.index(llm_model) if llm_model in openai_models else 0
            )
        elif st.session_state.llm_provider == "claude":
            claude_models = [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20240620-v1",
                "claude-3-opus-20240229-v1",
                "claude-3-sonnet-20240229-v1",
                "claude-3-haiku-20240307-v1"
            ]
            claude_model = get_env_variable("CLAUDE_MODEL", "claude-3-5-sonnet-20240620-v1")
            selected_claude_model = st.selectbox(
                "選擇 Claude 模型", 
                claude_models, 
                index=claude_models.index(claude_model) if claude_model in claude_models else 0
            )
            
            # 保存選定的 Claude 模型
            if selected_claude_model != claude_model:
                os.environ["CLAUDE_MODEL"] = selected_claude_model
                st.success(f"已切換到 {selected_claude_model} 模型")
                
        elif st.session_state.llm_provider == "deepseek":
            # DeepSeek 模型選擇
            deepseek_models = ["deepseek-chat"]
            deepseek_model = get_env_variable("DEEPSEEK_MODEL", "deepseek-chat")
            selected_deepseek_model = st.selectbox(
                "選擇 DeepSeek 模型", 
                deepseek_models, 
                index=deepseek_models.index(deepseek_model) if deepseek_model in deepseek_models else 0
            )
        
        # 進階設置
        with st.expander("進階設置"):
            st.session_state.knowledge_table = st.text_input("知識庫資料表名稱", value=st.session_state.knowledge_table)
            llm_temperature = st.slider("AI 回答溫度", min_value=0.0, max_value=1.0, value=float(get_env_variable("LLM_TEMPERATURE", "0.3")), step=0.1)
            
            # 添加記憶長度設置
            memory_length = st.slider("對話記憶長度", min_value=1, max_value=10, value=st.session_state.memory_length, step=1)
            if memory_length != st.session_state.memory_length:
                st.session_state.memory_length = memory_length
            
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

def display_chat_history():
    """顯示對話歷史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"]) 