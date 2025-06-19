import os
import streamlit as st
from dotenv import load_dotenv
import openai
from supabase import create_client
from pathlib import Path

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

def load_config():
    """載入所有配置及API連接"""
    # 載入環境變數
    load_dotenv()
    
    # 讀取環境變數
    config = {
        # OpenAI設置
        "openai_api_key": get_env_variable("OPENAI_API_KEY", ""),
        "llm_model": get_env_variable("LLM_MODEL", "gpt-4o"),
        "llm_temperature": float(get_env_variable("LLM_TEMPERATURE", "0.3")),
        
        # DeepSeek設置
        "deepseek_api_key": get_env_variable("DEEPSEEK_API_KEY", ""),
        "deepseek_model": get_env_variable("DEEPSEEK_MODEL", "deepseek-chat"),
        
        # Claude設置
        "claude_api_key": get_env_variable("CLAUDE_API_KEY", ""),
        "claude_model": get_env_variable("CLAUDE_MODEL", "claude-3-5-sonnet-20240620-v1"),
        
        # Supabase設置
        "supabase_url": get_env_variable("SUPABASE_URL", ""),
        "supabase_key": get_env_variable("SUPABASE_KEY", ""),
        
        # Voyage API設置
        "voyage_api_key": get_env_variable("VOYAGE_API_KEY", ""),
        "voyage_model": get_env_variable("VOYAGE_MODEL", "voyage-2"),
        
        # 系統提示詞和知識庫設置
        "knowledge_table": get_env_variable("KNOWLEDGE_TABLE", "knowledge_base"),
        
        # 認證設置
        "auth_username": get_env_variable("AUTH_USERNAME", "admin"),
        "auth_password": get_env_variable("AUTH_PASSWORD", "password"),
    }
    
    # 初始化OpenAI客戶端 (使用舊版API)
    if config["openai_api_key"]:
        openai.api_key = config["openai_api_key"]
        config["openai_client"] = openai
    else:
        config["openai_client"] = None
    
    # 初始化Claude客戶端
    config["claude_client"] = None
    if config["claude_api_key"]:
        try:
            from anthropic import Anthropic
            config["claude_client"] = Anthropic(api_key=config["claude_api_key"])
        except Exception as e:
            st.error(f"初始化 Claude 客戶端時出錯: {str(e)}")
    
    # 初始化Supabase客戶端
    config["supabase"] = None
    if config["supabase_url"] and config["supabase_key"]:
        try:
            config["supabase"] = create_client(config["supabase_url"], config["supabase_key"])
        except Exception as e:
            st.error(f"初始化 Supabase 客戶端時出錯: {str(e)}")
            
    return config

def load_system_prompt(prompt_name="mj"):
    """從 system_prompts 資料表或資料夾載入系統提示詞"""
    import streamlit as st
    
    # 嘗試從Supabase獲取系統提示詞
    supabase = st.session_state.get('supabase')
    if supabase:
        try:
            # 從system_prompts資料表獲取指定名稱的系統提示詞
            result = supabase.table("system_prompts").select("*").eq("name", prompt_name).execute()
         
            # 檢查是否找到結果
            if hasattr(result, 'data') and result.data:
                # 顯示找到的系統提示詞
                system_prompt = result.data[0]['system_prompt']
                # 返回找到的系統提示詞
                return system_prompt
            else:
                st.warning(f"在資料庫中找不到名稱為 {prompt_name} 的系統提示詞")
                return None
        except Exception as e:
            st.error(f"從資料庫獲取系統提示詞時出錯: {str(e)}")
            return None
    else:
        st.warning("未設定 Supabase 連接，無法從資料庫獲取系統提示詞")
        return None