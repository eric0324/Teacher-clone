import requests
from utils.config import get_env_variable

def generate_openai_response(messages, model):
    """使用 OpenAI API 生成回答，支援串流和非串流模式"""
    import openai
    
    openai_api_key = get_env_variable("OPENAI_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not openai_api_key:
        return "OpenAI API 金鑰未設定", "配置錯誤"
    
    try:
        # 設置 API 金鑰
        openai.api_key = openai_api_key
        
        # 依據模型設定適當的 max_tokens
        max_tokens_mapping = {
            "gpt-4": 10000,
        }
        max_tokens = max_tokens_mapping.get(model, 10000)
        
        # 確保所有消息內容不是 None
        sanitized_messages = []
        for msg in messages:
            if msg["content"] is None:
                msg_copy = msg.copy()
                msg_copy["content"] = ""
                sanitized_messages.append(msg_copy)
            else:
                sanitized_messages.append(msg)
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=sanitized_messages,
            temperature=llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )
        return response, "串流"
    except Exception as e:
        return f"OpenAI API 錯誤: {str(e)}", "錯誤"

def generate_claude_response(messages, model):
    """使用 Anthropic Claude API 生成回答，支援串流和非串流模式"""
    from anthropic import Anthropic
    import time
    from anthropic._exceptions import APIStatusError
    
    claude_api_key = get_env_variable("CLAUDE_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not claude_api_key:
        return "Claude API 金鑰未設定", "配置錯誤"
    
    max_tokens_mapping = {
        "claude-3-7-sonnet-20250219": 10000,
        "claude-3-5-sonnet-20241022": 8000,
    }
    
    max_tokens = max_tokens_mapping.get(model, 8000)
    
    try:
        # 創建Anthropic客戶端，使用適當的API版本設置
        client = Anthropic(
            api_key=claude_api_key,
            default_headers={
                "anthropic-version": "2023-06-01" # 設置API版本
            }
        )
        
        # 處理系統提示
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        system_content = system_message["content"] if system_message else None
        
        # 從消息列表中移除系統消息
        filtered_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # 準備消息格式
        formatted_messages = []
        for msg in filtered_messages:
            role = "user" if msg["role"] == "user" else "assistant"
            formatted_messages.append({"role": role, "content": msg["content"]})
            
        response = client.messages.create(
            model=model,
            messages=formatted_messages,
            system=system_content,
            temperature=llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )
        return response, "串流"
            
    except APIStatusError as e:
        # 檢查是否為過載錯誤
        error_json = e.body if hasattr(e, 'body') else {}
        is_overloaded = (
            isinstance(error_json, dict) and 
            error_json.get('error', {}).get('type') == 'overloaded_error'
        )
 
            
    except Exception as e:
        return f"Claude API 錯誤: {str(e)}", "錯誤"
    
    # 如果所有重試都失敗
    return "所有重試都失敗，Claude API 目前可能不可用", "錯誤"

def generate_deepseek_response(messages, model_id):
    """直接使用 DeepSeek API 生成回答，支援串流模式"""
    deepseek_api_key = get_env_variable("DEEPSEEK_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not deepseek_api_key:
        return "DeepSeek API 未正確配置，請確認您已設定 API 密鑰。", "配置錯誤"
    
    try:
        # 構建請求頭和請求體
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # 處理系統提示
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        system_content = system_message["content"] if system_message else ""
        
        # 構建訊息格式
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # 系統消息將單獨處理
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 依據模型設定適當的 max_tokens
        max_tokens_mapping = {
            "deepseek-chat": 8000
        }
        # 獲取該模型的 max_tokens，如果沒找到則默認為 10000
        max_tokens = max_tokens_mapping.get(model_id, 10000)
        
        # 構建請求體
        payload = {
            "model": model_id,
            "messages": formatted_messages,
            "temperature": llm_temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if system_content:
            payload["system"] = system_content
        
        # 發送請求到 DeepSeek API
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        
        # 檢查響應狀態
        response.raise_for_status()
        
        return response, "串流"
            
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_message = e.response.text
        return f"DeepSeek API 錯誤 ({status_code}): {error_message}", "錯誤"
    except requests.exceptions.RequestException as e:
        return f"請求 DeepSeek API 時發生錯誤: {str(e)}", "錯誤"
    except Exception as e:
        return f"生成回答時發生錯誤: {str(e)}", "錯誤" 