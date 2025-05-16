import requests
import openai
from anthropic import Anthropic
from anthropic._exceptions import APIStatusError
from utils.config import get_env_variable
from google import genai
from google.genai import types

def generate_openai_response(messages, model):
    """使用 OpenAI API 生成回答，支援串流和非串流模式"""
    
    openai_api_key = get_env_variable("OPENAI_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not openai_api_key:
        return "OpenAI API 金鑰未設定", "配置錯誤"
    
    try:
        # 創建 OpenAI 客戶端
        client = openai.OpenAI(api_key=openai_api_key)
        
        # 依據模型設定適當的 max_tokens
        max_tokens_mapping = {
            "gpt-4o": 16000,
            "gpt-4.1": 32768,
        }
        max_tokens = max_tokens_mapping.get(model, 16000)
        
        # 處理系統提示
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        system_content = system_message["content"] if system_message else ""
        
        # 構建訊息格式
        formatted_messages = []
        
        # 如果有系統消息，將其添加為第一條消息
        if system_content:
            formatted_messages.append({
                "role": "system",
                "content": system_content
            })
            
        # 添加其他非系統消息
        for msg in messages:
            if msg["role"] == "system":
                continue  # 已經處理過系統消息
            
            # 確保消息內容不是 None
            if msg["content"] is None:
                msg_copy = {
                    "role": msg["role"],
                    "content": ""
                }
                formatted_messages.append(msg_copy)
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 使用新版 API 創建聊天完成
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )
        return response, "串流"
    except Exception as e:
        return f"OpenAI API 錯誤: {str(e)}", "錯誤"

def generate_claude_response(messages, model):
    """使用 Anthropic Claude API 生成回答，支援串流和非串流模式"""
    
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

def generate_gemini_response(messages, model):
    """使用 Google Gemini API 生成回答，支援串流和非串流模式"""
    gemini_api_key = get_env_variable("GEMINI_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not gemini_api_key:
        return "Gemini API 金鑰未設定", "配置錯誤"
    allowed_models = ["gemini-2.0-flash"]
    
    # 設定重試機制
    max_retries = 3
    retry_count = 0
    
    # 嘗試使用非串流模式 (如果串流模式失敗)
    use_non_streaming = False
    
    while retry_count < max_retries:
        try:
            client = genai.Client(api_key=gemini_api_key)
            
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            system_content = system_message["content"] if system_message else None
            
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                
                role = msg["role"]
                if role == "assistant":
                    role = "model"
                    
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            # 使用串流模式
            response = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_content,
                    max_output_tokens=8000,  # 增加最大輸出令牌數
                    temperature=llm_temperature
                )
            )
            
            # 測試是否真的能夠迭代（提前檢測，避免後面出錯）
            try:
                # 只獲取一個元素確認可以迭代，但不消耗整個生成器
                iterator = iter(response)
                first_chunk = next(iterator, None)
                
                # 如果成功獲取第一個元素，創建一個新的生成器包含這個元素和原始迭代器的剩餘部分
                if first_chunk is not None:
                    def combined_generator():
                        yield first_chunk  # 先返回第一個已經讀取的元素
                        for chunk in iterator:  # 然後返回剩餘的元素
                            yield chunk
                    
                    return combined_generator(), "串流"
                else:
                    # 可迭代但為空
                    return "無法產生內容，請重新嘗試", "錯誤"
            except (TypeError, AttributeError):
                # 如果不可迭代，返回錯誤
                return "串流生成失敗，請重新嘗試", "錯誤"
            
                
        except Exception as e:
            # 檢查錯誤類型
            error_str = str(e).lower()
            
            # 處理提示被封鎖的錯誤
            if "blocked" in error_str or "safety" in error_str or "harm" in error_str:
                return f"Gemini API 錯誤: 提示被封鎖 - {str(e)}", "錯誤"
            
            # 處理生成中止的錯誤 
            elif "stop" in error_str or "candidate" in error_str or "cancel" in error_str:
                return f"Gemini API 錯誤: 回應生成中止 - {str(e)}", "錯誤"
            
            # 處理連接錯誤
            elif isinstance(e, requests.exceptions.ConnectionError):
                # 連接錯誤，可以重試
                retry_count += 1
                if retry_count >= max_retries:
                    return f"Gemini API 連接錯誤，已重試 {max_retries} 次: {str(e)}", "錯誤"
            
            # 處理超時錯誤
            elif isinstance(e, requests.exceptions.Timeout):
                # 超時錯誤，可以重試
                retry_count += 1
                if retry_count >= max_retries:
                    return f"Gemini API 超時，已重試 {max_retries} 次: {str(e)}", "錯誤"
            
            # 處理其他錯誤
            elif "iterations" in error_str or "object is not iterable" in error_str:
                # 改用非串流模式
                use_non_streaming = True
                continue
            elif any(keyword in error_str for keyword in ["overloaded", "capacity", "rate limit", "too many requests"]):
                # 可能是過載錯誤，可以重試
                retry_count += 1
                if retry_count >= max_retries:
                    return f"Gemini API 過載，已重試 {max_retries} 次: {str(e)}", "錯誤"
            else:
                # 其他未知錯誤，嘗試非串流模式
                if not use_non_streaming:
                    use_non_streaming = True
                    continue
                else:
                    # 非串流模式也失敗
                    return f"Gemini API 錯誤: {str(e)}", "錯誤"
    
    # 如果所有重試都失敗
    return "所有重試都失敗，Gemini API 目前可能不可用", "錯誤"

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
        
        # 如果有系統消息，將其添加為第一條消息
        if system_content:
            formatted_messages.append({
                "role": "system",
                "content": system_content
            })
            
        # 添加其他非系統消息
        for msg in messages:
            if msg["role"] == "system":
                continue  # 已經處理過系統消息
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 依據模型設定適當的 max_tokens
        max_tokens_mapping = {
            "deepseek-chat": 8000
        }
        max_tokens = max_tokens_mapping.get(model_id, 8000)
        
        # 構建請求體
        payload = {
            "model": model_id,
            "messages": formatted_messages,
            "temperature": llm_temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
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