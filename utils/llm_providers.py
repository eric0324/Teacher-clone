import json
import requests
import botocore.exceptions
import streamlit as st
from utils.config import get_env_variable

def generate_openai_response(messages, model, streaming=False):
    """使用 OpenAI API 生成回答，支援串流和非串流模式"""
    from openai import OpenAI
    
    openai_api_key = get_env_variable("OPENAI_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not openai_api_key:
        return "OpenAI API 金鑰未設定", "配置錯誤"
    
    try:
        client = OpenAI(api_key=openai_api_key)
        
        # 串流模式
        if streaming:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=llm_temperature,
                stream=True
            )
            return response, "串流"
        # 非串流模式
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=llm_temperature
            )
            return response.choices[0].message.content.strip(), "成功"
    except Exception as e:
        return f"OpenAI API 錯誤: {str(e)}", "錯誤"

def generate_bedrock_response(messages, model_id, streaming=False):
    """使用 Amazon Bedrock 生成回答"""
    import boto3
    
    aws_region = get_env_variable("AWS_REGION", "us-east-1")
    aws_access_key = get_env_variable("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = get_env_variable("AWS_SECRET_ACCESS_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not aws_access_key or not aws_secret_key:
        return "Amazon Bedrock 未正確配置，請確認您已設定 AWS 憑證。", "配置錯誤"
    
    try:
        # 初始化 Bedrock 客戶端
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # 確認是 Anthropic Claude 模型還是其他模型
        if "anthropic.claude" in model_id.lower():
            # 處理 Claude 格式的提示
            prompt = {"anthropic_version": "bedrock-2023-05-31"}
            
            # 處理系統提示
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            if system_message:
                prompt["system"] = system_message["content"]
                # 從消息列表中移除系統消息
                messages = [msg for msg in messages if msg["role"] != "system"]
            
            # 處理對話
            prompt["messages"] = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                prompt["messages"].append({"role": role, "content": msg["content"]})
            
            # 為 Claude 3.5 模型添加必要的 max_tokens 參數
            if "claude-3-5" in model_id.lower():
                prompt["max_tokens"] = 2048
            
            # 添加串流參數
            if streaming:
                try:
                    response = bedrock_runtime.invoke_model_with_response_stream(
                        modelId=model_id,
                        body=json.dumps(prompt)
                    )
                    # 確保返回的是流式響應對象，而不是字符串錯誤信息
                    if isinstance(response, dict) and 'body' in response:
                        return response, "串流"
                    else:
                        # 如果響應不是預期的流式對象，返回錯誤信息
                        error_msg = "無法獲取正確的串流回應格式"
                        if isinstance(response, str):
                            error_msg = response
                        return error_msg, "錯誤"
                except Exception as e:
                    # 捕獲並處理流式請求中的錯誤
                    return f"串流請求錯誤: {str(e)}", "錯誤"
            else:
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(prompt)
                )
                response_body = json.loads(response.get("body").read())
                return response_body.get("content")[0].get("text"), "成功"
        
        # 處理 Meta Llama 模型
        elif "meta.llama" in model_id.lower():
            # Llama 使用不同的格式
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            system_content = system_message["content"] if system_message else ""
            
            # 格式化對話
            prompt = ""
            if system_content:
                prompt += f"<system>\n{system_content}\n</system>\n\n"
            
            for msg in messages:
                if msg["role"] == "system":
                    continue
                if msg["role"] == "user":
                    prompt += f"<human>\n{msg['content']}\n</human>\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"<assistant>\n{msg['content']}\n</assistant>\n\n"
            
            # 添加最後的助手角色標籤
            prompt += "<assistant>\n"
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "prompt": prompt,
                    "temperature": llm_temperature,
                    "max_gen_len": 2048
                })
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("generation"), "成功"
        
        # 處理 Mistral 模型
        elif "mistral." in model_id.lower():
            # Mistral 使用不同的格式
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"<s>[INST] {msg['content']} [/INST]</s>\n"
                elif msg["role"] == "user":
                    prompt += f"<s>[INST] {msg['content']} [/INST]</s>\n"
                elif msg["role"] == "assistant":
                    prompt += f"<s>{msg['content']}</s>\n"
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "prompt": prompt,
                    "temperature": llm_temperature,
                    "max_tokens": 2048
                })
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("outputs")[0].get("text"), "成功"
        
        # 處理 DeepSeek-R1 模型
        elif "deepseek.r1" in model_id.lower():
            # 處理系統提示
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            
            # 格式化用戶查詢
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            user_content = user_messages[-1]["content"] if user_messages else ""
            
            # 使用 DeepSeek-R1 的特定格式封裝提示詞，確保沒有多餘空格
            prompt_text = f"<｜begin_of_sentence｜> {user_content} <｜Assistant｜><think>\n"
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "prompt": prompt_text,
                    "max_tokens": 2048,
                    "temperature": llm_temperature,
                    "top_p": 0.9
                })
            )
            response_body = json.loads(response.get("body").read())
            
            if "choices" in response_body and len(response_body["choices"]) > 0:
                return response_body["choices"][0]["text"], "成功"
            else:
                return "無法從 DeepSeek 模型獲取回應", "回應解析錯誤"
        
        # 處理 Amazon Nova Pro 模型
        elif "amazon.nova-pro" in model_id.lower():
            # 組合所有消息
            complete_prompt = ""
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            
            # 如果有系統提示，先加到提示詞
            if system_message:
                complete_prompt += f"{system_message['content']}\n\n"
            
            # 按照對話順序添加用戶和助手的消息
            for msg in messages:
                if msg["role"] == "system":
                    continue
                if msg["role"] == "user":
                    complete_prompt += f"Human: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    complete_prompt += f"Assistant: {msg['content']}\n"
            
            # 添加最後的助手標記
            complete_prompt += "Assistant: "
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "inputText": complete_prompt,
                    "textGenerationConfig": {
                        "temperature": llm_temperature,
                        "maxTokenCount": 2048,
                        "topP": 0.9,
                        "stopSequences": []
                    }
                })
            )
            response_body = json.loads(response.get("body").read())
            
            if "results" in response_body and len(response_body["results"]) > 0:
                return response_body["results"][0]["outputText"], "成功"
            else:
                return "無法從 Nova Pro 模型獲取回應", "回應解析錯誤"
        
        # 處理 Amazon Titan 模型
        elif "amazon.titan" in model_id.lower():
            # 先獲取系統消息
            system_message = next((msg for msg in messages if msg["role"] == "system"), None)
            system_content = system_message["content"] if system_message else ""
            
            # 組合所有消息
            combined_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue  # 系統消息將在下面處理
                if msg["role"] == "user":
                    combined_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    combined_messages.append({"role": "assistant", "content": msg["content"]})
            
            # 創建 Titan 模型的請求體
            body = {
                "inputText": system_content + "\n\n" + "\n".join([f"{m['role']}: {m['content']}" for m in combined_messages]) + "\nassistant:",
                "textGenerationConfig": {
                    "maxTokenCount": 1024,
                    "temperature": llm_temperature,
                    "topP": 0.9
                }
            }
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("results")[0].get("outputText"), "成功"
        
        else:
            return f"不支援的模型: {model_id}", "模型不支援"
            
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        if error_code == "AccessDeniedException":
            return f"存取被拒絕: {error_message}", "存取被拒絕"
        elif error_code == "ValidationException":
            return f"驗證錯誤: {error_message}", "驗證錯誤"
        elif error_code == "ModelTimeoutException":
            return "模型處理逾時，請嘗試較短的問題。", "處理逾時"
        else:
            return f"Bedrock 呼叫錯誤: {error_code} - {error_message}", f"錯誤: {error_code}"
    except Exception as e:
        return f"生成回答時發生錯誤: {str(e)}", "處理錯誤"

def generate_deepseek_response(messages, model_id, streaming=False):
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
        
        # 構建請求體
        payload = {
            "model": model_id,
            "messages": formatted_messages,
            "temperature": llm_temperature,
            "max_tokens": 2048,
            "stream": streaming
        }
        
        if system_content:
            payload["system"] = system_content
        
        # 發送請求到 DeepSeek API
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=streaming
        )
        
        # 檢查響應狀態
        response.raise_for_status()
        
        # 串流模式
        if streaming:
            return response, "串流"
        # 非串流模式
        else:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"], "成功"
            else:
                return "無法從 DeepSeek 獲取回應", "回應解析錯誤"
            
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_message = e.response.text
        return f"DeepSeek API 錯誤 ({status_code}): {error_message}", f"HTTP錯誤: {status_code}"
    except requests.exceptions.RequestException as e:
        return f"請求 DeepSeek API 時發生錯誤: {str(e)}", "請求錯誤"
    except Exception as e:
        return f"生成回答時發生錯誤: {str(e)}", "處理錯誤" 