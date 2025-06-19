import requests
import openai
from anthropic import Anthropic
from anthropic._exceptions import APIStatusError
from utils.config import get_env_variable
import datetime
import streamlit as st
import PyPDF2
import io

def process_pdf_for_upload(file_content, file_name, max_pages=50):
    """處理 PDF 檔案，如果超過頁數限制則進行裁切"""
    print(f"[DEBUG] 開始處理 PDF: {file_name}")
    print(f"[DEBUG] 原檔案大小: {len(file_content)} bytes")
    print(f"[DEBUG] 最大頁數限制: {max_pages}")
    
    try:
        # 創建 PDF 讀取器
        print(f"[DEBUG] 正在創建 PDF 讀取器...")
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        total_pages = len(pdf_reader.pages)
        print(f"[DEBUG] PDF 總頁數: {total_pages}")
        
        # 如果頁數不超過限制，直接返回原檔案
        if total_pages <= max_pages:
            print(f"[DEBUG] 頁數未超過限制，使用原檔案")
            return file_content, total_pages, False
        
        # 如果超過限制，裁切為前 max_pages 頁
        print(f"[DEBUG] 頁數超過限制，開始裁切為前 {max_pages} 頁...")
        pdf_writer = PyPDF2.PdfWriter()
        
        for page_num in range(max_pages):
            print(f"[DEBUG] 正在添加第 {page_num + 1} 頁...")
            pdf_writer.add_page(pdf_reader.pages[page_num])
        
        # 將裁切後的 PDF 輸出為 bytes
        print(f"[DEBUG] 正在生成裁切後的 PDF...")
        output_buffer = io.BytesIO()
        pdf_writer.write(output_buffer)
        trimmed_content = output_buffer.getvalue()
        output_buffer.close()
        
        print(f"[DEBUG] 裁切完成，新檔案大小: {len(trimmed_content)} bytes")
        return trimmed_content, total_pages, True
        
    except Exception as e:
        # 如果 PDF 處理失敗，返回原檔案
        print(f"[ERROR] PDF 處理失敗: {str(e)}")
        st.warning(f"PDF 處理時發生錯誤，將使用原檔案上傳: {str(e)}")
        return file_content, None, False

def upload_file_to_anthropic(file_content, file_name):
    """上傳檔案到 Anthropic Files API，如果是 PDF 且超過 100 頁會自動裁切"""
    print(f"[DEBUG] ===== 開始上傳檔案到 Anthropic Files API =====")
    print(f"[DEBUG] 檔案名稱: {file_name}")
    print(f"[DEBUG] 檔案大小: {len(file_content)} bytes")
    
    claude_api_key = get_env_variable("CLAUDE_API_KEY", "")
    
    if not claude_api_key:
        print(f"[ERROR] Claude API 金鑰未設定")
        return None, "Claude API 金鑰未設定", None
    
    print(f"[DEBUG] Claude API 金鑰已設定 (長度: {len(claude_api_key)})")
    
    try:
        # 如果是 PDF 檔案，先進行處理
        processed_content = file_content
        page_info = {}
        
        if file_name.lower().endswith('.pdf'):
            print(f"[DEBUG] 檔案是 PDF，開始處理...")
            processed_content, total_pages, was_trimmed = process_pdf_for_upload(file_content, file_name)
            page_info = {
                'total_pages': total_pages,
                'was_trimmed': was_trimmed,
                'final_pages': min(total_pages or 0, 50) if total_pages else None
            }
            print(f"[DEBUG] PDF 處理完成: {page_info}")
        else:
            print(f"[DEBUG] 檔案不是 PDF，跳過處理")
        
        # 使用 requests 直接呼叫 Files API
        headers = {
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14"
        }
        
        print(f"[DEBUG] 準備上傳請求...")
        print(f"[DEBUG] Headers: {dict(headers)}")  # 不顯示 API key 的完整內容
        print(f"[DEBUG] 最終檔案大小: {len(processed_content)} bytes")
        
        files = {
            "file": (file_name, processed_content, "application/pdf")
        }
        
        print(f"[DEBUG] 發送 POST 請求到 https://api.anthropic.com/v1/files")
        response = requests.post(
            "https://api.anthropic.com/v1/files",
            headers=headers,
            files=files
        )
        
        print(f"[DEBUG] 收到響應，狀態碼: {response.status_code}")
        print(f"[DEBUG] 響應 headers: {dict(response.headers)}")
        
        if response.status_code >= 400:
            print(f"[ERROR] HTTP 錯誤響應: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        print(f"[DEBUG] 響應內容: {result}")
        
        file_id = result.get("id")
        print(f"[DEBUG] 獲得檔案 ID: {file_id}")
        
        print(f"[DEBUG] ===== 檔案上傳成功 =====")
        return file_id, None, page_info
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP 錯誤: {e}")
        print(f"[ERROR] 狀態碼: {e.response.status_code}")
        print(f"[ERROR] 響應內容: {e.response.text}")
        error_message = f"檔案上傳失敗 (HTTP {e.response.status_code}): {e.response.text}"
        return None, error_message, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 請求錯誤: {e}")
        return None, f"請求錯誤: {str(e)}", None
    except Exception as e:
        print(f"[ERROR] 未預期的錯誤: {e}")
        print(f"[ERROR] 錯誤類型: {type(e).__name__}")
        import traceback
        print(f"[ERROR] 完整錯誤追蹤: {traceback.format_exc()}")
        return None, f"檔案上傳錯誤: {str(e)}", None

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
        
        # 檢查並截斷系統提示詞
        if system_content:
            max_system_chars = 50000  # 系統提示詞最大字符數限制
            if len(system_content) > max_system_chars:
                print(f"[WARNING] 系統提示詞過長 ({len(system_content)} 字符)，截斷到 {max_system_chars} 字符")
                system_content = system_content[:max_system_chars] + "\n[系統提示詞已截斷]"
        
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

def generate_claude_response(messages, model, file_id=None):
    """使用 Anthropic Claude API 生成回答，支援串流和非串流模式，並支援檔案"""
    
    claude_api_key = get_env_variable("CLAUDE_API_KEY", "")
    llm_temperature = float(get_env_variable("LLM_TEMPERATURE", "0.3"))
    
    if not claude_api_key:
        return "Claude API 金鑰未設定", "配置錯誤"
    
    max_tokens_mapping = {
        "claude-sonnet-4-20250514": 10000,
        "claude-3-5-sonnet-20241022": 8000,
    }
    
    max_tokens = max_tokens_mapping.get(model, 8000)
    
    try:
        # 創建Anthropic客戶端，使用適當的API版本設置
        default_headers = {
            "anthropic-version": "2023-06-01"
        }
        
        # 如果有檔案，添加 beta header
        if file_id:
            default_headers["anthropic-beta"] = "files-api-2025-04-14"
        
        client = Anthropic(
            api_key=claude_api_key,
            default_headers=default_headers
        )
        
        # 處理系統提示
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        system_content = system_message["content"] if system_message else None
        
        # 檢查並截斷系統提示詞
        if system_content:
            max_system_chars = 50000  # 系統提示詞最大字符數限制
            if len(system_content) > max_system_chars:
                print(f"[WARNING] 系統提示詞過長 ({len(system_content)} 字符)，截斷到 {max_system_chars} 字符")
                system_content = system_content[:max_system_chars] + "\n[系統提示詞已截斷]"
        
        # 從消息列表中移除系統消息
        filtered_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # 準備消息格式
        formatted_messages = []
        print(f"[DEBUG] ===== 準備 Claude API 訊息 =====")
        print(f"[DEBUG] 檔案 ID: {file_id}")
        print(f"[DEBUG] 過濾後的訊息數量: {len(filtered_messages)}")
        
        for i, msg in enumerate(filtered_messages):
            role = "user" if msg["role"] == "user" else "assistant"
            
            # 如果是最後一個用戶訊息且有檔案，則添加檔案內容
            is_last_user_msg = (role == "user" and msg == filtered_messages[-1])
            
            if (is_last_user_msg and file_id):
                print(f"[DEBUG] 在最後一個用戶訊息中添加檔案 (訊息 {i+1})")
                
                content = [
                    {
                        "type": "text",
                        "text": msg["content"]
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id
                        }
                    }
                ]
                formatted_messages.append({"role": role, "content": content})
                print(f"[DEBUG] 已添加包含檔案的訊息結構")
            else:
                formatted_messages.append({"role": role, "content": msg["content"]})
                print(f"[DEBUG] 添加普通訊息 {i+1} (角色: {role})")
        
        print(f"[DEBUG] 最終格式化的訊息數量: {len(formatted_messages)}")
        
        # 打印最終請求參數和內容長度分析
        print(f"[DEBUG] ===== 內容長度分析 =====")
        if system_content:
            system_tokens = len(system_content) // 2.5
            print(f"[DEBUG] 系統提示詞: {len(system_content)} 字符 (~{system_tokens:.0f} tokens 估算)")
        else:
            system_tokens = 0
            print(f"[DEBUG] 系統提示詞: 無")
        
        total_message_chars = 0
        total_message_tokens = 0
        
        for i, msg in enumerate(formatted_messages):
            if isinstance(msg.get('content'), str):
                char_count = len(msg['content'])
                token_count = char_count // 2.5
                total_message_chars += char_count
                total_message_tokens += token_count
                print(f"[DEBUG] 訊息 {i+1} ({msg['role']}): {char_count} 字符 (~{token_count:.0f} tokens 估算)")
                
                # 如果訊息特別長，顯示前 100 字符以幫助調試
                if char_count > 50000:
                    print(f"[WARNING] 訊息 {i+1} 內容很長！前 100 字符: {msg['content'][:100]}...")
                    
            elif isinstance(msg.get('content'), list):
                total_blocks_chars = 0
                total_blocks_tokens = 0
                for j, block in enumerate(msg['content']):
                    if block.get('type') == 'text':
                        block_chars = len(block.get('text', ''))
                        block_tokens = block_chars // 2.5
                        total_blocks_chars += block_chars
                        total_blocks_tokens += block_tokens
                        print(f"[DEBUG] 訊息 {i+1} 文字塊 {j+1}: {block_chars} 字符 (~{block_tokens:.0f} tokens 估算)")
                        
                        # 如果文字塊特別長，顯示前 100 字符
                        if block_chars > 50000:
                            print(f"[WARNING] 訊息 {i+1} 文字塊 {j+1} 內容很長！前 100 字符: {block.get('text', '')[:100]}...")
                            
                    elif block.get('type') == 'document':
                        print(f"[DEBUG] 訊息 {i+1} 檔案塊 {j+1}: 檔案 ID {block.get('source', {}).get('file_id', '未知')}")
                        # 檔案的 token 消耗通常很大，特別是 PDF
                        # 一個 50 頁的 PDF 可能消耗 100k-150k tokens
                        estimated_file_tokens = 120000  # 更保守的估算
                        total_blocks_tokens += estimated_file_tokens
                        print(f"[WARNING] 檔案預估消耗 ~{estimated_file_tokens} tokens")
                        
                total_message_chars += total_blocks_chars
                total_message_tokens += total_blocks_tokens
                print(f"[DEBUG] 訊息 {i+1} ({msg['role']}) 總計: {total_blocks_chars} 字符 (~{total_blocks_tokens:.0f} tokens 估算)")
        
        total_chars = (len(system_content) if system_content else 0) + total_message_chars
        estimated_tokens = system_tokens + total_message_tokens
        
        print(f"[DEBUG] ===== 總計估算 =====")
        print(f"[DEBUG] 系統提示詞 tokens: ~{system_tokens:.0f}")
        print(f"[DEBUG] 訊息內容 tokens: ~{total_message_tokens:.0f}")
        print(f"[DEBUG] 總字符數: {total_chars}")
        print(f"[DEBUG] 估算 tokens: ~{estimated_tokens:.0f}")
        print(f"[DEBUG] Claude 限制: 200,000 tokens")
        print(f"[DEBUG] 實際 API 回報: 218,711 tokens (如果有的話)")
        
        # 更嚴格的限制：預留更多 buffer
        if estimated_tokens > 150000:  # 降低預警閾值
            print(f"[WARNING] 估算 tokens 接近或超過限制！可能會被拒絕")
            
            # 提供建議
            if total_message_chars > 100000:  # 降低閾值
                print(f"[WARNING] 訊息內容很長，可能需要：")
                print(f"[WARNING] 1. 減少聊天歷史長度 (memory_length)")
                print(f"[WARNING] 2. 檢查是否有超大的知識庫搜索結果")
                print(f"[WARNING] 3. 檢查上傳的檔案是否過大")
            
            if system_content and len(system_content) > 30000:  # 降低閾值
                print(f"[WARNING] 系統提示詞很長，考慮縮短")
                
            # 如果預估超過 180k tokens，直接返回錯誤而不嘗試呼叫 API
            if estimated_tokens > 180000:  # 更嚴格的限制
                print(f"[ERROR] 預估 tokens ({estimated_tokens:.0f}) 超過安全限制 (180,000)，停止呼叫 API")
                return f"內容太長 (預估 {estimated_tokens:.0f} tokens)，超過 Claude 安全限制。請縮短內容或減少聊天歷史。", "錯誤"
        
        # 如果有檔案，打印檔案相關信息
        if file_id:
            print(f"[DEBUG] 包含檔案 ID: {file_id}")
            # 檢查最後一個訊息是否包含檔案
            last_msg = formatted_messages[-1] if formatted_messages else None
            if last_msg and isinstance(last_msg.get('content'), list):
                print(f"[DEBUG] 最後一個訊息包含 {len(last_msg['content'])} 個內容塊")
                for i, content_block in enumerate(last_msg['content']):
                    print(f"[DEBUG] 內容塊 {i+1}: {content_block.get('type', '未知類型')}")
            
        response = client.messages.create(
            model=model,
            messages=formatted_messages,
            system=system_content,
            temperature=llm_temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        print(f"[DEBUG] Claude API 呼叫成功，獲得響應")
        return response, "串流"
            
    except APIStatusError as e:
        print(f"[ERROR] ===== Claude API 狀態錯誤 =====")
        print(f"[ERROR] 狀態碼: {e.status_code}")
        print(f"[ERROR] 錯誤類型: {type(e).__name__}")
        print(f"[ERROR] 錯誤訊息: {str(e)}")
        
        # 檢查錯誤詳情
        error_json = e.body if hasattr(e, 'body') else {}
        print(f"[ERROR] 錯誤詳情: {error_json}")
        
        if isinstance(error_json, dict):
            error_type = error_json.get('error', {}).get('type', '未知')
            error_message = error_json.get('error', {}).get('message', '無詳細訊息')
            print(f"[ERROR] 錯誤類型: {error_type}")
            print(f"[ERROR] 錯誤訊息: {error_message}")
            
            # 檢查是否為檔案相關錯誤
            if file_id and ('file' in error_message.lower() or 'document' in error_message.lower()):
                print(f"[ERROR] 可能是檔案相關錯誤，檔案 ID: {file_id}")
        
        return f"Claude API 錯誤 (狀態碼 {e.status_code}): {str(e)}", "錯誤"
            
    except Exception as e:
        print(f"[ERROR] ===== Claude API 未預期錯誤 =====")
        print(f"[ERROR] 錯誤類型: {type(e).__name__}")
        print(f"[ERROR] 錯誤訊息: {str(e)}")
        import traceback
        print(f"[ERROR] 完整錯誤追蹤: {traceback.format_exc()}")
        return f"Claude API 錯誤: {str(e)}", "錯誤"

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

def save_question_to_supabase(question, answer, prompt_name, knowledge_table):
    """將用戶提問和回答記錄儲存到 Supabase"""
    try:
        supabase = st.session_state.supabase
        if supabase:
            question_data = {
                "question": question,
                "answer": answer,  # 新增回答
                "prompt_name": prompt_name,
                "knowledge_table": knowledge_table,
                "llm_provider": st.session_state.llm_provider,
                "created_at": datetime.datetime.now().isoformat()
            }
            result = supabase.table("question_logs").insert(question_data).execute()
            if result and hasattr(result, 'data') and result.data:
                return True
            else:
                print("插入問題記錄失敗")
                return False
        return False
    except Exception as e:
        print(f"儲存問題到 Supabase 時出錯: {str(e)}")
        return False