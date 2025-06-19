import streamlit as st
import json
import datetime
from streamlit_mermaid import st_mermaid

# 導入自定義模塊
from utils.config import load_config, get_env_variable, load_system_prompt
from utils.ui import setup_ui, display_chat_history, render_mermaid_diagrams
from utils.auth import check_password
from utils.knowledge import search_knowledge, extract_core_question_with_llm
from utils.llm_providers import (
    generate_openai_response, 
    generate_claude_response, 
    generate_deepseek_response,
    save_question_to_supabase,
    upload_file_to_anthropic
)

# 設置頁面配置和標題 - 移除側邊欄配置
st.set_page_config(page_title="數位分身系統", layout="wide")

# 只隱藏側邊欄和收起箭頭，不影響主要內容
st.markdown("""
<style>
    /* 隱藏側邊欄 */
    [data-testid="stSidebar"] {
        display: none !important;
        width: 0px !important;
    }
    
    /* 隱藏側邊欄控制按鈕 */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        width: 0px !important;
    }
    
    /* 隱藏箭頭按鈕 */
    section[data-testid="stSidebarContent"],
    div.st-emotion-cache-gsulwm,
    .st-emotion-cache-16j9m0,
    button[kind="headerNoPadding"] {
        display: none !important;
    }
    
    /* 只隱藏側邊欄的箭頭圖標，不影響其他SVG */
    [data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg {
        display: none !important;
    }
    
    /* 將 PDF 上傳按鈕固定在底部 */
    [data-testid="stFileUploader"] {
        position: fixed !important;
        bottom: 10px !important;
        left: 50px !important;
        right: 20px !important;
        background: white !important;
        padding: 10px 15px !important;
        z-index: 999 !important;
        width: auto !important;
        max-width: 600px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* 調整聊天輸入框的位置，為 PDF 上傳按鈕騰出空間 */
    [data-testid="stBottom"] {
        bottom: 100px !important;
        z-index: 1000 !important;
    }
    
    /* 確保主要內容區域有足夠的底部間距 */
    .stMainBlockContainer {
        padding-bottom: 120px !important;
    }
    
    /* PDF 上傳按鈕內部樣式調整 */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        min-height: 60px !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] label {
        margin-bottom: 8px !important;
    }
</style>


""", unsafe_allow_html=True)

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

if "prompt_name" not in st.session_state:
    st.session_state.prompt_name = "mj"

if "knowledge_table" not in st.session_state:
    st.session_state.knowledge_table = get_env_variable("KNOWLEDGE_TABLE", "knowledge_base")  

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "claude"  # 預設使用 Claude

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True  # 預設啟用串流回應

if "memory_length" not in st.session_state:
    st.session_state.memory_length = 3  # 降低預設值從 5 到 3
    
# 載入配置
config = load_config()

# 將配置添加到會話狀態中
st.session_state.supabase = config.get("supabase")

# 再載入系統提示詞
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = load_system_prompt(st.session_state.prompt_name)

# 聊天界面部分
st.subheader("數位分身聊天")

# 添加緊急控制按鈕
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🗑️ 清除聊天歷史 (解決 Token 超限)", key="clear_chat_history"):
        st.session_state.messages = []
        st.success("聊天歷史已清除！這應該能解決 token 超限問題。")
        st.rerun()

# 添加 Token 診斷功能
with st.expander("🔍 Token 使用診斷"):
    if st.button("分析當前 Token 使用情況"):
        # 計算各部分的 token 使用
        system_prompt = st.session_state.custom_prompt or ""
        system_tokens = len(system_prompt) / 2.5
        
        chat_history = st.session_state.messages[-st.session_state.memory_length*2:] if st.session_state.messages else []
        history_chars = sum(len(msg.get('content', '')) for msg in chat_history)
        history_tokens = history_chars / 2.5
        
        uploaded_file_id = st.session_state.get('uploaded_file_id', None)
        file_tokens = 120000 if uploaded_file_id else 0
        
        total_estimated = system_tokens + history_tokens + file_tokens
        
        st.write("📊 **Token 使用分析：**")
        st.write(f"- 系統提示詞：~{system_tokens:.0f} tokens ({len(system_prompt)} 字符)")
        st.write(f"- 聊天歷史：~{history_tokens:.0f} tokens ({history_chars} 字符)")
        st.write(f"- 上傳檔案：~{file_tokens} tokens")
        st.write(f"- **總計估算：~{total_estimated:.0f} tokens**")
        
        if total_estimated > 180000:
            st.error(f"⚠️ 預估 tokens ({total_estimated:.0f}) 超過安全限制！")
            st.write("**建議解決方案：**")
            if file_tokens > 0:
                st.write("1. 清除上傳的檔案")
            if history_tokens > 50000:
                st.write("2. 清除聊天歷史")
            if system_tokens > 20000:
                st.write("3. 縮短系統提示詞")
        elif total_estimated > 150000:
            st.warning(f"⚠️ 預估 tokens ({total_estimated:.0f}) 接近限制")
        else:
            st.success(f"✅ Token 使用正常 ({total_estimated:.0f}/200000)")

# 設置UI樣式
setup_ui()

# 初始化狀態歷史（如果不存在）
if "status_history" not in st.session_state:
    st.session_state.status_history = []

# 顯示聊天歷史
display_chat_history()

# 儲存功能已移至 utils.llm_providers 模組

# 搜索知識庫
def search_knowledge_base(query, update_status):
    """從知識庫中搜索與查詢相關的信息"""
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
            
            print(f"[DEBUG] ===== 知識庫搜索結果 =====")
            print(f"[DEBUG] 找到 {len(knowledge_points)} 個知識點")
            
            # 準備知識點信息
            knowledge_info = []
            total_knowledge_chars = 0
            max_knowledge_chars = 200000  # 設定最大知識庫內容字符數限制
            
            for i, item in enumerate(knowledge_points):
                match_info = f"({item.get('match_type', '未知匹配類型')}, 相似度: {item.get('similarity', 0):.2f})"
                knowledge_text = f"概念: {item['concept']} {match_info}\n解釋: {item['explanation']}"
                
                # 檢查添加這個知識點是否會超過限制
                if total_knowledge_chars + len(knowledge_text) > max_knowledge_chars:
                    print(f"[WARNING] 知識庫內容達到限制，截斷在第 {i} 個知識點")
                    break
                
                knowledge_info.append(knowledge_text)
                
                chars_count = len(knowledge_text)
                total_knowledge_chars += chars_count
                print(f"[DEBUG] 知識點 {i+1}: {chars_count} 字符")
            
            # 將知識點整合為上下文
            context = "\n\n".join(knowledge_info)
            print(f"[DEBUG] 知識庫上下文總長度: {len(context)} 字符 (~{len(context)//2.5:.0f} tokens 估算)")
            
            if len(context) > 300000:  # 降低警告閾值
                print(f"[WARNING] 知識庫內容很大，可能會導致 token 超限")
            
            return context
            
        except Exception as e:
            update_status(f"搜索知識庫時出錯: {str(e)}")
            return None
    except Exception as e:
        update_status(f"處理問題時發生錯誤: {str(e)}")
        return None

# 生成回答
def generate_answer(query, context, update_status):
    """使用搜索到的知識生成回答"""
    update_status("找到了相關資訊，正在生成回答...")
    try:
        # 獲取系統提示詞
        system_content = st.session_state.custom_prompt
        
        # 獲取最近 memory_length 條對話歷史
        # 檔案模式下為了避免 token 超限，暫時不使用聊天歷史
        print(f"[WARNING] 檔案模式：為避免 token 超限，跳過聊天歷史")
        chat_history = []  # 完全清空聊天歷史
        
        print(f"[DEBUG] 檔案模式聊天歷史總字符數: 0 (~0 tokens 估算)")
        
        # 構建消息
        messages = [{"role": "system", "content": system_content}]

        # 添加當前問題和上下文
        augmented_prompt = f"""
        <message>
            {query}
        </message>

        <retrieved_knowledge>
            {context}
        </retrieved_knowledge>
        """
        
        messages.append({"role": "user", "content": augmented_prompt})
        
        # RAG 模式不使用檔案，避免 token 消耗過大
        print(f"[DEBUG] ===== RAG 模式，不使用檔案 =====")
        
        # 生成回答（RAG 模式不傳送檔案 ID）
        return generate_response(messages, file_id=None)
    except Exception as e:
        error_message = f"生成回答時發生錯誤: {str(e)}"
        update_status(error_message)
        # 返回錯誤消息，而不是 None
        return [error_message, "錯誤"]

def generate_response(messages, file_id=None):
    """根據選定的LLM供應商生成回答"""
    llm_provider = st.session_state.llm_provider
    use_streaming = st.session_state.use_streaming
    
    if llm_provider == "openai":
        llm_model = get_env_variable("LLM_MODEL", "gpt-4o")
        response, _ = generate_openai_response(
            messages=messages,
            model=llm_model
        )
        return response, "串流"
    elif llm_provider == "claude":
        claude_model = get_env_variable("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        response, _ = generate_claude_response(
            messages=messages,
            model=claude_model,
            file_id=file_id
        )
        return response, "串流"
    elif llm_provider == "deepseek":
        deepseek_model = get_env_variable("DEEPSEEK_MODEL", "deepseek-chat")
        response, _ = generate_deepseek_response(
            messages=messages,
            model_id=deepseek_model
        )
        return response, "串流"
    else:
        # 預設使用 Claude
        claude_model = get_env_variable("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        response, _ = generate_claude_response(
            messages=messages,
            model=claude_model,
            file_id=file_id
        )
        return response, "串流"


def display_streaming_response(stream_response, message_placeholder):
    """顯示串流回應"""
    full_response = ""
    llm_provider = st.session_state.llm_provider
    
    # 檢查是否是字符串(錯誤信息或非串流回應)
    if isinstance(stream_response, str):
        message_placeholder.markdown(stream_response)
        return stream_response
    
    if llm_provider == "openai":
        # 處理 OpenAI 串流
        for chunk in stream_response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # 檢查是否有完整的 mermaid 圖表代碼塊，如果有就渲染
                        processed_content = render_mermaid_diagrams(full_response)
                        if "<MERMAID_CHART>" in processed_content:
                            # 重新渲染整個內容，包括圖表
                            message_placeholder.empty()
                            with message_placeholder.container():
                                parts = processed_content.split("<MERMAID_CHART>")
                                for i, part in enumerate(parts):
                                    if i == 0:
                                        # 第一部分是純文本
                                        if part:
                                            st.markdown(part, unsafe_allow_html=True)
                                    else:
                                        # 查找圖表代碼和後續文本
                                        chart_end = part.find("</MERMAID_CHART>")
                                        if chart_end != -1:
                                            chart_code = part[:chart_end]
                                            remaining_text = part[chart_end + 16:]  # 16是</MERMAID_CHART>的長度
                                            
                                            # 渲染圖表
                                            try:
                                                st_mermaid(chart_code, height=350)
                                            except Exception as e:
                                                st.error(f"圖表渲染失敗: {str(e)}")
                                                st.code(chart_code, language="mermaid")
                                            
                                            # 渲染剩餘文本
                                            if remaining_text:
                                                st.markdown(remaining_text, unsafe_allow_html=True)
                        else:
                            # 普通文本，直接更新
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
            # 處理不同類型的事件和結構
            if hasattr(chunk, 'type'):
                # 處理content_block_delta事件
                if chunk.type == 'content_block_delta' and hasattr(chunk, 'delta'):
                    if hasattr(chunk.delta, 'type') and chunk.delta.type == 'text_delta':
                        if hasattr(chunk.delta, 'text'):
                            content = chunk.delta.text
                            if content:
                                full_response += content
                                # 檢查是否有完整的 mermaid 圖表代碼塊，如果有就渲染
                                processed_content = render_mermaid_diagrams(full_response)
                                if "<MERMAID_CHART>" in processed_content:
                                    # 重新渲染整個內容，包括圖表
                                    message_placeholder.empty()
                                    with message_placeholder.container():
                                        parts = processed_content.split("<MERMAID_CHART>")
                                        for i, part in enumerate(parts):
                                            if i == 0:
                                                # 第一部分是純文本
                                                if part:
                                                    st.markdown(part, unsafe_allow_html=True)
                                            else:
                                                # 查找圖表代碼和後續文本
                                                chart_end = part.find("</MERMAID_CHART>")
                                                if chart_end != -1:
                                                    chart_code = part[:chart_end]
                                                    remaining_text = part[chart_end + 16:]  # 16是</MERMAID_CHART>的長度
                                                    
                                                    # 渲染圖表
                                                    try:
                                                        st_mermaid(chart_code, height=350)
                                                    except Exception as e:
                                                        st.error(f"圖表渲染失敗: {str(e)}")
                                                        st.code(chart_code, language="mermaid")
                                                    
                                                    # 渲染剩餘文本
                                                    if remaining_text:
                                                        st.markdown(remaining_text, unsafe_allow_html=True)
                                else:
                                    # 普通文本，直接更新
                                    message_placeholder.markdown(full_response)
                # Claude 2.x 舊版API
                elif chunk.type == 'completion' and hasattr(chunk, 'completion'):
                    content = chunk.completion
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response)
    
    elif llm_provider == "deepseek":
        # 處理 DeepSeek 串流
        try:
            for line in stream_response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        json_data = json.loads(line[6:])
                        if 'choices' in json_data and json_data['choices'] and 'delta' in json_data['choices'][0]:
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                full_response += content
                                # 檢查是否有完整的 mermaid 圖表代碼塊，如果有就渲染
                                processed_content = render_mermaid_diagrams(full_response)
                                if "<MERMAID_CHART>" in processed_content:
                                    # 重新渲染整個內容，包括圖表
                                    message_placeholder.empty()
                                    with message_placeholder.container():
                                        parts = processed_content.split("<MERMAID_CHART>")
                                        for i, part in enumerate(parts):
                                            if i == 0:
                                                # 第一部分是純文本
                                                if part:
                                                    st.markdown(part, unsafe_allow_html=True)
                                            else:
                                                # 查找圖表代碼和後續文本
                                                chart_end = part.find("</MERMAID_CHART>")
                                                if chart_end != -1:
                                                    chart_code = part[:chart_end]
                                                    remaining_text = part[chart_end + 16:]  # 16是</MERMAID_CHART>的長度
                                                    
                                                    # 渲染圖表
                                                    try:
                                                        st_mermaid(chart_code, height=350)
                                                    except Exception as e:
                                                        st.error(f"圖表渲染失敗: {str(e)}")
                                                        st.code(chart_code, language="mermaid")
                                                    
                                                    # 渲染剩餘文本
                                                    if remaining_text:
                                                        st.markdown(remaining_text, unsafe_allow_html=True)
                                else:
                                    # 普通文本，直接更新
                                    message_placeholder.markdown(full_response)
        except AttributeError:
            # 如果沒有iter_lines方法，可能是錯誤訊息
            if hasattr(stream_response, 'text'):
                message_placeholder.markdown(stream_response.text)
                return stream_response.text
            message_placeholder.markdown(str(stream_response))
            return str(stream_response)
    
    elif llm_provider == "gemini":
        # 處理 Gemini 串流
        try:
            # 檢查是否為字符串或錯誤消息
            if isinstance(stream_response, str):
                processed_content = render_mermaid_diagrams(stream_response)
                if "<MERMAID_CHART>" in processed_content:
                    # 如果包含圖表，需要特殊處理
                    message_placeholder.empty()
                    with message_placeholder.container():
                        parts = processed_content.split("<MERMAID_CHART>")
                        for i, part in enumerate(parts):
                            if i == 0:
                                if part:
                                    st.markdown(part, unsafe_allow_html=True)
                            else:
                                chart_end = part.find("</MERMAID_CHART>")
                                if chart_end != -1:
                                    chart_code = part[:chart_end]
                                    remaining_text = part[chart_end + 16:]
                                    try:
                                        st_mermaid(chart_code, height=350)
                                    except Exception as e:
                                        st.error(f"圖表渲染失敗: {str(e)}")
                                        st.code(chart_code, language="mermaid")
                                    if remaining_text:
                                        st.markdown(remaining_text, unsafe_allow_html=True)
                else:
                    # 普通文本，直接顯示
                    message_placeholder.markdown(stream_response)
                return stream_response
                
            # 檢查是否為生成器對象
            if not hasattr(stream_response, '__iter__') or not callable(stream_response.__iter__):
                error_msg = f"無法處理 Gemini 響應：收到非流式響應對象 ({type(stream_response).__name__})"
                message_placeholder.markdown(error_msg)
                return error_msg
                
            # 安全檢查
            max_chunk_count = 1000  # 最大允許的chunk數量
            chunk_count = 0
            max_time_without_update = 30  # 最大不更新時間限制（秒）
            import time
            last_update_time = time.time()
                
            for chunk in stream_response:
                # 安全檢查
                chunk_count += 1
                if chunk_count > max_chunk_count:
                    full_response += "\n\n[Gemini 回應長，已經被截斷]"
                    message_placeholder.markdown(full_response)
                    break
                    
                # 更新時間檢查
                current_time = time.time()
                if current_time - last_update_time > max_time_without_update:
                    full_response += "\n\n[Gemini 回應超時，已經自動停止]"
                    message_placeholder.markdown(full_response)
                    break
                
                content = None
                # 處理 Gemini API 的不同響應格式
                if hasattr(chunk, 'text'):
                    content = chunk.text
                elif hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            content = part.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        content = part.text
                
                # 如果成功提取到內容，則更新顯示
                if content:
                    last_update_time = time.time()  # 更新時間戳
                    full_response += content
                    processed_content = render_mermaid_diagrams(full_response)
                    if "<MERMAID_CHART>" in processed_content:
                        # 重新渲染整個內容，包括圖表
                        message_placeholder.empty()
                        with message_placeholder.container():
                            parts = processed_content.split("<MERMAID_CHART>")
                            for i, part in enumerate(parts):
                                if i == 0:
                                    # 第一部分是純文本
                                    if part:
                                        st.markdown(part, unsafe_allow_html=True)
                                else:
                                    # 查找圖表代碼和後續文本
                                    chart_end = part.find("</MERMAID_CHART>")
                                    if chart_end != -1:
                                        chart_code = part[:chart_end]
                                        remaining_text = part[chart_end + 16:]  # 16是</MERMAID_CHART>的長度
                                        
                                        # 渲染圖表
                                        try:
                                            st_mermaid(chart_code, height=350)
                                        except Exception as e:
                                            st.error(f"圖表渲染失敗: {str(e)}")
                                            st.code(chart_code, language="mermaid")
                                        
                                        # 渲染剩餘文本
                                        if remaining_text:
                                            st.markdown(remaining_text, unsafe_allow_html=True)
                    else:
                        # 普通文本，直接更新
                        message_placeholder.markdown(full_response)
        except Exception as e:
            # 捕獲所有可能的錯誤
            error_msg = f"Gemini 串流處理發生錯誤: {str(e)}"
            message_placeholder.markdown(error_msg)
            
            # 如果錯誤是因為收到了意外的響應格式
            if "object is not iterable" in str(e) or "object is not an iterator" in str(e):
                message_placeholder.markdown("收到的 Gemini 響應不是可迭代的串流格式，請稍後再試")
                
            # 最後嘗試顯示原始響應（如果可能）
            if isinstance(stream_response, str):
                return stream_response
            else:
                try:
                    # 嘗試提取有用信息
                    response_info = "無法解析響應"
                    if hasattr(stream_response, 'text'):
                        response_info = stream_response.text
                    elif hasattr(stream_response, 'candidates') and stream_response.candidates:
                        candidate = stream_response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                part = candidate.content.parts[0]
                                if hasattr(part, 'text'):
                                    response_info = part.text
                    
                    message_placeholder.markdown(response_info)
                    return response_info
                except:
                    return error_msg
    
    return full_response


# 初始化建議問題按鈕狀態
if "show_suggestion_buttons" not in st.session_state:
    st.session_state.show_suggestion_buttons = True

# 添加一個會話狀態變量來保存用戶選擇的建議問題
if "suggestion_prompt" not in st.session_state:
    st.session_state.suggestion_prompt = None

# 檢查是否有來自建議問題的提問
prompt_from_suggestion = None
if st.session_state.suggestion_prompt:
    prompt_from_suggestion = st.session_state.suggestion_prompt
    st.session_state.suggestion_prompt = None  # 清除建議問題，避免重複處理

# PDF 檔案上傳按鈕 - 只有在沒有檔案時才顯示
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# 只有在沒有上傳檔案時才顯示上傳按鈕
if st.session_state.uploaded_file is None:
    uploaded_pdf = st.file_uploader(
        "上傳 PDF 檔案進行對話",
        type=['pdf'],
        help="支援 PDF 格式。檔案將自動上傳到 Anthropic 並可用於對話。"
    )
    
    # 如果有檔案上傳，保存到 session state 並上傳到 Anthropic
    if uploaded_pdf is not None:
        print(f"[DEBUG] ===== 用戶上傳檔案 =====")
        print(f"[DEBUG] 檔案名稱: {uploaded_pdf.name}")
        print(f"[DEBUG] 檔案類型: {uploaded_pdf.type}")
        print(f"[DEBUG] 檔案大小: {uploaded_pdf.size} bytes")
        
        with st.spinner("正在處理並上傳檔案到 Anthropic..."):
            # 讀取檔案內容
            file_content = uploaded_pdf.read()
            print(f"[DEBUG] 已讀取檔案內容，大小: {len(file_content)} bytes")
            
            # 上傳到 Anthropic Files API
            print(f"[DEBUG] 開始呼叫 upload_file_to_anthropic...")
            file_id, error, page_info = upload_file_to_anthropic(file_content, uploaded_pdf.name)
            
            print(f"[DEBUG] 上傳結果:")
            print(f"[DEBUG] - 檔案 ID: {file_id}")
            print(f"[DEBUG] - 錯誤信息: {error}")
            print(f"[DEBUG] - 頁數信息: {page_info}")
            
            if file_id:
                st.session_state.uploaded_file = uploaded_pdf
                st.session_state.uploaded_file_id = file_id
                st.session_state.uploaded_file_page_info = page_info
                
                # 根據頁數信息顯示不同的成功訊息
                if page_info and page_info.get('was_trimmed'):
                    st.warning(f"檔案 {uploaded_pdf.name} 原有 {page_info['total_pages']} 頁，已自動裁切為前 50 頁並上傳成功！")
                elif page_info and page_info.get('total_pages'):
                    st.success(f"檔案 {uploaded_pdf.name} ({page_info['total_pages']} 頁) 上傳成功！")
                else:
                    st.success(f"檔案 {uploaded_pdf.name} 上傳成功！")
                
                st.rerun()
            else:
                st.error(f"檔案上傳失敗：{error}")
                # 重置檔案上傳狀態
                st.session_state.uploaded_file = None
                if "uploaded_file_id" in st.session_state:
                    del st.session_state.uploaded_file_id
else:
    # 顯示已上傳的檔案信息和清除按鈕
    uploaded_pdf = st.session_state.uploaded_file
    
    # 創建一個固定在底部的已上傳檔案顯示區域
    page_info = st.session_state.get('uploaded_file_page_info', {})
    
    # 構建頁數信息文字
    page_text = ""
    if page_info.get('total_pages'):
        if page_info.get('was_trimmed'):
            page_text = f" - 已裁切為前 50 頁 (原 {page_info['total_pages']} 頁)"
        else:
            page_text = f" - {page_info['total_pages']} 頁"
    
    st.markdown(f"""
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                background: white; padding: 10px 15px; z-index: 1001; 
                width: auto; max-width: 500px; box-sizing: border-box; 
                border-radius: 8px; text-align: center;">
        <div>
            <span style="color: #28a745; font-weight: bold;">✓ 已上傳：</span>
            <span style="color: #333;">{uploaded_pdf.name}</span>
            <span style="color: #666; font-size: 0.9em;">({uploaded_pdf.size / 1024 / 1024:.1f} MB{page_text})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 輸入框 - 始終顯示輸入框
prompt_from_input = st.chat_input("請輸入您的問題...")

# 決定要使用哪個提示
prompt = prompt_from_suggestion if prompt_from_suggestion else prompt_from_input


# 處理用戶輸入
if prompt:
    # 當用戶輸入問題時，隱藏建議問題按鈕
    st.session_state.show_suggestion_buttons = False
    
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
    elif llm_provider == "claude" and not get_env_variable("CLAUDE_API_KEY", ""):
        with st.chat_message("assistant"):
            st.error("請設置 Claude API Key 才能使用 Claude 模型。")
    elif llm_provider == "deepseek" and not get_env_variable("DEEPSEEK_API_KEY", ""):
        with st.chat_message("assistant"):
            st.error("請設置 DeepSeek API Key 才能使用 DeepSeek 模型。")
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
                st.write(status)
    
    # 初始狀態
    update_status("思考中...")
    
    # 使用知識庫或直接回答
    voyage_api_key = get_env_variable("VOYAGE_API_KEY", "")
    uploaded_file_id = st.session_state.get('uploaded_file_id', None)
    
    # 修改邏輯：如果有檔案上傳，就不使用 RAG 搜索，避免 token 消耗過大
    if uploaded_file_id:
        # 有檔案上傳時，直接使用檔案進行對話，不搜索知識庫
        update_status("正在分析您上傳的檔案並生成回答...")
        try:
            # 獲取系統提示詞
            system_content = st.session_state.custom_prompt
            
            # 獲取最近 memory_length 條對話歷史
            # 檔案模式下為了避免 token 超限，暫時不使用聊天歷史
            print(f"[WARNING] 檔案模式：為避免 token 超限，跳過聊天歷史")
            chat_history = []  # 完全清空聊天歷史
            
            print(f"[DEBUG] 檔案模式聊天歷史總字符數: 0 (~0 tokens 估算)")
            
            # 構建消息
            messages = [{"role": "system", "content": system_content}]

            # 添加當前問題
            messages.append({"role": "user", "content": prompt})
            
            print(f"[DEBUG] ===== 使用檔案模式，跳過 RAG 搜索 =====")
            print(f"[DEBUG] 檔案 ID: {uploaded_file_id}")
            
            # 生成回答
            response_result = generate_response(messages, file_id=uploaded_file_id)
        except Exception as e:
            error_message = f"生成回答時發生錯誤: {str(e)}"
            update_status(error_message)
            response_result = [error_message, "錯誤"]
    elif st.session_state.supabase and voyage_api_key:
        # 沒有檔案上傳時，使用知識庫搜索
        update_status("正在搜索知識庫...")
        context = search_knowledge_base(prompt, update_status)
        response_result = generate_answer(prompt, context, update_status)
    else:
        # 既沒有檔案也沒有知識庫，直接使用 LLM 回答
        update_status("正在生成回答...")
        try:
            # 獲取系統提示詞
            system_content = st.session_state.custom_prompt
            
            # 獲取最近 memory_length 條對話歷史
            chat_history = st.session_state.messages[-st.session_state.memory_length*2:] if len(st.session_state.messages) > 0 else []
            
            # 智能截斷聊天歷史，避免 token 超限
            max_history_chars = 120000  # 一般對話模式的限制
            total_history_chars = 0
            filtered_chat_history = []
            
            # 從最新的訊息開始，向前添加到達限制為止
            for msg in reversed(chat_history):
                char_count = len(msg.get('content', ''))
                if total_history_chars + char_count > max_history_chars:
                    print(f"[WARNING] 一般對話模式：聊天歷史達到限制，截斷較早的訊息")
                    break
                filtered_chat_history.insert(0, msg)  # 插入到開頭保持順序
                total_history_chars += char_count
            
            chat_history = filtered_chat_history
            print(f"[DEBUG] 一般對話模式聊天歷史總字符數: {total_history_chars} (~{total_history_chars//2.5:.0f} tokens 估算)")
            
            # 構建消息
            messages = [{"role": "system", "content": system_content}]

            # 添加歷史訊息
            for message in chat_history:
                messages.append(message)

            # 添加當前問題
            messages.append({"role": "user", "content": prompt})
            
            print(f"[DEBUG] ===== 一般對話模式 =====")
            
            # 生成回答（不傳送檔案 ID）
            response_result = generate_response(messages, file_id=None)
        except Exception as e:
            error_message = f"生成回答時發生錯誤: {str(e)}"
            update_status(error_message)
            response_result = [error_message, "錯誤"]
    
    # 檢查 response_result 是否為 None 或無效
    if response_result is None:
        with st.chat_message("assistant"):
            st.error("生成回答時發生錯誤，請重新嘗試或聯繫管理員")
        # 添加錯誤消息到聊天記錄
        st.session_state.messages.append({"role": "assistant", "content": "生成回答時發生錯誤，請重新嘗試"})
        st.rerun()
    
    stream_response = response_result[0]
    is_streaming = True
    
    # 清空狀態訊息
    status_msg.empty()
    
    # 顯示回答（串流或完整）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 處理串流響應
        full_response = display_streaming_response(stream_response, message_placeholder)
        response_text = full_response  # 保存完整的回答用於添加到聊天歷史
    
    # 添加助手消息到聊天記錄
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 儲存問題和回答到 Supabase
    save_question_to_supabase(
        question=prompt,
        answer=response_text,
        prompt_name=st.session_state.prompt_name,
        knowledge_table=st.session_state.knowledge_table
    )
    
    # 重新載入頁面以重置聊天界面
    st.rerun() 