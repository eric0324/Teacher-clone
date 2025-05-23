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
    save_question_to_supabase
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
    st.session_state.prompt_name = "wang"

if "knowledge_table" not in st.session_state:
    st.session_state.knowledge_table = get_env_variable("KNOWLEDGE_TABLE", "knowledge_base")  

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "claude"  # 預設使用 Claude

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True  # 預設啟用串流回應

if "memory_length" not in st.session_state:
    st.session_state.memory_length = 5
    
# 載入配置
config = load_config()

# 將配置添加到會話狀態中
st.session_state.supabase = config.get("supabase")

# 再載入系統提示詞
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = load_system_prompt(st.session_state.prompt_name)

# 聊天界面部分
st.subheader("數位分身聊天")

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
            
            # 準備知識點信息
            knowledge_info = []
            for item in knowledge_points:
                match_info = f"({item.get('match_type', '未知匹配類型')}, 相似度: {item.get('similarity', 0):.2f})"
                knowledge_info.append(f"概念: {item['concept']} {match_info}\n解釋: {item['explanation']}")
            
            # 將知識點整合為上下文
            context = "\n\n".join(knowledge_info)
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
        chat_history = st.session_state.messages[-st.session_state.memory_length*2:] if len(st.session_state.messages) > 0 else []
        
        # 構建消息
        messages = [{"role": "system", "content": system_content}]

        # 添加歷史訊息
        for message in chat_history:
            messages.append(message)

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
        
        # 生成回答
        return generate_response(messages)
    except Exception as e:
        error_message = f"生成回答時發生錯誤: {str(e)}"
        update_status(error_message)
        # 返回錯誤消息，而不是 None
        return [error_message, "錯誤"]

def generate_response(messages):
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
            model=claude_model
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
            model=claude_model
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
    if st.session_state.supabase and voyage_api_key:
        context = search_knowledge_base(prompt, update_status)
        response_result = generate_answer(prompt, context, update_status)
    
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