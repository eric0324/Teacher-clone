import streamlit as st
from utils.config import get_env_variable, load_system_prompt
import os
from streamlit_mermaid import st_mermaid
import re

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
        background-color: white;
        border: 1px solid #000;
    }

    th {
        background-color: white;
        color: #333;
        font-weight: bold;
        text-align: left;
        padding: 10px;
        border: 1px solid #000;
    }

    td {
        padding: 8px 10px;
        border: 1px solid #000;
        background-color: white;
    }

    tr:nth-child(even) {
        background-color: white;
    }

    tr:hover {
        background-color: white;
    }

    /* 建議問題按鈕樣式 */
    div[data-testid="stButton"] button {
        border-radius: 12px;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        color: #333;
        font-size: 0.9em;
        margin-bottom: 8px;
        width: 100%;
        height: auto;
        min-height: 55px;
        text-align: left;
        padding: 8px 12px;
        transition: all 0.3s ease;
        white-space: normal;
        line-height: 1.4;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #e6e9ef;
        border-color: #bbb;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
        llm_provider_options = ["claude", "deepseek", "gemini", "openai"]
        selected_provider = st.selectbox(
            "選擇大語言模型供應商", 
            llm_provider_options,
            index=llm_provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in llm_provider_options else 0
        )
        
        if selected_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = selected_provider
            st.success(f"已切換到 {selected_provider} 模型")
        
        # 串流模式始終啟用
        st.session_state.use_streaming = True
        
        # 根據供應商顯示對應的模型選擇
        if st.session_state.llm_provider == "openai":
            openai_models = [
                "gpt-4.1",
                "gpt-4o"
            ]
            llm_model = get_env_variable("LLM_MODEL", "gpt-4.1")
            selected_openai_model = st.selectbox(
                "選擇 OpenAI 模型", 
                openai_models, 
                index=openai_models.index(llm_model) if llm_model in openai_models else 0
            )
            
            # 保存選定的 OpenAI 模型
            if selected_openai_model != llm_model:
                os.environ["LLM_MODEL"] = selected_openai_model
                st.success(f"已切換到 {selected_openai_model} 模型")
        elif st.session_state.llm_provider == "claude":
            claude_models = [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
            ]
            claude_model = get_env_variable("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
            selected_claude_model = st.selectbox(
                "選擇 Claude 模型", 
                claude_models, 
                index=claude_models.index(claude_model) if claude_model in claude_models else 0
            )
            
            # 保存選定的 Claude 模型
            if selected_claude_model != claude_model:
                os.environ["CLAUDE_MODEL"] = selected_claude_model
                st.success(f"已切換到 {selected_claude_model} 模型")
        elif st.session_state.llm_provider == "gemini":
            # Gemini 模型選擇
            gemini_models = [
                "gemini-2.0-flash"
            ]
            gemini_model = get_env_variable("GEMINI_MODEL", "gemini-1.5-flash")
            selected_gemini_model = st.selectbox(
                "選擇 Gemini 模型", 
                gemini_models, 
                index=gemini_models.index(gemini_model) if gemini_model in gemini_models else 0
            )
            
            # 保存選定的 Gemini 模型
            if selected_gemini_model != gemini_model:
                os.environ["GEMINI_MODEL"] = selected_gemini_model
                st.success(f"已切換到 {selected_gemini_model} 模型")
                
        elif st.session_state.llm_provider == "deepseek":
            # DeepSeek 模型選擇
            deepseek_models = [
                "deepseek-chat"
            ]
            deepseek_model = get_env_variable("DEEPSEEK_MODEL", "deepseek-chat")
            selected_deepseek_model = st.selectbox(
                "選擇 DeepSeek 模型", 
                deepseek_models, 
                index=deepseek_models.index(deepseek_model) if deepseek_model in deepseek_models else 0
            )
        
        # 進階設置
        with st.expander("進階設置"):
            st.session_state.knowledge_table = st.text_input("知識庫資料表名稱", value=st.session_state.knowledge_table)
            
            # 輸入系統提示詞名稱
            prompt_name = st.text_input("系統提示詞名稱", value=st.session_state.prompt_name)
            if prompt_name != st.session_state.prompt_name:
                st.session_state.prompt_name = prompt_name
                new_prompt = load_system_prompt(prompt_name)
                if new_prompt:
                    st.session_state.custom_prompt = new_prompt
                    st.success(f"已載入提示詞: {prompt_name}")
                    # 顯示載入的提示詞內容
                    with st.expander("查看提示詞內容"):
                        st.text_area("系統提示詞", value=new_prompt, height=200, disabled=True)
                else:
                    st.error(f"無法載入提示詞: {prompt_name}")
                    with st.expander("除錯資訊"):
                        st.warning("未能從 Supabase 獲取提示詞，請檢查數據庫連接及提示詞名稱是否正確")
                        
            st.slider("AI 回答溫度", min_value=0.0, max_value=1.0, value=float(get_env_variable("LLM_TEMPERATURE", "0.3")), step=0.1)
            
            # 添加記憶長度設置
            memory_length = st.slider("對話記憶長度", min_value=1, max_value=10, value=st.session_state.memory_length, step=1)
            if memory_length != st.session_state.memory_length:
                st.session_state.memory_length = memory_length
        
        # 添加登出按鈕
        if st.button("登出"):
            st.session_state["authentication_status"] = False
            st.rerun()

def fix_markdown_table(content):
    """修復格式可能不正確的 Markdown 表格"""
    # 查找表格的模式 - 查找以 | 開頭和結尾的行，後面跟著一行分隔符
    table_pattern = r'(\|[^\n]+\|(?:\s*\n\|[^\n]+\|)+)'
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    # 如果沒有找到表格，直接返回原始內容
    if not tables:
        return content
    
    # 處理每個表格
    for original_table in tables:
        # 分割表格行
        lines = original_table.strip().split('\n')
        if len(lines) < 2:  # 需要至少兩行
            continue
            
        # 解析行，將每行分成單元格
        parsed_rows = []
        for line in lines:
            line = line.strip()
            if not line.startswith('|') or not line.endswith('|'):
                continue  # 不是有效的表格行
                
            # 去掉開頭和結尾的 |，然後分割
            cells = [cell.strip() for cell in line[1:-1].split('|')]
            parsed_rows.append(cells)
            
        if len(parsed_rows) < 2:
            continue  # 沒有足夠的行
            
        # 確定列數 (使用第一行的列數)
        column_count = len(parsed_rows[0])
        
        # 檢查第二行是否是分隔符行 (包含 --- 或 :--:)
        is_second_row_separator = all('-' in cell or ':' in cell for cell in parsed_rows[1])
        
        # 準備新的表格行
        new_table_lines = []
        
        # 添加標題行
        header_cells = parsed_rows[0]
        new_table_lines.append('| ' + ' | '.join(header_cells) + ' |')
        
        # 生成標準分隔行
        separator_line = '| ' + ' | '.join(['---' for _ in range(column_count)]) + ' |'
        new_table_lines.append(separator_line)
        
        # 添加數據行
        start_idx = 2 if is_second_row_separator else 1
        for row in parsed_rows[start_idx:]:
            # 如果行的單元格數與標題行不同，調整它
            if len(row) < column_count:
                row.extend([''] * (column_count - len(row)))
            elif len(row) > column_count:
                row = row[:column_count]
                
            new_table_lines.append('| ' + ' | '.join(row) + ' |')
            
        # 生成修復後的表格
        fixed_table = '\n'.join(new_table_lines)
        
        # 替換原始表格
        content = content.replace(original_table, fixed_table)
        
    return content

def render_mermaid_diagrams(content):
    """從文本中提取並渲染Mermaid圖表，返回處理後的文本"""
    # 首先修復可能存在的表格問題
    content = fix_markdown_table(content)
    
    # 檢查是否包含mermaid圖表
    if "```mermaid" in content or "```pie" in content or "```graph" in content:
        # 分割內容，處理每個部分
        parts = []
        current_pos = 0
        
        # 尋找mermaid代碼區塊
        while True:
            # 查找開始標記
            start_pos = -1
            for marker in ["```mermaid", "```pie", "```graph"]:
                pos = content.find(marker, current_pos)
                if pos != -1 and (start_pos == -1 or pos < start_pos):
                    start_pos = pos
                    start_marker = marker
            
            if start_pos == -1:
                # 沒有找到更多的圖表代碼塊
                parts.append(content[current_pos:])
                break
            
            # 添加圖表之前的文本
            if start_pos > current_pos:
                parts.append(content[current_pos:start_pos])
            
            # 查找結束標記
            end_pos = content.find("```", start_pos + len(start_marker))
            if end_pos == -1:
                # 如果沒有找到結束標記，將剩餘內容作為普通文本
                parts.append(content[start_pos:])
                break
            
            # 提取圖表代碼
            chart_code = content[start_pos + len(start_marker):end_pos].strip()
            
            # 添加圖表的渲染標記
            parts.append(f"<MERMAID_CHART>{chart_code}</MERMAID_CHART>")
            
            # 更新位置
            current_pos = end_pos + 3
        
        # 重新組合內容
        processed_content = "".join(parts)
        return processed_content
    else:
        return content

def display_chat_history():
    """顯示對話歷史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            # 處理可能包含的mermaid圖表
            processed_content = render_mermaid_diagrams(content)
            
            # 如果內容中有圖表標記，則分別渲染
            if "<MERMAID_CHART>" in processed_content:
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
                # 沒有圖表，直接渲染
                st.markdown(content, unsafe_allow_html=True) 