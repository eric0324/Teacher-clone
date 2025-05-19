import streamlit as st
import datetime
from utils.config import load_config, get_env_variable
from utils.auth import check_password

# 設置頁面配置和標題
st.set_page_config(page_title="問題記錄查詢", layout="wide", page_icon="📊")

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

# 載入配置
config = load_config()
supabase = config.get("supabase")

# 顯示問題記錄頁面內容
st.title("📊 問題記錄查詢")
st.caption("查看用戶提問歷史紀錄")

def get_question_records(limit=100):
    """從 Supabase 獲取問題記錄"""
    try:
        if not supabase:
            return []
            
        # 從 question_logs 表中獲取最近的記錄
        result = supabase.table("question_logs").select("*").order("created_at", desc=True).limit(limit).execute()
        
        if result and hasattr(result, 'data'):
            return result.data
        return []
    except Exception as e:
        st.error(f"獲取問題記錄時出錯: {str(e)}")
        return []

def format_datetime(date_string):
    """格式化日期時間字符串"""
    try:
        # 解析ISO格式的日期時間
        dt = datetime.datetime.fromisoformat(date_string)
        # 格式化為更易讀的形式
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return date_string

# 獲取記錄
records = get_question_records(limit=100)

if not records:
    st.info("目前沒有任何問題記錄，或者連接到 Supabase 時出現問題。")
    st.stop()

# 過濾選項
col1, col2 = st.columns(2)

with col1:
    # 獲取所有獨特的提示詞名稱
    prompt_names = list(set(record.get('prompt_name', '') for record in records if record.get('prompt_name')))
    prompt_names.insert(0, "全部")  # 添加"全部"選項
    
    selected_prompt = st.selectbox("按提示詞過濾", prompt_names)

with col2:
    # 獲取所有獨特的知識庫名稱
    knowledge_tables = list(set(record.get('knowledge_table', '') for record in records if record.get('knowledge_table')))
    knowledge_tables.insert(0, "全部")  # 添加"全部"選項
    
    selected_knowledge = st.selectbox("按知識庫過濾", knowledge_tables)

# 過濾記錄
filtered_records = records
if selected_prompt != "全部":
    filtered_records = [r for r in filtered_records if r.get('prompt_name') == selected_prompt]
if selected_knowledge != "全部":
    filtered_records = [r for r in filtered_records if r.get('knowledge_table') == selected_knowledge]

# 顯示記錄
st.subheader(f"找到 {len(filtered_records)} 條記錄")

# 創建自定義表格顯示
if filtered_records:
    # 將數據結構化為表格形式
    data = []
    for record in filtered_records:
        data.append({
            "時間": format_datetime(record.get('created_at', '')),
            "問題": record.get('question', ''),
            "回答": record.get('answer', ''),
            "提示詞": record.get('prompt_name', ''),
            "知識庫": record.get('knowledge_table', ''),
            "模型": record.get('llm_provider', '')
        })
    
    # 添加一些自定義樣式
    st.markdown("""
    <style>
    .record-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    .record-header {
        margin-bottom: 10px;
    }
    .expander-content {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-top: 5px;
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 逐筆顯示記錄
    for i, item in enumerate(data):
        st.markdown(f"""
        <div class="record-card">
            <div class="record-header">
                <strong>時間:</strong> {item["時間"]} | 
                <strong>提示詞:</strong> {item["提示詞"]} | 
                <strong>知識庫:</strong> {item["知識庫"]} | 
                <strong>模型:</strong> {item["模型"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 問題部分 - 可收合
            question = item["問題"] or ""
            if len(question) > 30:
                with st.expander(f"**問題:** {question[:30]}...", expanded=False):
                    st.markdown(f'<div class="expander-content">{question}</div>', unsafe_allow_html=True)
            else:
                st.write(f"**問題:** {question}")
        
        with col2:
            # 回答部分 - 可收合
            answer = item["回答"] or ""
            if len(answer) > 30:
                with st.expander(f"**回答:** {answer[:30]}...", expanded=False):
                    st.markdown(f'<div class="expander-content">{answer}</div>', unsafe_allow_html=True)
            else:
                st.write(f"**回答:** {answer}")
        
        st.markdown("---")
else:
    st.info("沒有符合過濾條件的記錄。")

