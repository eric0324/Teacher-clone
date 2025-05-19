import streamlit as st
import datetime
from utils.config import load_config, get_env_variable
from utils.auth import check_password

# 設置頁面配置和標題
st.set_page_config(page_title="問題記錄查詢", layout="wide", page_icon="📊")

# 完全隱藏側邊欄和所有收起箭頭
st.markdown("""
<style>
    /* 通用隱藏規則 - 直接隱藏整個頂部欄位避免所有按鈕 */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* 隱藏側邊欄 */
    [data-testid="stSidebar"], aside.st-emotion-cache-16txtl3, aside.st-emotion-cache-4oy321 {
        display: none !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        visibility: hidden !important;
        z-index: -1 !important;
        overflow: hidden !important;
        opacity: 0 !important;
    }
    
    /* 隱藏側邊欄控制元素 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarNav"], 
    div:has([data-testid="stSidebarCollapsedControl"]),
    section[data-testid="stSidebarContent"] {
        display: none !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* 徹底隱藏所有按鈕和圖標 */
    button[kind="headerNoPadding"], 
    button[data-testid="baseButton-headerNoPadding"],
    button.st-emotion-cache-1w7bu1y, 
    .st-emotion-cache-169dgwr,
    .st-emotion-cache-gsulwm,
    div.st-emotion-cache-gsulwm,
    .edtmxes14,
    div[data-testid="stDecoration"],
    div.st-emotion-cache-16j9m0,
    div.st-emotion-cache-16j9m1 {
        display: none !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* 徹底隱藏所有SVG箭頭和圖標 */
    svg, 
    svg[class*="st-emotion"], 
    svg path[d*="M10 6"] {
        display: none !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* 添加額外空間讓上方區域不顯得空蕩 */
    .block-container {
        padding-top: 1rem !important;
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

# 創建表格
if filtered_records:
    data = []
    for record in filtered_records:
        data.append({
            "時間": format_datetime(record.get('created_at', '')),
            "問題": record.get('question', ''),
            "提示詞": record.get('prompt_name', ''),
            "知識庫": record.get('knowledge_table', ''),
            "模型": record.get('llm_provider', '')
        })
    
    st.table(data)
else:
    st.info("沒有符合過濾條件的記錄。")

