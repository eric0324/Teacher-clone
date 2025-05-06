# Galaxy Chat 🌌

一個基於Streamlit的智能聊天應用程式，整合Supabase知識庫檢索增強生成(RAG)和數位分身功能。

## 功能

- **知識庫問答**：連接Supabase知識庫，使用Voyage API生成嵌入向量進行精確檢索
- **數位分身**：支持自定義系統提示詞，打造專屬AI助手形象
- **一般對話**：當無法使用知識庫時，直接使用OpenAI進行對話
- **智能核心問題提取**：自動從用戶問題中提取核心意思和關鍵詞
- **多種檢索策略**：精確匹配、關鍵詞匹配、向量相似度搜索等多重保障

## 安裝方法

1. 複製本專案
```bash
git clone https://github.com/yourusername/galaxy-chat.git
cd galaxy-chat
```

2. 安裝依賴庫
```bash
pip install -r requirements.txt
```

3. 建立環境變數檔案`.env`（**必須**）
```
# 必須設置
OPENAI_API_KEY=your_openai_key

# 使用知識庫功能時必須設置
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-2

# 可選設置
LLM_TEMPERATURE=0.3
```

4. 啟動應用
```bash
streamlit run app.py
```

## 使用前提

- OpenAI API Key（**必須**在.env檔案中設置）
- Supabase帳號與專案（使用知識庫功能時需要在.env檔案中設置）
- Voyage AI帳號（生成向量嵌入時需要在.env檔案中設置）

## Supabase 知識庫設置

要使用RAG功能，您需要在Supabase中設置知識庫：

1. 在Supabase中創建一個表格（默認名稱為`knowledge_base`）
2. 表格結構應該包含以下列：
   - `id`: UUID (primary key)
   - `concept`: TEXT (知識點名稱/標題)
   - `explanation`: TEXT (知識點詳細解釋)
   - `embedding`: vector(1536) (知識點的向量嵌入)
3. 創建向量搜索函數`match_knowledge`

```sql
-- 啟用向量擴展
create extension if not exists vector;

-- 創建匹配函數
create or replace function match_knowledge_TABLE_NAME(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  concept text,
  explanation text,
  similarity float
)
language sql stable
as $$
  select
    id,
    concept,
    explanation,
    1 - (embedding <=> query_embedding) as similarity
  from knowledge_base
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;
```

## 使用方法

1. **必須**先在`.env`檔案中設置所有需要的API金鑰
2. 啟動應用後在瀏覽器中打開 (通常是 http://localhost:8501)
3. 在應用界面上可以看到各API連接狀態
4. 選擇使用模式:
   - **知識庫問答**：當配置好Supabase和Voyage API後，系統會從知識庫中查找相關知識
   - **數位分身模式**：在進階設置中啟用「使用數位分身」，並上傳系統提示詞檔案
   - **一般對話**：當未配置知識庫時，直接使用OpenAI進行對話
5. 在聊天框中輸入問題，開始對話

### 安全說明

為了保護您的API金鑰安全：
- 請確保`.env`檔案不會被加入版本控制系統（在`.gitignore`中排除）
- 請勿在公共環境或不安全的網絡中部署此應用程式
- 定期更換您的API金鑰以降低風險

## 技術細節

- **前端框架**: Streamlit
- **LLM整合**: OpenAI API
- **向量數據庫**: Supabase + pgvector
- **向量嵌入**: Voyage AI 