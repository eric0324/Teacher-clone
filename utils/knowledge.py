import re
import json
import requests
import streamlit as st
from utils.config import get_env_variable

def generate_embeddings(text):
    """使用 Voyage API 生成文本的向量嵌入"""
    try:
        voyage_api_key = get_env_variable("VOYAGE_API_KEY", "")
        voyage_model = get_env_variable("VOYAGE_MODEL", "voyage-2")
        
        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {voyage_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": voyage_model,
                "input": text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        st.error(f"生成嵌入時出錯: {str(e)}")
        return None

def extract_keywords(query):
    """從查詢中提取可能的關鍵詞，改進版"""
    # 常見的問句開頭和修飾詞，可能會干擾精確配斷
    stopwords = [
        '想知道', '請問', '告訴我', '關於', '誰是', '是誰', '什麼是', '的', '是',
        '嗎', '呢', '啊', '吧', '了', '哦', '喔', '耶', '呀', '？', '?',
        '請', '幫我', '可以', '能', '應該', '會', '要', '需要', '如何', '怎麼',
        '為什麼', '為何', '怎樣', '有沒有', '有什麼', '有哪些', '還有', '跟', '和', '與'
    ]
    
    # 清理標點符號和特殊字符
    cleaned_query = re.sub(r'[^\w\s]', ' ', query)
    
    # 先移除常見停用詞
    for word in stopwords:
        cleaned_query = cleaned_query.replace(word, ' ')
    
    # 分割並過濾空字串，保持原始順序
    keywords = []
    seen = set()  # 用於去重
    for k in cleaned_query.split():
        k = k.strip()
        if k and k not in seen and len(k) >= 2:  # 至少要有兩個字符
            keywords.append(k)
            seen.add(k)
    
    # 提取特殊專有名詞 - 用正則表達式識別可能的專有名詞
    potential_entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}', query)
    for entity in potential_entities:
        if entity.lower() not in [k.lower() for k in keywords]:
            keywords.append(entity)
    
    # 如果沒有找到任何關鍵詞，使用分詞方式再試一次，但保留較長的詞
    if not keywords:
        words = re.findall(r'\w+', query)
        for word in words:
            if len(word) >= 3 and word not in stopwords:
                keywords.append(word)
    
    # 產生雙詞組合
    original_keywords = keywords.copy()
    if len(original_keywords) >= 2:
        for i in range(len(original_keywords) - 1):
            combined = f"{original_keywords[i]}{original_keywords[i+1]}"
            if combined not in keywords:
                keywords.append(combined)
    
    return keywords

def extract_core_question_with_llm(query):
    """使用LLM提取查詢的核心問題和關鍵詞"""
    try:
        from utils.llm_providers import (
            generate_openai_response, 
            generate_claude_response, 
            generate_deepseek_response
        )
        
        # 檢查當前設定的LLM提供者
        llm_provider = st.session_state.get('llm_provider', 'claude')  # 預設使用 Claude
        
        # 構建查詢訊息
        system_prompt = """
        role: 關鍵詞提取專家
        instructions:
            primary_task: 從使用者問題中提取最核心的問題和關鍵詞
            rules:
                - 移除所有禮貌用語、修飾詞和冗餘內容
                - 提取可以用於向量搜索的有效關鍵詞
                - 保留專有名詞和技術術語
        output_format:
            type: json
            structure:
                core_question: 精簡後的核心問題
                keywords: 
                - 主要關鍵詞列表
        examples:
            output: |
                {
                    "core_question": "人工智慧在醫療領域的應用",
                    "keywords": ["人工智慧", "醫療", "應用"]
                }

        response_requirements:
            format: json
            structure_validation: true
            content_focus: only extracted information
        """
        
        user_prompt = f"提取這個問題的核心問題和關鍵詞：{query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 根據設定的提供者來產生回應
        if llm_provider == "openai":
            llm_model = get_env_variable("LLM_MODEL", "gpt-4o")
            response, response_type = generate_openai_response(
                messages=messages,
                model=llm_model
            )
            
            # 處理OpenAI的串流響應
            if response_type == "串流" and not isinstance(response, str):
                # 處理OpenAI串流響應
                full_text = ""
                try:
                    for chunk in response:
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content
                                if content:
                                    full_text += content
                            elif hasattr(chunk.choices[0], 'text'):
                                # 舊版API可能使用text而非content
                                content = chunk.choices[0].text
                                if content:
                                    full_text += content
                    response_text = full_text
                except Exception as e:
                    return {"core_question": query, "keywords": extract_keywords(query)}
            else:
                # 處理錯誤或直接回應
                response_text = response
                if not response_text or not isinstance(response_text, str):
                    return {"core_question": query, "keywords": extract_keywords(query)}
                
        elif llm_provider == "claude":
            claude_model = get_env_variable("CLAUDE_MODEL", "claude-sonnet-4-20250514")
            response, response_type = generate_claude_response(
                messages=messages,
                model=claude_model
            )
            
            # 處理Claude的串流響應
            if response_type == "串流" and not isinstance(response, str):
                # 處理Claude串流響應
                full_text = ""
                try:
                    for chunk in response:
                        # 處理不同類型的事件和結構
                        if hasattr(chunk, 'type'):
                            # 處理content_block_delta事件
                            if chunk.type == 'content_block_delta' and hasattr(chunk, 'delta'):
                                if hasattr(chunk.delta, 'type') and chunk.delta.type == 'text_delta':
                                    if hasattr(chunk.delta, 'text'):
                                        content = chunk.delta.text
                                        if content:
                                            full_text += content
                            # Claude 2.x 舊版API
                            elif chunk.type == 'completion' and hasattr(chunk, 'completion'):
                                content = chunk.completion
                                if content:
                                    full_text += content
                    response_text = full_text
                except Exception as e:
                    return {"core_question": query, "keywords": extract_keywords(query)}
            else:
                # 處理錯誤或直接回應
                response_text = response
                if not response_text or not isinstance(response_text, str):
                    return {"core_question": query, "keywords": extract_keywords(query)}
                
        elif llm_provider == "deepseek":
            deepseek_model = get_env_variable("DEEPSEEK_MODEL", "deepseek-chat")
            response, response_type = generate_deepseek_response(
                messages=messages,
                model_id=deepseek_model
            )
            
            # 處理DeepSeek的響應 - 可能是串流或錯誤
            if response_type == "串流" and not isinstance(response, str):
                # 手動處理串流響應抓取完整內容
                full_text = ""
                try:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: ') and line != 'data: [DONE]':
                                json_data = json.loads(line[6:])
                                if 'choices' in json_data and json_data['choices'] and 'delta' in json_data['choices'][0]:
                                    content = json_data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        full_text += content
                    response_text = full_text
                except Exception as e:
                    return {"core_question": query, "keywords": extract_keywords(query)}
            else:
                # 處理錯誤或其他情況 - 可能是字符串錯誤消息
                response_text = response
                
                if not response_text or not isinstance(response_text, str):
                    return {"core_question": query, "keywords": extract_keywords(query)}
        else:
            # 如果沒有有效的提供者，回退到基本方法
            return {"core_question": query, "keywords": extract_keywords(query)}
        
        # 處理回應
        if not response_text or isinstance(response_text, str) and not response_text.strip():
            return {"core_question": query, "keywords": extract_keywords(query)}
            
        try:
            # 嘗試解析JSON回應
            if isinstance(response_text, str):
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    clean_json = response_text[json_start:json_end]
                    parsed_result = json.loads(clean_json)
                else:
                    parsed_result = json.loads(response_text)
            else:
                # 可能已經是解析過的物件
                parsed_result = response_text
                
        except json.JSONDecodeError:
            return {"core_question": query, "keywords": extract_keywords(query)}
        
        # 確保回傳結果包含關鍵詞
        if 'keywords' not in parsed_result:
            parsed_result['keywords'] = extract_keywords(query)
        
        return parsed_result
    except Exception as e:
        st.error(f"使用LLM提取核心問題時出錯: {str(e)}")
        return {
            "core_question": query,
            "keywords": extract_keywords(query)
        }

def search_knowledge(query, match_threshold=0.7, match_count=10, rpc_func="match_knowledge_wang"):
    """從向量知識庫中搜索相關的知識點，使用多種策略提高命中率"""
    from utils.config import get_env_variable
    import streamlit as st
    
    # 獲取Supabase客戶端
    supabase = st.session_state.get('supabase')
    if not supabase:
        st.error("Supabase 客戶端未初始化")
        return []
    
    # 提取核心問題和關鍵詞，增加搜索效果
    core_result = extract_core_question_with_llm(query)
    core_question = core_result.get("core_question", query)
    all_keywords = core_result.get("keywords", extract_keywords(query))
    
    # 生成查詢的嵌入向量 - 核心問題比原始問題更能捕捉語義
    query_embedding = generate_embeddings(core_question)
    
    if not query_embedding:
        return []
    
    try:
        all_results = []  # 存儲所有匹配結果
        
        # 1. 首先嘗試直接檢查是否有完全配對的關鍵詞
        # 增加在concept和explanation欄位中搜索
        concept_query = f"%{query}%"
        explanation_query = f"%{query}%"
        
        # 修正filter方法，使用正確的參數格式
        exact_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", concept_query).execute()
        
        # 合併結果
        if hasattr(exact_result, 'data') and exact_result.data:
            # 添加相似度欄位以與向量搜索結果格式一致
            for item in exact_result.data:
                item['similarity'] = 1.0  # 設定為最高相似度
                item['match_type'] = "概念完全匹配"  # 標記匹配類型
            all_results.extend(exact_result.data)
        
        # 嘗試在解釋欄位中搜索
        exp_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("explanation", explanation_query).execute()
        
        if hasattr(exp_result, 'data') and exp_result.data:
            for item in exp_result.data:
                item['similarity'] = 0.98  # 略低於概念完全匹配
                item['match_type'] = "解釋完全匹配"
                # 避免重複添加
                if not any(existing['id'] == item['id'] for existing in all_results):
                    all_results.append(item)
        
        # 1.2 嘗試用核心問題進行匹配
        if core_question != query:
            core_concept_query = f"%{core_question}%"
            core_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", core_concept_query).execute()
            
            if hasattr(core_result, 'data') and core_result.data:
                for item in core_result.data:
                    item['similarity'] = 0.99  # 僅次於完全匹配
                    item['match_type'] = "核心問題概念匹配"
                    if not any(existing['id'] == item['id'] for existing in all_results):
                        all_results.append(item)
            
            # 同時在解釋欄位中搜索核心問題
            core_exp_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("explanation", core_concept_query).execute()
            
            if hasattr(core_exp_result, 'data') and core_exp_result.data:
                for item in core_exp_result.data:
                    item['similarity'] = 0.97  # 略低於核心問題概念匹配
                    item['match_type'] = "核心問題解釋匹配"
                    if not any(existing['id'] == item['id'] for existing in all_results):
                        all_results.append(item)
        
        # 2. 嘗試提取關鍵詞進行部分配對
        if all_keywords:
            # 2.1 單一關鍵詞搜索
            for keyword in all_keywords:
                if len(keyword) >= 2:  # 確保關鍵詞至少2個字
                    # 首先搜索概念欄位
                    keyword_concept_query = f"%{keyword}%"
                    keyword_concept_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", keyword_concept_query).execute()
                    
                    if hasattr(keyword_concept_result, 'data') and keyword_concept_result.data:
                        # 添加相似度欄位
                        for item in keyword_concept_result.data:
                            item['similarity'] = 0.95  # 關鍵詞在概念欄位匹配，相似度較高
                            item['match_type'] = f"關鍵詞在概念欄位匹配: {keyword}"
                            # 避免重複添加相同的結果
                            if not any(existing['id'] == item['id'] for existing in all_results):
                                all_results.append(item)
                    
                    # 然後搜索解釋欄位
                    keyword_exp_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("explanation", keyword_concept_query).execute()
                    
                    if hasattr(keyword_exp_result, 'data') and keyword_exp_result.data:
                        for item in keyword_exp_result.data:
                            item['similarity'] = 0.85  # 關鍵詞在解釋欄位匹配，相似度較低
                            item['match_type'] = f"關鍵詞在解釋欄位匹配: {keyword}"
                            if not any(existing['id'] == item['id'] for existing in all_results):
                                all_results.append(item)

            # 2.2 嘗試相鄰關鍵詞組合搜索（提高精確度）
            if len(all_keywords) >= 2:
                for i in range(len(all_keywords) - 1):
                    combined_keyword = f"{all_keywords[i]} {all_keywords[i+1]}"
                    if len(combined_keyword) >= 3:
                        # 搜索組合關鍵詞在概念欄位
                        combined_query = f"%{combined_keyword}%"
                        combined_concept_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("concept", combined_query).execute()
                        
                        if hasattr(combined_concept_result, 'data') and combined_concept_result.data:
                            for item in combined_concept_result.data:
                                item['similarity'] = 0.98  # 組合關鍵詞在概念欄位匹配，得分最高
                                item['match_type'] = f"關鍵詞組合在概念匹配: {combined_keyword}"
                                if not any(existing['id'] == item['id'] for existing in all_results):
                                    all_results.append(item)
                        
                        # 搜索組合關鍵詞在解釋欄位
                        combined_exp_result = supabase.table(st.session_state.knowledge_table).select("*").ilike("explanation", combined_query).execute()
                        
                        if hasattr(combined_exp_result, 'data') and combined_exp_result.data:
                            for item in combined_exp_result.data:
                                item['similarity'] = 0.9  # 組合關鍵詞在解釋欄位匹配
                                item['match_type'] = f"關鍵詞組合在解釋匹配: {combined_keyword}"
                                if not any(existing['id'] == item['id'] for existing in all_results):
                                    all_results.append(item)
                                    
        # 如果已經找到足夠的匹配結果，直接返回
        if len(all_results) >= max(3, match_count // 2):
            # 根據相似度排序結果
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_results[:match_count]
        
        # 3. 使用向量搜索進行相似性搜索 - 使用核心問題的向量，更能捕捉語義
        result = supabase.rpc(
            rpc_func, 
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }
        ).execute()
        
        # 處理向量搜索結果
        if hasattr(result, 'data') and result.data:
            for item in result.data:
                item['match_type'] = "向量相似度匹配"
                if not any(existing['id'] == item['id'] for existing in all_results):
                    all_results.append(item)
        
        # 後續向量搜索降低閾值的部分保持不變
        if len(all_results) < max(3, match_count // 2) and match_threshold > 0.5:
            lower_threshold_result = supabase.rpc(
                rpc_func, 
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,  # 降低閾值
                    "match_count": match_count
                }
            ).execute()
            
            if hasattr(lower_threshold_result, 'data') and lower_threshold_result.data:
                for item in lower_threshold_result.data:
                    item['match_type'] = "低閾值向量匹配"
                    if not any(existing['id'] == item['id'] for existing in all_results):
                        all_results.append(item)
        
        # 如果還是找不到足夠結果，再降低閾值到更低
        if len(all_results) < max(2, match_count // 3) and match_threshold > 0.3:
            lowest_threshold_result = supabase.rpc(
                rpc_func, 
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.3,  # 極低閾值
                    "match_count": match_count
                }
            ).execute()
            
            if hasattr(lowest_threshold_result, 'data') and lowest_threshold_result.data:
                for item in lowest_threshold_result.data:
                    item['match_type'] = "最低閾值向量匹配"
                    if not any(existing['id'] == item['id'] for existing in all_results):
                        all_results.append(item)
            
        # 4. 最後搜索邏輯保持不變
        if len(all_results) == 0:
            backup_result = supabase.table(st.session_state.knowledge_table).select("*").execute()
            if hasattr(backup_result, 'data') and backup_result.data:
                # 簡單的文本配對
                for item in backup_result.data:
                    combined_text = f"{item['concept']}: {item['explanation']}".lower()
                    # 檢查是否包含查詢中的任何關鍵詞
                    matched_keywords = [keyword for keyword in all_keywords if keyword.lower() in combined_text]
                    if matched_keywords:
                        # 相似度根據匹配到的關鍵詞數量調整
                        match_score = min(0.6 + 0.05 * len(matched_keywords), 0.75)  
                        item['similarity'] = match_score
                        item['match_type'] = f"文本包含匹配: {', '.join(matched_keywords[:3])}"
                        all_results.append(item)
        
        # 根據相似度排序結果
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:match_count]
    
    except Exception as e:
        st.error(f"搜索知識時出錯: {str(e)}")
        return [] 