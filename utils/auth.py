import streamlit as st
from utils.config import get_env_variable

def check_password():
    """返回`True`如果用戶輸入正確的用戶名和密碼"""
    
    # 從環境變數讀取認證信息
    correct_username = get_env_variable("AUTH_USERNAME", "admin")
    correct_password = get_env_variable("AUTH_PASSWORD", "password")
    
    # 檢查是否已經登入
    if "authentication_status" in st.session_state and st.session_state["authentication_status"]:
        return True
    
    # 顯示登入表單
    st.title("數位分身系統 (Beta)")
    st.header("登入")
    
    # 使用表單，可以讓用戶按下 Enter 鍵提交
    with st.form("login_form"):
        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        submit_button = st.form_submit_button("登入")
        
        if submit_button:
            if username == correct_username and password == correct_password:
                st.session_state["authentication_status"] = True
                st.rerun()  # 強制頁面重新渲染，確保表單消失
                return True
            else:
                st.error("帳號或密碼錯誤")
                return False
    
    return False 