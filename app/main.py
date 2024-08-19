import streamlit as st
from utils import print_messages
from langchain_core.messages import ChatMessage

# 페이지 타이틀 설정
st.set_page_config(page_title="MyEcho", page_icon="🦦") 
st.title("Chat MyEcho 🦦")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

print_messages()

# streamlit 채팅 컴포넌트 사용
if user_input := st.chat_input("메세지를 입력해주세요"):
    st.chat_message("user").write(f"{user_input}")
    # st.session_state["messages"].append(("user", user_input))
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    with st.chat_message("assistant"):
        msg = f"당신이 입력한 내용: {user_input}"
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role="assistant", content=msg))