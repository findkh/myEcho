import streamlit as st
from langchain_community.llms import Ollama
from utils import print_messages
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Ollama 모델 초기화
llm = Ollama(model="EEVE-Korean-Instruct-10.8B")

# Streamlit 페이지 설정
st.set_page_config(page_title="MyEcho", page_icon="🦦")
st.title("Chat MyEcho 🦦")

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ChatMessageHistory()

# 채팅 메시지 출력
print_messages()

# 사용자 입력 처리
user_input = st.chat_input("Watermoon님에 대한 궁금한 사항을 물어보세요! 👀")

if user_input:
    # 사용자 메시지 저장 및 히스토리에 추가
    st.session_state['messages'].append(ChatMessage(role="user", content=user_input))
    st.session_state['chat_history'].add_user_message(user_input)

    # 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # 프롬프트를 설정하여 Watermoon에 대한 답변 생성
    prompt = ChatPromptTemplate.from_template("""
                나는 'MyEcho'로 AI 어시스턴트입니다.
                사용자가 어떤 질문을 하든, 모든 질문에 대해 'Watermoon'이라는 사람에 관한 답변을 간략하게 제공합니다. 
                질문: {query}\n 답변:""")

    # 응답 생성 중 상태 표시
    response_placeholder = st.empty()  # 응답을 실시간으로 업데이트할 빈 공간
    with st.spinner("답변 생성중..."):
        # 이전 대화 내용 추가
        messages = st.session_state['chat_history'].messages
        
        # 최근 5개의 질문과 답변 세트만 사용
        recent_messages = messages[-10:]  # 5개의 질문과 답변 세트는 총 10개의 메시지
        
        # 메시지를 문자열로 변환
        previous_messages = ""
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                previous_messages += f"human: {recent_messages[i].content}\n"
                previous_messages += f"ai: {recent_messages[i + 1].content}\n"
        
        final_prompt = f"{previous_messages}\n\n{prompt.format(query=user_input)}"

        response = ""
        # LLM의 스트리밍 기능 사용
        for chunk in llm.stream(final_prompt):
            response += chunk
            # 빈 공간에 실시간으로 응답 업데이트
            response_placeholder.write(response)
    
    # 최종 응답 저장 및 화면에 출력
    st.session_state['messages'].append(ChatMessage(role="assistant", content=response))
    st.session_state['chat_history'].add_ai_message(response)
    response_placeholder.empty()  # 빈 공간을 비우고 최종 응답을 출력
    st.chat_message("assistant").write(response)