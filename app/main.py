import streamlit as st
from langchain_community.llms import Ollama

# Ollama 모델 초기화
llm = Ollama(model="EEVE-Korean-Instruct-10.8B")

# Streamlit 페이지 설정
st.set_page_config(page_title="MyEcho", page_icon="🦦")
st.title("Chat MyEcho 🦦")

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 메시지 출력 함수
def print_messages():
    """세션 상태의 메시지를 화면에 출력하는 함수"""
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        elif message['role'] == 'assistant':
            st.chat_message("assistant").write(message['content'])

# 채팅 메시지 출력
print_messages()

# 사용자 입력 처리
user_input = st.chat_input("Watermoon님에 대한 궁금한 사항을 물어보세요! 👀")

if user_input:
    # 사용자 메시지 저장
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    
    # 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # 프롬프트를 설정하여 Watermoon에 대한 답변 생성
    prompt = f"나는 'MyEcho'라는 이름의 AI 어시스턴트입니다. 사용자가 어떤 질문을 하든, 모든 질문에 대해 'Watermoon'이라는 사람에 관한 답변을 제공합니다. 질문: {user_input}\n답변:"

    # 응답 생성 중 상태 표시
    response_placeholder = st.empty()  # 응답을 실시간으로 업데이트할 빈 공간
    with st.spinner("답변 생성중..."):
        response = ""
        # `st.chat_message("assistant")`를 사용하여 응답을 추가
        for chunk in llm.stream(prompt):
            response += chunk
            # 빈 공간에 실시간으로 응답 업데이트
            response_placeholder.write(response, end='', flush=True)
    
    # 최종 응답 저장 및 화면에 출력
    st.session_state['messages'].append({'role': 'assistant', 'content': response})
    response_placeholder.empty()  # 빈 공간을 비우고 최종 응답을 출력
    st.chat_message("assistant").write(response)



# import streamlit as st
# from langchain_community.llms import Ollama

# # Ollama 모델 초기화
# llm = Ollama(model="EEVE-Korean-Instruct-10.8B")

# # Streamlit 페이지 설정
# st.set_page_config(page_title="MyEcho", page_icon="🦦")
# st.title("Chat MyEcho 🦦")

# # 세션 상태 초기화
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = []

# # 메시지 출력 함수
# def print_messages():
#     """세션 상태의 메시지를 화면에 출력하는 함수"""
#     for message in st.session_state['messages']:
#         if message['role'] == 'user':
#             st.chat_message("user").write(message['content'])
#         elif message['role'] == 'assistant':
#             st.chat_message("assistant").write(message['content'])

# # 채팅 메시지 출력
# print_messages()

# # 사용자 입력 처리
# user_input = st.chat_input("수달님에 대한 궁금한 사항을 물어보세요! 👀")

# if user_input:
#     # 사용자 메시지 저장
#     st.session_state['messages'].append({'role': 'user', 'content': user_input})
    
#     # 사용자 메시지 출력
#     st.chat_message("user").write(user_input)
    
#     # 응답 생성
#     response_placeholder = st.empty()  # 응답을 실시간으로 업데이트할 빈 공간

#     # 프롬프트에 역할과 주제 설정
#     prompt = f"나는 'MyEcho'라는 이름의 AI 어시스턴트입니다. 내가 답변할 수 있는 주제는 오직 '수달'에 관한 것입니다. 사용자가 '수달'에 대한 질문을 하면 정확하게 답변해주세요. 사용자가 다른 주제에 대해 질문하면, '수달'에 관한 질문을 해달라고 안내해주세요.\n\n질문: {user_input}\n답변:"

#     # 응답 생성 중 상태 표시
#     with st.spinner("답변 생성중..."):
#         response = ""
#         # `st.chat_message("assistant")`를 사용하여 응답을 추가
#         for chunk in llm.stream(prompt):
#             response += chunk
#             # 빈 공간에 실시간으로 응답 업데이트
#             response_placeholder.write(response, end='', flush=True)
    
#     # 최종 응답 저장 및 화면에 출력
#     st.session_state['messages'].append({'role': 'assistant', 'content': response})
#     response_placeholder.empty()  # 빈 공간을 비우고 최종 응답을 출력
#     st.chat_message("assistant").write(response)