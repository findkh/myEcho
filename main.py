import os
import streamlit as st
import ollama
import pandas as pd
from langchain_core.messages import ChatMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 벡터 스토어 초기화 함수
def initialize_vectorstore(persist_directory, embedding_model):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    # CSV 파일에서 데이터 읽기
    df = pd.read_csv("./watermoonInfo.csv")
    texts = df['text'].tolist()
    metadatas = [{"text": text} for text in texts]
    return Chroma.from_texts(texts, embedding_model, metadatas=metadatas, persist_directory=persist_directory)

# 세션 상태 초기화 함수
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

# 메시지 저장 함수
def save_message(role, content):
    message = ChatMessage(role=role, content=content)
    st.session_state['messages'].append(message)

# 프롬프트 템플릿 생성 함수
def create_prompt_template(user_input, info):
    all_messages = st.session_state['messages']
    
    # 전체 메시지를 역순으로 정렬
    reversed_messages = list(reversed(all_messages))
    
    # 최신 질문을 제외한 메시지 리스트 생성
    if reversed_messages and reversed_messages[0].role == "user":
        reversed_messages = reversed_messages[1:]
    
    # 최신 6개의 대화 쌍만 포함
    history = reversed_messages[:6]
    
    # 최근 대화 내용 포맷팅
    recent_message_pairs = "\n".join(f"{msg.role}: {msg.content}" for msg in reversed(history))

    return f"""
        나의 이름은 'MyEcho'로, AI 어시스턴트입니다. 
        저는 오직 'watermoon'이라는 사람에 대한 정보만을 바탕으로 답변을 제공합니다. 
        'watermoon'은 이 프로그램을 만든 실존 인물입니다.
        
        아래에 제공된 정보만을 참고하여 질문에 답변하세요. 주어진 정보 외의 내용을 추측하거나 추가하지 마세요:
        {info}
        
        최근 대화 내용:
        {recent_message_pairs}
        
        사용자가 물어본 질문: {user_input}

        답변:
    """

# 사용자 입력 처리 함수
def process_input(user_input, retriever):
    if any(keyword in user_input for keyword in ["요약", "안녕", "하이"]):
        info = ""  # 벡터 DB를 사용하지 않고 빈값으로 설정
        return info
    else:
        # retriever를 사용하여 문서 가져오기
        result = retriever.get_relevant_documents(user_input)

        # 가져온 문서의 page_content를 직접 반환
        return result[0].page_content

# 채팅 히스토리 표시 함수
def display_chat_history():
    for message in st.session_state['messages']:
        st.chat_message(message.role).write(message.content)

# LLM 스트리밍 응답 함수
def llm_stream(model_name, prompt):
    response = ollama.chat(model_name, [{"content": prompt, "role": "user"}], stream=True)
    for chunk in response:
        yield chunk['message']['content']

# 모델과 Embedding 초기화
embedding_model = HuggingFaceEmbeddings(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')
persist_directory = "./chromadb_data"
vectorstore = initialize_vectorstore(persist_directory, embedding_model)

# Retriever 설정
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

# Streamlit 페이지 설정
st.set_page_config(page_title="MyEcho", page_icon="🦦")
st.title("Chat MyEcho 🦦")

# 세션 상태 초기화
initialize_session_state()

# 사이드바에서 대화 제목 입력
st.sidebar.header("Chat Settings")
chat_title = st.sidebar.text_input("Enter chat title", value="My Chat")

# 사이드바에서 모델 선택
try:
    OLLAMA_MODELS = ollama.list()["models"]
except Exception as e:
    st.warning("Ollama를 먼저 설치해주세요.")
    st.stop()

model_names = [model["name"] for model in OLLAMA_MODELS]
llm_name = st.sidebar.selectbox("Choose Agent", [""] + model_names)
if not llm_name:
    st.stop()

# 채팅 메시지 출력
display_chat_history()

# 사용자 입력 처리
user_input = st.chat_input("Watermoon님에 대한 궁금한 사항을 물어보세요! 👀")

if user_input:
    st.chat_message("user").write(user_input)
    save_message("user", user_input)
    info = process_input(user_input, retriever)
    prompt = create_prompt_template(user_input, info)

    # AI 응답을 스트리밍하면서 UI에 표시
    with st.chat_message("assistant"):
        chat_box = st.empty()
        response_message = chat_box.write_stream(llm_stream(llm_name, prompt))

    # 최종 응답을 세션 상태에 저장
    save_message("assistant", response_message)

# 사이드바에서 대화 다운로드 버튼
if st.session_state['messages']:
    chat_history_text = "\n".join(f"{msg.role}: {msg.content}" for msg in st.session_state['messages'])
    st.sidebar.download_button(
        label="Download Chat",
        data=chat_history_text,
        file_name=f"{chat_title}.txt",
        mime="text/plain"
    )

# 대화 리셋 버튼 구현
if st.session_state['messages'] and st.sidebar.button("Reset Chat"):
    st.session_state.clear()
    st.rerun()
