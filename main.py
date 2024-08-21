import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import pandas as pd
import os

# ChromaDB와 SentenceTransformer 임베딩 초기화
embedding_model = HuggingFaceEmbeddings(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')
persist_directory = "./chromadb_data"
collection_name = "test"

# ChromaDB 벡터 스토어 초기화 및 데이터 로드
if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
    print("기존 데이터베이스 로드 중...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
else:
    print("데이터베이스 생성 중...")
    df = pd.read_csv("./watermoonInfo.csv")
    texts = df['text'].tolist()
    metadatas = [{"text": text} for text in texts]
    
    vectorstore = Chroma.from_texts(texts, embedding_model, metadatas=metadatas, persist_directory=persist_directory)

# Retriever 초기화
retriever = VectorStoreRetriever(vectorstore=vectorstore)

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

# 채팅 메시지 출력 함수
def print_messages():
    """세션 상태의 메시지를 출력하는 함수."""
    for message in st.session_state['messages']:
        st.chat_message(message.role).write(message.content)

print_messages()

# 사용자 입력 처리
user_input = st.chat_input("Watermoon님에 대한 궁금한 사항을 물어보세요! 👀")

if user_input:
    # 사용자 메시지 저장 및 히스토리에 추가
    st.session_state['messages'].append(ChatMessage(role="user", content=user_input))
    st.session_state['chat_history'].add_user_message(user_input)
    st.chat_message("user").write(user_input)

    # 프롬프트 설정
    prompt = ChatPromptTemplate.from_template("""
                나의 이름은 'MyEcho'로 AI 어시스턴트입니다.
                사용자가 어떤 질문을 하든, 모든 질문에 대해 'Watermoon'이라는 사람에 관한 답변을 간략하게 제공합니다. 
                사용자가 물어본 질문: {query}
                아래의 정보를 기반으로 질문과 가장 유사한 정보를 선택하여 답변 작성해주세요:
                {info}
                답변:""")

    # 응답 생성 중 상태 표시
    response_placeholder = st.empty()
    # with st.spinner("답변 생성 중..."):
    # 최근 대화 5개 추출
    recent_messages = st.session_state['chat_history'].messages[-10:]

    # 이전 대화를 프롬프트에 추가
    previous_messages = "\n".join(
        f"human: {msg.content}" if i % 2 == 0 else f"ai: {msg.content}"
        for i, msg in enumerate(recent_messages)
    )

    # 사용자의 질문에 '요약'이라는 단어가 있는지 확인
    if "요약" in user_input:
        info = ""  # 벡터 DB를 사용하지 않고 빈값으로 설정
    else:
        # 사용자의 질문을 바탕으로 정보 검색
        results = retriever.get_relevant_documents(user_input)
        info = "\n".join(result.metadata["text"] for result in results)

    # 이전 대화 내용과 함께 최종 프롬프트 생성
    final_prompt = f"{previous_messages}\n\n{prompt.format(query=user_input, info=info)}"

    # # LLM의 스트리밍 기능 사용
    response = ""
    for chunk in llm.stream(final_prompt):
        response += chunk
        response_placeholder.write(response)

    # 최종 응답 저장 및 화면에 출력
    st.session_state['messages'].append(ChatMessage(role="assistant", content=response))
    st.session_state['chat_history'].add_ai_message(response)
    response_placeholder.empty()
    st.chat_message("assistant").write(response)