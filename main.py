import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
from chromadb.config import Settings
import os
from tqdm import tqdm

# ChromaDB 설정
settings = Settings(
    persist_directory="./chromadb_data"
)
client = chromadb.PersistentClient(settings=settings)

# SentenceTransformer 모델 초기화
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def embedding_exists(directory):
    """임베딩 데이터가 존재하는지 확인하는 함수."""
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

def insert_data_into_collection(df, model, collection):
    """데이터프레임을 받아 임베딩을 생성하고 ChromaDB 컬렉션에 삽입하는 함수."""
    print("insert_data_into_collection 호출됨")
    ids, metadatas, embeddings = [], [], []

    # 데이터프레임의 각 행에 대해 임베딩 생성
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row['text']
        metadata = {"text": text}
        embedding = model.encode(text, normalize_embeddings=True)
        ids.append(str(index))
        metadatas.append(metadata)
        embeddings.append(embedding.tolist())  # numpy 배열을 리스트로 변환

    # 청크 단위로 데이터를 컬렉션에 추가
    chunk_size = 1024
    for start_idx in tqdm(range(0, len(embeddings), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(embeddings))
        collection.add(
            embeddings=embeddings[start_idx:end_idx],
            ids=ids[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx]
        )

def create_embeddings(df, model, answers):
    """데이터프레임을 받아 임베딩을 생성하고 ChromaDB 컬렉션에 삽입하는 함수."""
    print("create_embeddings 호출됨")
    ids, metadatas, embeddings = [], [], []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row['text']
        metadata = {"text": text}
        embedding = model.encode(text, normalize_embeddings=True)
        ids.append(str(index))
        metadatas.append(metadata)
        embeddings.append(embedding.tolist())  # numpy 배열을 리스트로 변환

    # 청크 단위로 데이터를 컬렉션에 추가
    chunk_size = 1024
    for start_idx in tqdm(range(0, len(embeddings), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(embeddings))
        answers.add(
            embeddings=embeddings[start_idx:end_idx],
            ids=ids[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx]
        )
    answers.persist()  # 데이터 저장

# ChromaDB 컬렉션 설정
collection_name = "test"
collections = client.list_collections()
collection_exists = any(collection.name == collection_name for collection in collections)

if collection_exists:
    print("컬렉션 존재")
    answers = client.get_collection(name=collection_name)
else:
    print("컬렉션 존재하지 않음")
    answers = client.create_collection(name=collection_name)
    df = pd.read_csv("./watermoonInfo.csv")
    insert_data_into_collection(df, model, answers)

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
    with st.spinner("답변 생성 중..."):
        messages = st.session_state['chat_history'].messages
        recent_messages = messages[-10:]

        # 메시지를 문자열로 변환
        previous_messages = "\n".join(
            f"human: {msg.content}\n" for msg in recent_messages[::2]
        ) + "\n" + "\n".join(
            f"ai: {msg.content}\n" for msg in recent_messages[1::2]
        )

        # 쿼리 임베딩 생성
        query_embedding = model.encode(user_input, normalize_embeddings=True).tolist()
        result = answers.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        # 정보 추출
        metadatas = result['metadatas'][0]
        info = "\n".join(str(data) for data in metadatas)

        final_prompt = f"{previous_messages}\n\n{prompt.format(query=user_input, info=info)}"

        # LLM의 스트리밍 기능 사용
        response = ""
        for chunk in llm.stream(final_prompt):
            response += chunk
            response_placeholder.write(response)

    # 최종 응답 저장 및 화면에 출력
    st.session_state['messages'].append(ChatMessage(role="assistant", content=response))
    st.session_state['chat_history'].add_ai_message(response)
    response_placeholder.empty()
    st.chat_message("assistant").write(response)