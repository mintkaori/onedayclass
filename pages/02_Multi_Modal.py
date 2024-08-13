import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os

# API KEY 정보로드
# load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("수학 교과서 채점 봇 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 이미지 업로드 (여러 파일 가능)
    uploaded_files = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
        "시스템 프롬프트",
        "당신은 초등학교 수학 선생님입니다. 사진을 업로드하면 파란색 선으로 10개씩 잘 묶었는지 확인하고 잘 묶었다면 10개씩 잘 묶었다고 칭찬해주세요.",
        height=200,
    )

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 이미지 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

# 체인 생성
def generate_answer(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,  # 모델명
        openai_api_key = st.session_state.api_key
    )

    # 멀티모달 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 파일로 부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)
    return answer

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["uploaded_files"] = []

# 페이지 레이아웃을 두 열로 나누기
col1, col2 = st.columns([1, 2])

# 왼쪽 열에 이미지 표시
with col1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 이미지 파일을 처리
            image_filepath = process_imagefile(uploaded_file)
            st.image(image_filepath, use_column_width=True)
            st.session_state["uploaded_files"].append(image_filepath)
    else:
        st.write("이미지를 업로드 해주세요.")

# 오른쪽 열에 대화내용 표시
with col2:
    # 이전 대화 기록 출력
    print_messages()

    # 사용자의 입력
    user_input = st.chat_input("궁금한 내용을 물어보세요!")

    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    # 만약에 사용자 입력이 들어오면...
    if user_input:
        if st.session_state["uploaded_files"]:
            for image_filepath in st.session_state["uploaded_files"]:
                # 답변 요청
                response = generate_answer(
                    image_filepath, system_prompt, user_input, selected_model
                )

                # 사용자의 입력
                st.chat_message("user").write(user_input)

                with st.chat_message("assistant"):
                    # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                    container = st.empty()

                    ai_answer = ""
                    for token in response:
                        ai_answer += token.content
                        container.markdown(ai_answer)

                # 대화기록을 저장한다.
                add_message("user", user_input)
                add_message("assistant", ai_answer)
        else:
            # 이미지를 업로드 하라는 경고 메시지 출력
            warning_msg.error("이미지를 업로드 해주세요.")
