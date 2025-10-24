# ✅ 정리된 import (중복 없이 정확한 버전)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import tempfile
import os

def search_web():
    search = SerpAPIWrapper()
    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."
    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )


def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # PDF 로드
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # ✅ 텍스트를 더 잘게 분할 (기존보다 작게)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_documents)

    st.info(f"🔍 총 {len(split_docs)}개의 문서 청크로 분할되었습니다.")

    embeddings = OpenAIEmbeddings()
    vectors = []
    metadatas = []

    # ✅ 배치 단위로 안전하게 임베딩 (한 번에 너무 많이 안 보냄)
    batch_size = 80
    for i in tqdm(range(0, len(split_docs), batch_size), desc="Embedding in batches"):
        batch = split_docs[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        meta = [doc.metadata for doc in batch]

        try:
            batch_embeddings = embeddings.embed_documents(texts)
            vectors.extend(batch_embeddings)
            metadatas.extend(meta)
        except Exception as e:
            st.error(f"⚠️ 임베딩 중 오류 발생 (배치 {i // batch_size + 1}): {e}")
            continue

    # ✅ FAISS 벡터스토어 생성
    vector = FAISS.from_embeddings(
        embeddings=vectors,
        metadatas=metadatas,
        embedding=embeddings
    )

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the uploaded PDF documents."
    )

    return retriever_tool


# ✅ Agent 대화 실행
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result['output']

# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# ✅ 메인 실행
def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        st.image('./chatbot_logo.png', use_container_width=True)
        st.markdown('---')
        st.title("안녕하세요! RAG를 활용한 'AI 비서 톡톡이' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)
        tools.append(search_web())

        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                "Be sure to answer in Korean. You are a helpful assistant. "
                "Make sure to use the `pdf_search` tool for searching information from the pdf document. "
                "If you can't find the information from the PDF document, use the `web_search` tool for searching information from the web. "
                "If the user’s question contains words like '최신', '현재', or '오늘', you must ALWAYS use the `web_search` tool to ensure real-time information is retrieved. "
                "Please always include emojis in your responses with a friendly tone. "
                "Your name is `AI 비서 톡톡이`. Please introduce yourself at the beginning of the conversation."),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )


        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input('질문이 무엇인가요?')

        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            if session_history.messages:
                prev_msgs = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_message({"role": "user", "content": user_input})
            session_history.add_message({"role": "assistant", "content": response})

        print_messages()

    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
