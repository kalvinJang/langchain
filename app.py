import traceback
import streamlit as st
from funcy import chunks
from llm_agent import OpenAIChatAgent, GenFAQsLLM
from langchain.document_loaders import PyPDFLoader
from config import OPENAI_API_KEY
from suggestion_prompt import FAQ_prompt
from langchain.chains import LLMChain
from langchain.prompts.chat import (SystemMessagePromptTemplate, AIMessagePromptTemplate,
                                    HumanMessagePromptTemplate, ChatPromptTemplate)


N_FAQS = st.sidebar.number_input("Number of FAQs", min_value=1, max_value=10, value=4)
openai_model_name = "gpt-3.5-turbo-16k"

st.title('🦜🔗 Langchain TEST')
pdf_file = st.file_uploader("Upload Files", type=['pdf'])
if pdf_file:
    bytes_data = pdf_file.getvalue()
    file_url = f"{pdf_file.name}"
    with open(file_url, "wb") as f:
        f.write(bytes_data)
        
    test_scope = (6, 12)
    docs = PyPDFLoader(file_url).load()
    doc = '\\n'.join([page.page_content for page in docs[test_scope[0]:test_scope[1]]])

# session에 대화 내역을 저장할 messages 생성
if "messages" not in st.session_state:
    st.session_state.messages = []
# session에 있는 메세지 업데이트
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try: # langchain chain, agent 객체 생성
    llm_agent = OpenAIChatAgent()
    gen_faq_llm = GenFAQsLLM(llm_temp=openai_model_name)
except Exception as e:
    st.error(f"Error initializing agent...\n\n{traceback.format_exc()}\n{e.__class__.__name__}: {e}")
    st.stop()

# 대화 입력 공간 생성
input_content = st.chat_input("What is up?")
if "clicked_faq" in st.session_state:
    input_content = st.session_state['clicked_faq']
    del st.session_state['clicked_faq']

else:
    def faq_button_callback(clicked_faq: str):
        st.session_state['clicked_faq'] = clicked_faq
    
    if input_content:
        # session에 input값 업로드
        st.session_state.messages.append({'role':'human', 'content':input_content})
        # human 대상자로 input값 채팅에 업로드
        with st.chat_message('human'):
            st.markdown(input_content)

        # ai 대상자로 agent를 이용하여 llm response를 가져온다
        with st.chat_message('ai'):
            response = llm_agent.run(
                chat_history=st.session_state.messages[:-1],
                human_input=input_content,
                st_gen=st,
                text = doc
            )
            st.markdown(response)
            st.markdown("---")
            st.session_state.messages.append({'role':'ai', 'content':response})

            faqs = gen_faq_llm.run(
                chat_history=st.session_state.messages[:-1],
                text=doc,
                n_faqs=4,
                language='Korean'
            )

            btn_id = 1
            n_cols = 2
            for faqs in chunks(n_cols, faqs):
                cols = st.columns([1]* n_cols)
                for col, faq in zip(cols, faqs):
                    col.button(label=f"*{btn_id}.* {faq}", key=btn_id, on_click=faq_button_callback, args=(faq,))
                    btn_id +=1