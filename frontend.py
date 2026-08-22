import streamlit as st
from backend import load_llm, process_pdfs, get_retriever, query_rag_auto
import os

st.set_page_config(
    page_title='ChatPDF',
    page_icon=':material/question_answer:',
    layout='wide',
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; max-width: 850px !important;}
    .step-row {display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;}
    .step-num {display: inline-flex; align-items: center; justify-content: center;
               width: 22px; height: 22px; border-radius: 50%; font-size: 12px;
               font-weight: 600; color: white; flex-shrink: 0;}
    .step-1 {background: #4a90d9;}
    .step-2 {background: #7b61ff;}
    .step-3 {background: #f59e0b;}
    .step-4 {background: #10b981;}
    .step-5 {background: #ef4444;}
    .step-text {font-size: 13px; color: #444;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_models():
    return load_llm()

embeddings, llm, scorer_llm = init_models()


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'retriever_ready' not in st.session_state:
    st.session_state.retriever_ready = get_retriever(embeddings) is not None

has_tavily = bool(os.getenv('TAVILY_API_KEY')) or bool(st.secrets.get('TAVILY_API_KEY'))


with st.sidebar:
    st.markdown('#### :material/description: Documents')

    uploaded_files = st.file_uploader(
        'Upload PDFs',
        type=['pdf'],
        accept_multiple_files=True,
        label_visibility='collapsed',
        help='Upload one or more PDF files to query against',
    )

    if uploaded_files:
        st.caption(f'{len(uploaded_files)} file(s) selected')

        if st.button('Index documents', type='primary', use_container_width=True, icon=':material/sync:'):
            with st.spinner('Reading and indexing...'):
                retriever, chunk_count = process_pdfs(uploaded_files, embeddings)
                if retriever:
                    st.session_state.retriever_ready = True
                    st.toast(f'Indexed {chunk_count} chunks', icon=':material/check_circle:')
                else:
                    st.toast('No readable content found', icon=':material/error:')

    if st.session_state.retriever_ready:
        st.markdown(':green-badge[:material/database: Ready]')
    else:
        st.markdown(':orange-badge[:material/warning: No documents]')

    if not has_tavily:
        st.warning('TAVILY_API_KEY not set — web search fallback disabled.', icon=':material/warning:')

    st.markdown('')
    with st.container(border=True):
        st.markdown('**How it works**')
        st.markdown("""
<div class="step-row"><span class="step-num step-1">1</span><span class="step-text">Retrieval — fetches relevant chunks</span></div>
<div class="step-row"><span class="step-num step-2">2</span><span class="step-text">Evaluation — scores each chunk</span></div>
<div class="step-row"><span class="step-num step-3">3</span><span class="step-text">Routing — decides source</span></div>
<div class="step-row"><span class="step-num step-4">4</span><span class="step-text">Web search — if docs aren't enough</span></div>
<div class="step-row"><span class="step-num step-5">5</span><span class="step-text">Generation — synthesizes answer</span></div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown('# :material/question_answer: ChatPDF', text_alignment='center')
    st.markdown('##### :material/hub: Ask anything about your documents', text_alignment='center')
    st.caption('Upload PDFs in the sidebar, then ask questions here.', text_alignment='center')
    st.markdown('')

    SUGGESTIONS = {
        ':blue[:material/science:]  What is CRAG?': 'What is CRAG and how does it work?',
        ':green[:material/upload:]  How do I start?': 'How do I upload documents and start asking questions?',
        ':violet[:material/quiz:]  What is superconductivity?': 'What is superconductivity?',
    }

    cols = st.columns([1, 2, 1])
    with cols[1]:
        selected = st.pills('Try asking:', list(SUGGESTIONS.keys()), label_visibility='collapsed')
        if selected:
            st.session_state.messages.append({'role': 'user', 'content': SUGGESTIONS[selected]})
            st.rerun()


for msg in st.session_state.messages:
    avatar = ':material/account_circle:' if msg['role'] == 'user' else ':material/auto_awesome:'
    with st.chat_message(msg['role'], avatar=avatar):
        st.markdown(msg['content'])
        verdict = msg.get('verdict', '')
        if verdict == 'CORRECT':
            st.markdown(':green-badge[:material/check_circle: From documents]')
        elif verdict == 'INCORRECT':
            st.markdown(':blue-badge[:material/language: From web search]')
        elif verdict == 'AMBIGUOUS':
            st.markdown(':orange-badge[:material/hybrid: Documents + web]')


if prompt := st.chat_input('Ask me anything...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user', avatar=':material/account_circle:'):
        st.markdown(prompt)

    with st.chat_message('assistant', avatar=':material/auto_awesome:'):
        if not st.session_state.retriever_ready:
            response = 'Please upload and process some PDFs first using the sidebar.'
        else:
            with st.spinner('Thinking...'):
                response = query_rag_auto(prompt)

        st.markdown(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})

st.markdown("""
<style>
    [data-testid="stChatInput"] {border-radius: 20px !important; max-width: 850px !important; margin: 0 auto !important;}
    [data-testid="stChatInput"] textarea {border-radius: 20px !important; padding-left: 24px !important; max-height: 120px !important; overflow-y: auto !important;}
    [data-testid="stChatInput"] > div {border-radius: 20px !important;}
    [data-testid="stChatInput"] button {border-radius: 16px !important;}
</style>
""", unsafe_allow_html=True)
