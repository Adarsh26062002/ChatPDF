from typing import List, TypedDict
from pydantic import BaseModel
import re

import os, tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

def _get(key: str, default=None):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

CHROMA_DIR = _get('CHROMA_DIR', 'chroma_db')
    
def load_llm():
    primary = ChatMistralAI(model='mistral-small-latest', temperature=0.0)
    fallback = ChatGroq(model='openai/gpt-oss-120b', temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    secondary_llm = ChatGoogleGenerativeAI(model='gemini-flash-lite-latest', temperature=0.0)
    tertiary_llm = ChatMistralAI(model='mistral-small-latest', temperature=0.0, api_key=_get('MISTRAL_API_KEY_2'))
    llm = primary.with_fallbacks([fallback])
    scorer_llm = tertiary_llm.with_fallbacks([secondary_llm])
    return embeddings, llm, scorer_llm

def process_pdfs(uploaded_files, embeddings):
    all_chunks = []
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.page_content = chunk.page_content.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            all_chunks.extend(chunks)
        except Exception:
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    if not all_chunks:
        return None, 0
    
    if os.path.exists(CHROMA_DIR):
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        vector_store.add_documents(documents=all_chunks)
    else:
        vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5}), len(all_chunks)

embeddings, llm, scorer_llm = load_llm()

def get_retriever(embeddings):
    if not os.path.exists(CHROMA_DIR):
        return None
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

retriever = get_retriever(embeddings)

UPPER_TH = 0.7
LOWER_TH = 0.3

class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_docs: List[Document]
    web_query: str
    answer: str

def retrieve_node(state: State) -> State:
    q = state['question']
    return {'docs': retriever.invoke(q)}

class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ('system',
    'You are a strict retrieval evaluator for RAG.\n'
    'You will be given ONE retrieved chunk and a question.\n'
    'Return a relevance score in [0.0, 1.0].\n'
    '- 1.0: chunk alone is sufficient to answer fully/mostly\n'
    '- 0.0: chunk is irrelevant\n'
    'Be conservative with high scores.\n'
    'Also return a short reason.\n'
    'Output JSON only.'
    ),
    ('human', 'Question: {question}\n\nChunk:\n{chunk}')
])

doc_eval_chain = doc_eval_prompt | scorer_llm.with_structured_output(DocEvalScore)

def eval_each_doc(state: State) -> State:
    q = state['question']
    scores: list[float] = []
    good_docs: list[Document] = []

    for d in state['docs']:
        out = doc_eval_chain.invoke({'question': q, 'chunk': d.page_content})
        scores.append(out.score)

        if out.score >= UPPER_TH:
            good_docs.append(d)
        
    if any(s >= UPPER_TH for s in scores):
        return {
            'good_docs': good_docs,
            'verdict': 'CORRECT',
            'reason': f'At least one retrieved chunk scored > {UPPER_TH}.'
        }

    why = 'Mixed relevance signals.'
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            'good_docs': [],
            'verdict': 'INCORRECT',
            'reason': f'All retrieved chunks stored < {LOWER_TH}. {why}'
        }

    return {
        'good_docs': [],
        'verdict': 'AMBIGUOUS',
        'reason': f'No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}'
    }
    
def decompose_to_sentence(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ' , text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 5]

class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages([
    ('system', 
    'You are a relevance filter for a RAG system.\n'
    'Return keep=true if the sentence is related to the question.\n'
    'Background, definitions, context, and related facts should be kept.\n'
    'Only drop sentences that are completely unrelated to the topic.\n'
    'Use ONLY the sentence to judge. Output JSON only.',
    ),
    ('human', 'Question: {question}\n\nSentence:\n{sentence}')
])

filter_chats = filter_prompt | scorer_llm.with_structured_output(KeepOrDrop)

def refine(state: State) -> State:
    q = state['question']
    
    if state.get('verdict') == 'CORRECT':
        docs_to_use = state['good_docs']
    elif state.get('verdict') == 'INCORRECT':
        docs_to_use = state['web_docs']
    else:
        docs_to_use = state['good_docs'] + state['web_docs']

    context = '\n\n'.join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentence(context)
    
    kept: List[str] = []
    for s in strips:
        if filter_chats.invoke({'question': q, 'sentence': s}).keep:
            kept.append(s)
    refined_context = '\n'.join(kept).strip()

    return {
        'strips': strips,
        'kept_strips': kept,
        'refined_context': refined_context
    }

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ('system',
    'Rewrite the user question into a web search query composed of keywords.\n'
    'Rules:\n'
    '- Keep it short (6-14 words).\n'
    '- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last week) to the query.\n'
    '- Do NOT answer the question.\n'
    '- Return JSON with a single key: query'
    ),
    ('human', 'Question: {question}'),
])

rewrite_chain = rewrite_prompt | scorer_llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({'question': state['question']})
    return {'web_query': out.query}

tavily = None

def get_tavily():
    global tavily
    if tavily is None:
        tavily = TavilySearchResults(max_results=5)
    return tavily

def web_search_node(state: State) -> State:
    q = state.get('web_query') or state['question']
    tool = get_tavily()
    results = tool.invoke({'query': q})

    web_docs = []
    for r in results or []:
        title = r.get('title', '')
        url = r.get('url', '')
        content = r.get('content', '') or r.get('snippet', '')

        text = f'TITLE: {title}\nURL: {url}\nCONTENT:\n{content}'
        web_docs.append(Document(page_content=text, metadata={'url': url, 'title': title}))
    return {'web_docs': web_docs}

answer_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        'You are a helpful assistant.\n'
        'Use the provided context to answer the question.\n'
        'Synthesize the information into a clear, concise answer.\n'
        'Do not copy-paste from the context.\n'
        "If the context is empty or insufficient, say: 'I don't know.'",
    ),
    ('human', 'Question: {question}\n\nContext:\n{refined_context}')
])

def generate(state: State) -> State:
    ctx = state['refined_context']
    if not ctx and state.get('web_docs'):
        ctx = '\n\n'.join(d.page_content for d in state['web_docs']).strip()
    out = (answer_prompt | llm).invoke({'question': state['question'], 'refined_context': ctx})
    return {'answer': out.content}

def route_after_eval(state: State) -> str:
    return 'refine' if state['verdict'] == 'CORRECT' else 'rewrite_query'


g = StateGraph(State)

g.add_node('retrieve', retrieve_node)
g.add_node('eval_each_doc', eval_each_doc)
g.add_node('rewrite_query', rewrite_query_node)
g.add_node('web_search', web_search_node)
g.add_node('refine', refine)
g.add_node('generate', generate)

g.add_edge(START, 'retrieve')
g.add_edge('retrieve', 'eval_each_doc')
g.add_conditional_edges(
    'eval_each_doc',
    route_after_eval,
    {'refine': 'refine', 'rewrite_query': 'rewrite_query'}
)
g.add_edge('rewrite_query', 'web_search')
g.add_edge('web_search', 'generate')
g.add_edge('refine', 'generate')
g.add_edge('generate', END)

app = g.compile()


def query_rag(question: str) -> str:
    if retriever is None:
        return 'No documents loaded. Please upload a PDF first.'
    result = app.invoke({
        'question': question,
        'docs': [],
        'good_docs': [],
        'verdict': '',
        'reason': '',
        'strips': [],
        'kept_strips': [],
        'refined_context': '',
        'web_docs': [],
        'web_query': '',
        'answer': '',
    })
    return result['answer']


def query_rag_with_meta(question: str) -> dict:
    if retriever is None:
        return {'answer': 'No documents loaded. Please upload a PDF first.', 'verdict': 'NONE', 'reason': ''}
    result = app.invoke({
        'question': question,
        'docs': [],
        'good_docs': [],
        'verdict': '',
        'reason': '',
        'strips': [],
        'kept_strips': [],
        'refined_context': '',
        'web_docs': [],
        'web_query': '',
        'answer': '',
    })
    return {'answer': result['answer'], 'verdict': result['verdict'], 'reason': result['reason']}


def split_questions(text: str) -> list[str]:
    parts = re.split(r'\?\s+', text.strip())
    questions = [p.strip() + '?' for p in parts if p.strip()]
    if questions and not text.strip().endswith('?'):
        questions[-1] = questions[-1].rstrip('?')
    return questions if questions else [text.strip()]


def query_rag_auto(question: str) -> str:
    questions = split_questions(question)
    if len(questions) == 1:
        return query_rag_with_meta(questions[0])['answer']
    results = []
    for q in questions:
        r = query_rag_with_meta(q)
        results.append(f"**{q}**\n\n{r['answer']}")
    return '\n\n---\n\n'.join(results)