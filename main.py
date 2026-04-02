import os
import re
import streamlit as st
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pdf2image import convert_from_path                                 
import pytesseract

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


 # LOAD EMBEDDINGS (CACHE)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

embeddings = load_embeddings()


# CLEAN OCR TEXT

def clean_text(text):

    text = text.replace("|", " ")
    text = re.sub(r'[^a-zA-Z0-9\s:/.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# OCR PDF

def extract_text_from_pdf(pdf_path):

    pages = convert_from_path(
    pdf_path,
    dpi=150
)

    data = []

    for i, page in enumerate(pages):

        raw_text = pytesseract.image_to_string(page, lang="eng", config="--psm 6")  
        text = clean_text(raw_text)

        data.append({
            "text": text,
            "page": i + 1
        })

    return data


# CHUNKING

def layout_chunking(text, page):

    sections = re.split(r'\n\s*\d+\.\s+', text)

    records = []

    for sec in sections:

        sec = sec.strip()

        if len(sec) > 80:

            records.append({
                "content": sec,
                "page": page
            })

    return records


# CREATE RECORDS

def create_records(ocr_data):

    records = []

    for doc in ocr_data:

        page = doc["page"]
        text = doc["text"]

        chunks = layout_chunking(text, page)

        records.extend(chunks)

    return records


# VECTOR DB

def build_vector_db(records):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    texts = [r["content"] for r in records]
    metadatas = [{"page": r["page"]} for r in records]

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db"
    )

    return db


# BM25

def build_bm25(records):

    corpus = [r["content"] for r in records]
    tokenized = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    return bm25


# CACHE FULL PDF PROCESSING

@st.cache_resource
def process_pdf(pdf_path):

    ocr_data = extract_text_from_pdf(pdf_path)

    records = create_records(ocr_data)

    vector_db = build_vector_db(records)

    bm25 = build_bm25(records)

    return vector_db, bm25, records


# HYBRID SEARCH

def hybrid_search(query, vector_db, bm25, records):

    vector_docs = vector_db.similarity_search(query, k=6)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    keyword_docs = [records[i] for i in top_n]

    return vector_docs, keyword_docs


# LLM

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0
)

prompt = PromptTemplate.from_template("""
### ROLE
You are a Legal Document Assistant. 
Your job is to answer questions strictly using the provided document context.

### INSTRUCTIONS
Follow these instructions carefully:

1. Use ONLY the information present in the provided document context.
2. If the answer is clearly available in the context, provide a concise and accurate answer.
3. If the information is NOT present in the context, respond exactly with:
   Not enough information in the document.
4. Do NOT use external knowledge.
5. Do NOT guess or assume missing information.
6. Do NOT generate explanations that are not directly supported by the context.
7. If the user asks for a summary, summarize ONLY the information from the context.

### OUTPUT RULES
- Answer must be based strictly on the context.
- If information is missing, return:
  Not enough information in the document.

### DOCUMENT CONTEXT
{context}

### USER QUESTION
{question}

### FINAL ANSWER
""")

parser = StrOutputParser()

chain = prompt | llm | parser


# STREAMLIT UI

st.title("📄 Internal Legal Document RAG Assistant")

st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")


# PROCESS PDF

if uploaded_file:

    with st.spinner("Processing PDF..."):

        pdf_path = "temp.pdf"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        ocr_data = extract_text_from_pdf(pdf_path)

        records = create_records(ocr_data)

        vector_db = build_vector_db(records)

        bm25 = build_bm25(records)

    st.success("PDF processed successfully!")

    
    # QUESTION INPUT
    
    question = st.text_input("Enter your question")

    if question:

        vector_docs, keyword_docs = hybrid_search(
            question,
            vector_db,
            bm25,
            records
        )

        context_parts = []
        pages = []

        for d in vector_docs:

            context_parts.append(d.page_content)
            pages.append(d.metadata["page"])

        for d in keyword_docs:

            context_parts.append(d["content"])
            pages.append(d["page"])

        context = "\n\n".join(context_parts)

        pages = sorted(set(pages))

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        st.subheader("Answer")

        st.write(answer)

        if "Not enough information" in answer:
            st.write("Sources: None")
        else:
            st.write("Sources:", pages)