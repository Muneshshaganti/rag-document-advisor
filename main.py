import os
import re
import streamlit as st

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

# -----------------------------
# CLEAN OCR TEXT
# -----------------------------
def clean_text(text):

    text = text.replace("|", " ")
    text = re.sub(r'[^a-zA-Z0-9\s:/.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# OCR PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):

    pages = convert_from_path(pdf_path)

    data = []

    for i, page in enumerate(pages):

        raw_text = pytesseract.image_to_string(page)
        text = clean_text(raw_text)

        data.append({
            "text": text,
            "page": i + 1
        })

    return data


# -----------------------------
# CHUNKING
# -----------------------------
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


# -----------------------------
# CREATE RECORDS
# -----------------------------
def create_records(ocr_data):

    records = []

    for doc in ocr_data:

        page = doc["page"]
        text = doc["text"]

        chunks = layout_chunking(text, page)

        records.extend(chunks)

    return records


# -----------------------------
# VECTOR DB
# -----------------------------
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


# -----------------------------
# BM25
# -----------------------------
def build_bm25(records):

    corpus = [r["content"] for r in records]
    tokenized = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    return bm25


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def hybrid_search(query, vector_db, bm25, records):

    vector_docs = vector_db.similarity_search(query, k=3)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    keyword_docs = [records[i] for i in top_n]

    return vector_docs, keyword_docs


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate.from_template("""
You are a document assistant.

Answer ONLY using the provided context.

If the answer is not present in the context reply:
"Not enough information in the document."

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()

chain = prompt | llm | parser


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📄 Legal Document RAG Chatbot")

st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# -----------------------------
# PROCESS PDF
# -----------------------------
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

    # -----------------------------
    # QUESTION INPUT
    # -----------------------------
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