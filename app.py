from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
import pytesseract
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# -------------------------------
# Load API key
# -------------------------------
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
api_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# Tesseract path (Windows)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄 Bangla PDF RAG System")

uploaded_file = st.file_uploader("Upload a Bangla PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save PDF
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Convert PDF → images
        poppler_path = POPPLER_PATH
        images = convert_from_path("temp.pdf", dpi=150, poppler_path=poppler_path)

        # OCR
        pages = []
        for img in images:
            text = pytesseract.image_to_string(img, lang="ben")
            text = text.replace("\n\n", "\n").strip()
            pages.append(text)

        full_text = "\n".join(pages)

        if not full_text.strip():
            st.error("❌ No text could be extracted from this PDF.")
            st.stop()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=150,
            separators=["\n\n", "\n", "।", " ", ""]
        )
        chunks = splitter.split_text(full_text)

        # Embeddings + FAISS
        embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            encode_kwargs={"normalize_embeddings": True}
        )

        vector_store = FAISS.from_texts(
            texts=[f"passage: {c}" for c in chunks],
            embedding=embedding
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key
        )

    st.success("✅ PDF processed successfully!")

  # -------------------------------
# Question Answering
# -------------------------------
query = st.text_input("Ask a question about the PDF")

if query:   # ← query exists only AFTER this line
    docs = retriever.invoke(query)

    if not docs:
        st.warning("No relevant information found in the document.")
        st.stop()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "তুমি একজন সহকারী। নিচের পাঠ্যাংশের উপর ভিত্তি করে প্রশ্নের উত্তর দাও। "
         "নিজে থেকে কোনো তথ্য বানাবে না।\n\nপাঠ্য:\n{context}"),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    with st.spinner("Generating answer..."):
        response = chain.invoke({
            "context": format_docs(docs),
            "question": query
        })

    st.subheader("📌 Answer")
    st.write(response)
