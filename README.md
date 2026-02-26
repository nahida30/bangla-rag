📄 Bangla PDF RAG System (OCR + LLM)

A Retrieval-Augmented Generation (RAG) system that allows users to upload Bangla (Bengali) PDF documents, extract text using OCR, store the content in a vector database, and ask questions in Bangla to get accurate answers based only on the uploaded document.

Built with Streamlit, Tesseract OCR, LangChain, FAISS, and Groq LLM.

✨ Features

📄 Upload Bangla PDF files

🔍 OCR-based text extraction using Tesseract (ben)

🧩 Smart text chunking for better retrieval

🧠 Semantic search using multilingual embeddings

🤖 LLM-based question answering (no hallucination)

🖥️ Simple and interactive Streamlit UI

🧠 How It Works (Pipeline)

PDF Upload

User uploads a Bangla PDF via Streamlit

PDF → Image Conversion

pdf2image converts each page into images using Poppler

OCR (Bangla)

pytesseract extracts Bangla text (lang="ben")

Text Chunking

Large text is split into overlapping chunks using RecursiveCharacterTextSplitter

Embedding

Chunks are converted into vectors using
intfloat/multilingual-e5-base

Vector Store

Embeddings are stored in FAISS for fast similarity search

Retrieval

Relevant chunks are retrieved based on user query

Answer Generation

Retrieved context + user question → Groq LLM

Answer is generated only from the document content

🛠️ Tech Stack
Component	Technology
UI	Streamlit
OCR	Tesseract OCR (Bangla)
PDF Processing	pdf2image + Poppler
Embeddings	HuggingFace (multilingual-e5-base)
Vector DB	FAISS
LLM	Groq (LLaMA 3.1)
Framework	LangChain
📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/nahida30/bangla-rag.git
cd bangla-pdf-rag
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate  # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
⚙️ System Requirements
🔹 Install Tesseract OCR (Bangla)

Download: https://github.com/UB-Mannheim/tesseract/wiki

During installation, enable Bangla language

Note the installation path

🔹 Install Poppler (Windows)

Download Poppler for Windows

Extract and note the bin path

🔐 Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\Library\bin

⚠️ Do not push .env to GitHub

▶️ Run the App
streamlit run app.py

Then open:

Upload a Bangla PDF

Wait for OCR & processing

Ask questions in Bangla

Get answers strictly based on the document

🚫 Limitations

OCR quality depends on PDF scan quality

Very large PDFs may take time to process

Handwritten Bangla text is not supported

🚀 Future Improvements

Multi-PDF support

Source citations per answer

PDF text + OCR hybrid mode

Bangla spell correction

Deployment (Docker / HuggingFace Spaces)

👩‍💻 Author

Nahida Zaman Bina
Computer Science / AI Enthusiast
Bangladesh 🇧🇩

⭐ Acknowledgements

Tesseract OCR

HuggingFace

LangChain

Groq

Streamlit
