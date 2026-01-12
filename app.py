from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os, shutil, uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/session")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = None
    return {"session_id": session_id}

@app.post("/upload_report")
async def upload_pdf(file: UploadFile, session_id: str = Form(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDFs are supported."}

    contents = PdfReader(file.file)
    full_text = ""
    for page in contents.pages:
        full_text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = text_splitter.split_text(full_text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embedding=embeddings)
    sessions[session_id] = db
    return {"message": f"Successfully processed '{file.filename}'"}

@app.post("/query_agent")
async def query_agent(session_id: str = Form(...), user_query: str = Form(...)):
    db = sessions.get(session_id)
    if not db:
        return {"error": "No document uploaded for this session"}

    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(user_query)
    return {"answer": result}

