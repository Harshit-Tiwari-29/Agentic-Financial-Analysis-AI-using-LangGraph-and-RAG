import os, uuid
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from tools import get_rag_tool, get_stock_data_tool, get_prediction_tool
from graph import build_postprocessing_graph

# ---- ENV ----
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ---- APP ----
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory="templates")

# ---- STATE ----
sessions = {}
post_graph = build_postprocessing_graph()

# ---- LLM ----
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0
    )

PROMPT = PromptTemplate.from_template("""
You are a financial assistant.

Tools:
{tools}

Question: {input}
{agent_scratchpad}
""")

# ---- ROUTES ----
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/session")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"db": None, "agent": None}
    return {"session_id": sid}

@app.post("/upload_report")
async def upload_pdf(file: UploadFile, session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(400, "Invalid session_id")

    reader = PdfReader(file.file)
    text = "".join(p.extract_text() or "" for p in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embeddings)

    llm = get_llm()
    tools = [get_rag_tool(db), get_stock_data_tool(), get_prediction_tool()]
    agent = create_react_agent(llm, tools, PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    sessions[session_id]["db"] = db
    sessions[session_id]["agent"] = executor

    return {"message": "PDF processed"}

@app.post("/query_agent")
async def query_agent(session_id: str = Form(...), user_query: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(400, "Invalid session_id")

    executor = sessions[session_id]["agent"]
    if executor is None:
        raise HTTPException(400, "Upload a PDF first")

    result = executor.invoke({"input": user_query})
    raw_output = result["output"]

    final_state = post_graph.invoke({
        "user_query": user_query,
        "raw_tool_output": raw_output,
        "analyst_notes": None,
        "verified_output": None,
        "final_answer": None
    })

    return {"answer": final_state["final_answer"]}

