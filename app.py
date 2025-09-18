# import os
# import yfinance as yf
# import numpy as np
# import shutil
# import uuid
# from fastapi import FastAPI, UploadFile, File, HTTPException, Header
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.agents import tool, AgentExecutor, create_react_agent
# from langchain_core.prompts import PromptTemplate
# from pydantic import BaseModel
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # --- Configuration and Initialization ---
# if os.getenv("GROQ_API_KEY") is None:
#     raise EnvironmentError(
#         "GROQ_API_KEY not set in .env file. Please get a key from https://console.groq.com/keys"
#     )

# app = FastAPI(
#     title="Full-Stack AI Financial Analyst Agent (Session-Based)",
#     description="A robust AI agent that handles user-specific sessions for financial analysis.",
#     version="2.0.0",
# )

# # --- CORS Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Session Management ---
# # This dictionary will hold the state for each user session.
# # Key: session_id (str), Value: dict containing 'vector_store' and 'agent_executor'
# SESSIONS = {}

# # --- Agent Tools ---
# # Note: Tools are now defined within functions that can access session-specific data.

# def create_financial_document_qa_tool(session_id: str):
#     @tool
#     def financial_document_qa(query: str):
#         """
#         Answers questions about the financial document uploaded in the current session.
#         Use this to find information about company strategy, risks, financial results, etc.
#         """
#         if session_id not in SESSIONS or "vector_store" not in SESSIONS[session_id]:
#             return "No financial document has been uploaded for this session yet. Please upload a document first."
        
#         vector_store = SESSIONS[session_id]["vector_store"]
#         retriever = vector_store.as_retriever()
#         relevant_docs = retriever.invoke(query)
#         context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
#         prompt = f"Based on the following context, answer the user's question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
#         llm = ChatGroq(model="gemma-7b-it", temperature=0)
#         response = llm.invoke(prompt)
#         return response.content
#     return financial_document_qa

# @tool
# def get_stock_data(ticker: str):
#     """
#     Fetches real-time stock data, including price, news, and key financial metrics for a given stock ticker.
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         key_info = {
#             "companyName": info.get("longName", "N/A"),
#             "currentPrice": info.get("currentPrice", "N/A"),
#             "marketCap": info.get("marketCap", "N/A"),
#         }
#         news = stock.news[:3]
#         return f"Real-time data for {ticker}:\nKey Info: {key_info}\nRecent News: {news}"
#     except Exception as e:
#         return f"Could not fetch data for ticker {ticker}. Error: {str(e)}"

# @tool
# def mock_price_forecast(ticker: str):
#     """
#     Generates a simplified, mock price forecast for a given stock ticker for the next quarter.
#     """
#     try:
#         hist = yf.Ticker(ticker).history(period="1y")
#         if hist.empty:
#             return "Could not generate forecast: No historical data."
#         last_price = hist['Close'].iloc[-1]
#         trend = np.random.uniform(-0.05, 0.10)
#         projected_price = last_price * (1 + trend)
#         return f"Mock Forecast for {ticker}: Projected price around ${projected_price:.2f} next quarter. (Demonstration only)."
#     except Exception as e:
#         return f"Could not generate forecast for {ticker}. Error: {str(e)}"

# # --- Agent Initialization ---
# def initialize_agent(session_id: str):
#     """Initializes or retrieves an agent for a specific session."""
#     if session_id in SESSIONS and "agent_executor" in SESSIONS[session_id]:
#         return SESSIONS[session_id]["agent_executor"]

#     tools = [create_financial_document_qa_tool(session_id), get_stock_data, mock_price_forecast]
    
#     prompt_template = """
#     You are a helpful Financial Analyst AI Agent. Answer the user's question based on the tools available.

#     Tools: {tools}

#     Use the following format:
#     Thought: Do I need to use a tool? Yes
#     Action: The action to take, one of [{tool_names}]
#     Action Input: The input to the action
#     Observation: The result of the action

#     (this Thought/Action/Action Input/Observation can repeat N times)

#     Thought: I now know the final answer
#     Final Answer: [your response here]

#     Begin!
#     Question: {input}
#     Thought: {agent_scratchpad}
#     """
    
#     prompt = PromptTemplate.from_template(prompt_template)
#     llm = ChatGroq(model_name="gemma-7b-it", temperature=0)
    
#     agent = create_react_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
    
#     if session_id not in SESSIONS:
#         SESSIONS[session_id] = {}
#     SESSIONS[session_id]["agent_executor"] = agent_executor
    
#     return agent_executor

# # --- FastAPI Endpoints ---

# @app.get("/session", tags=["Session Management"])
# async def get_session():
#     """Establishes a new user session and returns a unique session ID."""
#     session_id = str(uuid.uuid4())
#     SESSIONS[session_id] = {}
#     print(f"New session created: {session_id}")
#     return JSONResponse({"session_id": session_id})

# @app.post("/upload_report", tags=["Document Processing"])
# async def upload_report(session_id: str = Header(...), file: UploadFile = File(...)):
#     """Uploads a PDF, processes it, and stores the vector store in the user's session."""
#     if not session_id or session_id not in SESSIONS:
#         raise HTTPException(status_code=400, detail="Invalid or missing session ID.")
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")

#     temp_file_path = f"temp_{session_id}_{file.filename}"
#     try:
#         with open(temp_file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         loader = PyPDFLoader(temp_file_path)
#         documents = loader.load()
#         if not documents:
#             raise HTTPException(status_code=500, detail="Could not extract text from the PDF.")
            
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
#         chunked_documents = text_splitter.split_documents(documents)
        
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
#         vector_store = FAISS.from_documents(chunked_documents, embeddings)
#         SESSIONS[session_id]["vector_store"] = vector_store
        
#         print(f"Vector store created for session: {session_id}")
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
#     finally:
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

#     return JSONResponse({"message": f"Successfully processed '{file.filename}'."})

# class QueryRequest(BaseModel):
#     query: str
#     ticker: str | None = None

# @app.post("/query_agent", tags=["AI Agent"])
# async def query_agent(request: QueryRequest, session_id: str = Header(...)):
#     """Sends a query to the session-specific AI agent."""
#     if not session_id or session_id not in SESSIONS:
#         raise HTTPException(status_code=400, detail="Invalid or missing session ID.")

#     agent_executor = initialize_agent(session_id)
    
#     agent_input = f"User Query: '{request.query}'"
#     if request.ticker:
#         agent_input += f" The user is also interested in the stock ticker: {request.ticker}."

#     try:
#         response = agent_executor.invoke({"input": agent_input})
#         return JSONResponse({"response": response.get("output")})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agent failed to process query: {str(e)}")

# # --- Frontend Serving ---
# # This part tells FastAPI where to find your static files (like CSS, JS, and index.html)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", include_in_schema=False)
# async def root():
#     """Serves the main frontend application from the static directory."""
#     # This is the corrected line
#     return FileResponse('static/index.html')


# app.py
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
