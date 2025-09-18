# main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# Import tools
from tools import get_rag_tool, get_stock_data_tool, get_prediction_tool

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Financial Analysis AI Agent",
    description="An AI agent that can analyze financial reports, fetch stock data, and predict future prices.",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}

# --- AI Agent Setup ---
def get_llm():
    """Initializes and returns the Groq LLM."""
    return ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0)

# --- Modified Prompt Template to handle missing information ---
# This custom prompt explicitly tells the agent how to act when it's missing required information.
PROMPT_TEMPLATE = """
You are a helpful financial assistant. Respond to the user's questions as best as you can.

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the user, or if you need to ask for more information, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response or question to the user]
```

**CRITICAL RULE:** If you need more information from the user to use a tool (e.g., a stock ticker symbol), you MUST stop and ask the user for it by using the "Final Answer" format. Do not try to guess or make up information.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serves the favicon."""
    return FileResponse("static/favicon.ico")

@app.post("/session")
async def create_session():
    """Creates a new user session and initializes memory."""
    session_id = str(uuid.uuid4())
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    sessions[session_id] = {"db": None, "agent_executor": None, "memory": memory}
    print(f"New session created: {session_id}")
    return {"session_id": session_id}

@app.post("/upload_report")
async def upload_pdf(file: UploadFile, session_id: str = Form(...)):
    """Handles PDF upload and creates the agent executor for the session."""
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_reader = PdfReader(file.file)
        full_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

        if not full_text.strip():
            return {"message": "Could not extract text from the PDF."}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_text(full_text)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_texts(texts, embedding=embeddings)
        sessions[session_id]["db"] = db

        llm = get_llm()
        tools = [get_rag_tool(db), get_stock_data_tool(), get_prediction_tool()]
        agent = create_react_agent(llm, tools, prompt)
        memory = sessions[session_id]["memory"]
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            max_iterations=7,
            handle_parsing_errors="I'm having trouble understanding that. Could you please rephrase your question?"
        )
        sessions[session_id]["agent_executor"] = agent_executor
        
        print(f"Agent created for session: {session_id}")
        return {"message": f"Successfully processed '{file.filename}'"}

    except Exception as e:
        print(f"Error processing PDF for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/query_agent")
async def query_agent(session_id: str = Form(...), user_query: str = Form(...)):
    """Queries the pre-existing AI agent for the session."""
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")

    agent_executor = sessions[session_id].get("agent_executor")
    
    if not agent_executor:
        llm = get_llm()
        tools = [get_rag_tool(None), get_stock_data_tool(), get_prediction_tool()]
        agent = create_react_agent(llm, tools, prompt)
        memory = sessions[session_id]["memory"]
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            max_iterations=7,
            handle_parsing_errors="I'm having trouble understanding that. Could you please rephrase your question?"
        )
        sessions[session_id]["agent_executor"] = agent_executor

    try:
        result = agent_executor.invoke({
            "input": user_query,
            "chat_history": sessions[session_id]["memory"].chat_memory.messages
        })
        return {"answer": result.get("output", "No answer could be generated.")}
    except Exception as e:
        print(f"Agent execution error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
