Agentic Financial Analysis AI (LangGraph + RAG)

A full-stack, session-based Financial Analysis AI application built with FastAPI, LangChain / LangGraph, and Retrieval-Augmented Generation (RAG).
The system allows users to upload financial reports (PDFs), ask document-specific questions, fetch live stock data, and run basic price predictions through an agentic workflow.

🚀 Overview

This project implements a single AI agent orchestrated via LangGraph, capable of:

Answering questions from uploaded financial PDFs using RAG

Fetching real-time stock data via tools

Running basic price predictions using a lightweight ML model

Maintaining session-isolated state so each user’s uploaded data is private

The backend is built with FastAPI, while the frontend uses HTML, CSS, and vanilla JavaScript for a simple chat-based interface.

✨ Key Features
🔹 Agentic Reasoning (LangGraph)

Uses an agent graph instead of a monolithic chain

Enables structured reasoning → tool use → response

Clean separation between analysis, verification, and final response

🔹 Retrieval-Augmented Generation (RAG)

Upload a PDF financial report

Text is extracted, chunked, embedded, and stored in FAISS

Queries are answered only from the uploaded document

🔹 Financial Tools

Live stock data using yfinance

Price prediction using a simple regression model (educational purpose)

🔹 Session-Based Architecture

Each user gets a unique session_id

Vector store and agent executor are stored per session

Prevents data leakage between users

🔹 Lightweight Frontend

Simple chat UI

File upload + text input

No frontend frameworks → easy to debug and extend

```🧱 Tech Stack
Backend

FastAPI – API server and routing

LangChain + LangGraph – agent orchestration

Groq (llama3-8b-8192) – fast LLM inference

FAISS – in-memory vector store

Sentence-Transformers – text embeddings

yfinance – stock market data

scikit-learn – price prediction model

Frontend

HTML

CSS

Vanilla JavaScript 
```

```
📂 Project Structure
project-root/
│
├── main.py                # FastAPI app + session handling
├── agents.py              # Agent construction logic
├── graph.py               # LangGraph workflow
├── graph_state.py         # Agent state definitions
├── tools.py               # PDF, stock, and prediction tools
├── ml_model.py            # Price prediction model
├── requirements.txt
├── .env                   # Environment variables
│
├── templates/
│   └── index.html         # Frontend UI
│
└── static/
    ├── script.js          # Frontend logic
    ├── styles.css         # UI styling
    └── favicon.ico
```

⚙️ Installation & Setup
1️⃣ Prerequisites
```
Python 3.8+
Git
```

2️⃣ Clone Repository
```
git clone <your-repository-url>
cd <repository-directory>
```
3️⃣ Create Virtual Environment
# Windows
```
python -m venv venv
.\venv\Scripts\activate
```
# macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```
4️⃣ Install Dependencies
```
pip install -r requirements.txt
```
5️⃣ Environment Variables

Create a .env file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
▶️ Running the Application

```
uvicorn main:app --reload
```

Open in browser:
```
http://127.0.0.1:8000
```
🧪 How to Use
Step 1: Upload PDF

Upload a financial report in PDF format

The system processes and indexes it for RAG

Step 2: Ask Questions

Examples:

From document:
What were the total revenues mentioned in the report?

Live stock data:
What is the current price of AAPL?

Prediction:
Predict the stock price for NVDA

🧠 Agent Design
Agent Workflow

The agent follows a ReAct-style loop implemented via LangGraph:

Analyze user query

Decide whether a tool is needed

Call the appropriate tool

Observe results

Produce final answer

Prompt Philosophy

The agent is explicitly instructed to:

Ask for missing information (e.g., missing ticker)

Avoid hallucination

Use tools only when necessary

⚠️ Important Notes & Limitations

Price prediction is simplistic and for demonstration only

FAISS is in-memory → sessions reset on server restart

Not production-hardened (no auth, no persistence)

Designed for learning, demos, and experimentation


📜 License

MIT License — free to use, modify, and distribute.

🙌 Acknowledgements

Groq – ultra-fast LLM inference

LangChain / LangGraph – agent frameworks

FastAPI – backend framework
